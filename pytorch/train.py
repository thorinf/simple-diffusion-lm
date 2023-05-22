import os
import math
import random
import argparse
from typing import List

from tqdm import tqdm
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rotary_embedding_torch import RotaryEmbedding


def get_text(path: str) -> str:
    with open(path, "r", encoding='utf-8') as file:
        return file.read()


def get_line_offsets(path: str, chunk_size: int = 2 ** 20) -> List[int]:
    offsets = [0]
    with open(path, "rb") as file:
        chunk = file.readlines(chunk_size)
        while chunk:
            for line in chunk:
                offsets.append(offsets[-1] + len(line))
            print(f"Lines found: {len(offsets)}", end='\r')
            chunk = file.readlines(chunk_size)
    return offsets


class SentencePieceTokenizer:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def __len__(self):
        return len(self.sp)

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    def encode(self, text):
        return self.sp.encode(text, enable_sampling=True, alpha=0.1, nbest_size=5)

    def decode(self, encoded):
        return self.sp.decode(encoded)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: SentencePieceTokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.offsets = get_line_offsets(path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, 'r', encoding='utf-8') as file:
            file.seek(self.offsets[idx])
            text = file.readline().strip()
        ids = self.tokenizer.encode(text)
        return ids


class Collate:
    def __init__(self, crop_length=-1, eos_id=-1, pad_id=-1, length_includes_pad=False, fold_size=None):
        assert not (pad_id < 0 and length_includes_pad)
        assert not (pad_id < 0 and fold_size)
        self.crop_length = crop_length
        self.fold_size = fold_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.pad_insert_rate = 0.0
        self.length_includes_pad = length_includes_pad

    def fold(self, ids):
        # pad the list for folding
        remainder = len(ids) % self.fold_size
        if remainder != 0:
            ids += [self.pad_id] * (self.fold_size - remainder)
        # fold the list
        ids = [ids[i:i + self.fold_size] for i in range(0, len(ids), self.fold_size)]
        return ids

    def generate_mask(self, length):
        conditional_mask = [False] * length
        mask_span_length = random.randint(0, length - 1)
        start_index = random.randint(0, length - mask_span_length)
        conditional_mask[start_index:start_index + mask_span_length] = [True] * mask_span_length
        # half of the masks will be completely random
        if random.random() < 0.5:
            random.shuffle(conditional_mask)
        return conditional_mask

    def process_ids(self, ids):
        # and the eos token
        if self.eos_id >= 0:
            ids.append(self.eos_id)
        # randomly insert pads into ids
        if self.pad_id >= 0 and self.pad_insert_rate > 0:
            pad_count = int(len(ids) * self.pad_insert_rate)
            pad_indices = random.sample(range(len(ids)), pad_count)
            for index in pad_indices:
                ids.insert(index, self.pad_id)
        if self.fold_size is not None:
            ids = self.fold(ids)
        # crops the length
        if 0 < self.crop_length < len(ids):
            ids = ids[:self.crop_length]
        # create a conditional mask
        conditional_mask = self.generate_mask(len(ids))
        return ids, len(ids), conditional_mask

    def __call__(self, batch):
        processed = list(map(self.process_ids, batch))
        ids, lengths, conditional_mask = zip(*processed)

        # sample a random amount of padding
        padded_lengths = [random.randint(length, max(lengths)) for length in lengths]
        lengths = torch.tensor(padded_lengths) if self.length_includes_pad else torch.tensor(lengths)

        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in ids],
            batch_first=True,
            padding_value=self.pad_id
        )
        conditional_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.bool) for x in conditional_mask],
            batch_first=True,
            padding_value=False
        )

        return ids, lengths, conditional_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, rotary_embedding=None):
        super(MultiHeadAttention, self).__init__()
        assert (dim % num_heads == 0)
        self.model_dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_o = nn.Linear(dim, dim)

        self.rotary_emb = rotary_embedding

    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, _ = q.size()

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.1 if self.training else 0.0)

        # score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     score = score.masked_fill(mask == 0, -1e9)
        # score = F.softmax(score, dim=-1)
        # out = score @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim)
        return self.w_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, drop_prob=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            rotary_embedding=RotaryEmbedding(dim=32)
        )
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask, gammas=(0.0, 0.0), betas=(0.0, 0.0)):
        res = x
        x = self.norm1(x)
        x = (gammas[0] * x) + betas[0]
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = res + self.dropout1(x)

        res = x
        x = self.norm2(x)
        x = (gammas[1] * x) + betas[1]
        x = self.ffn(x)
        x = res + self.dropout2(x)
        return x


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x):
        freq = torch.einsum('b,d->bd', x, self.weights) * 2 * math.pi
        return torch.cat([freq.sin(), freq.cos()], dim=-1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, model_dim, num_layers=8, learned_sinusoidal_dim=128, dropout_prob=0.0,
                 layerdrop_prob=0.0):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layerdrop_prob = layerdrop_prob

        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(learned_sinusoidal_dim, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_layers * 4 * model_dim),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.Dropout(p=dropout_prob)
        )

        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim=model_dim,
                hidden_dim=4 * model_dim,
                num_heads=8,
                drop_prob=dropout_prob
            )
            for _ in range(num_layers))

        self.out = nn.Linear(model_dim, target_dim)

    def forward(self, x, t, length_mask=None):
        time_emb = self.time_mlp(t)
        x = self.project(x)

        if length_mask is not None:
            x = x * length_mask.unsqueeze(-1)
            length_mask = length_mask.unsqueeze(1).unsqueeze(1)

        scaling_weights = time_emb.view(-1, self.num_layers * 4, self.model_dim).split(1, dim=1)
        for i, layer in enumerate(self.encoder_layers):
            if self.training and random.uniform(0, 1) < self.layerdrop_prob:
                continue
            gammas = scaling_weights[4 * i], scaling_weights[4 * i + 1]
            betas = scaling_weights[4 * i + 2], scaling_weights[4 * i + 3]
            x = layer(x, length_mask, gammas=gammas, betas=betas)

        return self.out(x), x


class Diffusion:
    def __init__(self, estimator: nn.Module, interpolate=None, self_conditioning=True, normalize=False, masking=None,
                 sampling_method='difflm'):
        super(Diffusion).__init__()
        self.estimator = estimator
        self.interpolate = interpolate
        self.self_conditioning = self_conditioning
        self.normalize = normalize
        self.masking = masking
        self.sampling_method = sampling_method

    def gamma(self, t, ns=0.0002, ds=0.00025):
        return torch.cos(((t + ns) / (1 + ds)) * math.pi / 2) ** 2

    def forward_diffusion(self, x_0, t):
        time = t.unsqueeze(1).unsqueeze(1)
        mean_weight = torch.sqrt(self.gamma(time))
        std = torch.sqrt(1 - self.gamma(time))
        z = torch.randn_like(x_0)
        x_t = (mean_weight * x_0) + (z * std)
        return x_t, z, std

    @torch.no_grad()
    def reverse_diffusion(self, x_T, steps):
        x_t = x_T
        x_estimation = torch.zeros_like(x_t)

        t_now = torch.ones(x_T.shape[0], dtype=x_t.dtype, device=x_t.device, requires_grad=False)

        for step in range(steps):
            if not self.self_conditioning:
                x_estimation = torch.zeros_like(x_t)

            if self.normalize:
                x_t = x_t / x_t.std(dim=-1, keepdim=True)

            x_estimation, latent = self.estimator(torch.cat([x_t, torch.zeros_like(x_t), x_estimation], dim=-1), t_now)

            if self.interpolate is not None:
                x_estimation = self.interpolate(latent)

            t_next = torch.clamp(t_now - 1 / steps, 0.0, 1.0)

            if self.sampling_method == 'ddim':
                x_t = self.ddim_step(x_t, x_estimation, t_now, t_next)
            elif self.sampling_method == 'ddpm':
                x_t = self.ddpm_step(x_t, x_estimation, t_now, t_next)
            elif self.sampling_method == 'difflm':
                x_t = self.diff_lm_step(x_t, x_estimation, t_now, t_next)
            else:
                ValueError(f"Sampling method {self.sampling_method} not available.")

            t_now = t_next

        t_final = torch.zeros(x_T.shape[0], device=x_T.device)
        _, latent = self.estimator(torch.cat([x_t, torch.zeros_like(x_t), x_estimation], dim=-1), t_final)

        return x_t, latent

    def diff_lm_step(self, x_t, x_0_estimation, t_now, t_next):
        gamma_next = self.gamma(t_next).unsqueeze(1).unsqueeze(1)
        eps = torch.randn_like(x_0_estimation)
        return torch.sqrt(gamma_next) * x_0_estimation + torch.sqrt(1 - gamma_next) * eps

    def ddim_step(self, x_t, x_0_estimation, t_now, t_next):
        gamma_now = self.gamma(t_now).unsqueeze(1).unsqueeze(1)
        gamma_next = self.gamma(t_next).unsqueeze(1).unsqueeze(1)
        eps = torch.rsqrt(1 - gamma_now) * (x_t - torch.sqrt(gamma_now) * x_0_estimation)
        return torch.sqrt(gamma_next) * x_0_estimation + torch.sqrt(1 - gamma_next) * eps

    def ddpm_step(self, x_t, x_0_estimation, t_now, t_next):
        gamma_now = self.gamma(t_now).unsqueeze(1).unsqueeze(1)
        alpha_now = gamma_now / self.gamma(t_next).unsqueeze(1).unsqueeze(1)
        std_now = torch.sqrt(1.0 - alpha_now)
        z = torch.randn_like(x_t)
        eps = torch.rsqrt(1 - gamma_now) * (x_t - torch.sqrt(gamma_now) * x_0_estimation)
        return torch.rsqrt(alpha_now) * (x_t - (1 - alpha_now) * torch.rsqrt(1 - gamma_now) * eps) + std_now * z

    def loss_t(self, x, t, len_mask, cond_mask):
        x_target = x.detach()

        if self.masking is not None:
            x = self.masking(x)

        x_t, z, std = self.forward_diffusion(x, t)

        if self.normalize:
            x_t = x_t / x_t.std(dim=-1, keepdim=True)

        x_noised = x_t.masked_fill(cond_mask.unsqueeze(-1), 0.0)
        x_cond = x.masked_fill(~cond_mask.unsqueeze(-1), 0.0)

        x_estimation = torch.zeros_like(x_t)
        if self.self_conditioning and random.uniform(0, 1) < 0.5:
            with torch.no_grad():
                x_estimation, latent = self.estimator(torch.cat([x_noised, x_cond, x_estimation], dim=-1), t,
                                                      len_mask)

                if self.interpolate is not None:
                    x_estimation = self.interpolate(latent)

                x_estimation = x_estimation.masked_fill(cond_mask.unsqueeze(-1), 0.0)
                x_estimation = x_estimation.detach()

        x_estimation, latent = self.estimator(torch.cat([x_noised, x_cond, x_estimation], dim=-1), t, len_mask)

        return ((x_estimation - x_target) ** 2.0).mean(-1), x_estimation, latent

    def compute_loss(self, x_0, len_mask, cond_mask, offset=1e-5):
        t = torch.rand(x_0.shape[0], dtype=x_0.dtype, device=x_0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        loss, x_0_estimation, latent = self.loss_t(x_0, t, len_mask, cond_mask)
        return loss, x_0_estimation, latent


class DiffusionLM(nn.Module):
    def __init__(self, num_embeddings=1000, embedding_dim=64, model_dim=512, num_layers=8, dropout_prob=0.2,
                 layerdrop_prob=0.0):
        super(DiffusionLM, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding_grad_scale = 1.0
        self.interpolate_temperature = 1.0

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.norm = nn.LayerNorm(self.embedding_dim)

        self.mask_emb = nn.Parameter(torch.FloatTensor(self.embedding_dim).uniform_())

        self.estimator = TransformerModel(
            input_dim=self.embedding_dim * 3,
            target_dim=self.embedding_dim,
            model_dim=self.model_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            layerdrop_prob=layerdrop_prob
        )
        self.diffusion = Diffusion(
            estimator=self.estimator,
            interpolate=self.interpolate,
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.lm_head = nn.Linear(self.model_dim, self.num_embeddings)

        self.loss_ce = nn.CrossEntropyLoss(reduction='none')

    def get_embeddings(self, ids):
        e = self.embedding(ids)
        e = self.norm(e)
        return e

    def get_logits(self, x):
        x = self.dropout(x)
        x = self.lm_head(x)
        return x

    def interpolate(self, x):
        logits = self.get_logits(x) / self.interpolate_temperature
        weights = logits.softmax(dim=-1)
        e = self.embedding.weight
        e = self.norm(e)
        interpolated = torch.einsum('nle,ed->nld', weights, e)
        return interpolated

    def dist_embedding(self, x):
        e = self.embedding.weight
        e = self.norm(e)
        return torch.cdist(x, e)

    def cosine_similarity(self, x):
        e = self.embedding.weight
        e = F.normalize(e, dim=-1)
        x = F.normalize(x, dim=-1)
        cossim = torch.einsum('nld,ed->nle', x, e)
        return cossim

    def apply_mask(self, x):
        batch_size, seq_length, _ = x.size()
        token_mask = torch.rand(batch_size, seq_length, 1, device=x.device) < 0.15
        return torch.where(token_mask, self.mask_emb, x)

    @torch.no_grad()
    def clip_critic(self, clip_value=0.01):
        for param in self.lm_head.parameters():
            param.clamp_(-clip_value, clip_value)

    def compute_loss(self, ids, lengths, conditional_mask=None):
        x = self.get_embeddings(ids)
        x = self.embedding_grad_scale * x + (1.0 - self.embedding_grad_scale) * x.detach()

        length_mask = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        diff_mask = torch.logical_and(length_mask, torch.logical_not(conditional_mask))
        num_elems = diff_mask.sum()

        loss_diff, x_estimation, latent = self.diffusion.compute_loss(x, length_mask, conditional_mask)
        loss_diff = loss_diff * diff_mask
        loss_diff = loss_diff.sum() / num_elems

        logits = self.get_logits(latent)
        ids = ids.masked_fill(torch.logical_not(diff_mask), -100)
        loss_reconstruction = self.loss_ce(logits.transpose(2, 1), ids)

        accuracy = (logits.argmax(dim=-1) == ids).float().sum() / num_elems

        loss_diff = loss_diff.sum() / num_elems
        loss_reconstruction = loss_reconstruction.sum() / num_elems
        loss = loss_diff + loss_reconstruction

        return loss, loss_diff, loss_reconstruction, accuracy

    @torch.no_grad()
    def forward(self, z):
        x, latent = self.diffusion.reverse_diffusion(z, 200)
        return self.get_logits(latent).argmax(dim=-1)


def linear_decay_with_warmup(step, max_learning_rate, warmup_steps, hold_steps, decay_steps, min_learning_rate=1e-8):
    if step < warmup_steps:
        return max_learning_rate * (step / warmup_steps)
    elif step < warmup_steps + hold_steps:
        return max_learning_rate
    else:
        offset = warmup_steps + hold_steps
        scale = 1 - (step - offset) / decay_steps
        return max(max_learning_rate * scale, min_learning_rate)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wup', '--warmup_steps', type=int, default=2000)
    parser.add_argument('-decs', '--decay_steps', type=int, default=800000)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)
    parser.add_argument('-crtc', '--clip_critic', type=bool, default=True)

    parser.add_argument('-edim', '--embedding_dim', type=int, default=64)
    parser.add_argument('-mdim', '--model_dim', type=int, default=512)
    parser.add_argument('-numl', '--num_layers', type=int, default=8)
    parser.add_argument('-do', '--dropout_prob', type=float, default=0.1)
    parser.add_argument('-ld', '--layerdrop_prob', type=float, default=0.0)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-spm', '--spm_model', type=str, required=True)
    parser.add_argument('-cl', '--crop_length', type=int, default=64)
    parser.add_argument('-ngen', '--num_examples', type=int, default=8)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tokenizer = SentencePieceTokenizer(args.spm_model)

    model = DiffusionLM(
        num_embeddings=len(tokenizer),
        embedding_dim=args.embedding_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout_prob,
        layerdrop_prob=args.layerdrop_prob
    )
    model.to(device)

    if os.path.exists(args.checkpoint):
        print(f"Restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"Starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    with torch.no_grad():
        x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
        outputs = model(x_T).tolist()
        [print(tokenizer.decode(encoded)) for encoded in outputs]

    dataset = TextDataset(path=args.data_path, tokenizer=tokenizer)
    collate = Collate(
        crop_length=args.crop_length,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        length_includes_pad=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay
    )

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    lr_lambda = lambda step: linear_decay_with_warmup(step, args.learning_rate, args.warmup_steps, 0, args.decay_steps,
                                                      args.learning_rate * 0.1)

    for ep in range(0, args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        pbar.set_description(f"epoch: {ep}")

        for idx, (ids, lengths, conditional_mask) in enumerate(pbar):

            ids, lengths, conditional_mask = ids.to(device), lengths.to(device), conditional_mask.to(device)

            loss, loss_diff, loss_reconstruction, accuracy = model.compute_loss(ids, lengths, conditional_mask)

            pbar.set_postfix({
                "loss": loss.item(),
                "mse": loss_diff.item(),
                "ce": loss_reconstruction.item(),
                "accuracy": accuracy.item(),
            })

            (loss / args.accumulation_steps).backward()

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.param_groups[0]['lr'] = lr_lambda(global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                if args.clip_critic:
                    model.clip_critic()

                optim.zero_grad()
                global_step += 1

            if ((idx + 1) % 500 == 0) or (idx + 1 == len(dataloader)):
                checkpoint = {
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }
                torch.save(checkpoint, args.checkpoint)

        model.eval()
        with torch.no_grad():
            x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
            outputs = model(x_T).tolist()
            [print(tokenizer.decode(encoded)) for encoded in outputs]


if __name__ == "__main__":
    train()
