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

    def encode_from_file_path(self, file_path):
        text = get_text(file_path)
        return self.encode(text), text

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, encoded):
        return self.sp.decode(encoded)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: SentencePieceTokenizer):
        self.path = path
        self.offsets = get_line_offsets(path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, 'r', encoding='utf-8') as file:
            file.seek(self.offsets[idx])
            text = file.readline().strip()
        encoded = self.tokenizer.encode(text)
        return encoded


class Collate:
    def __init__(self, crop_length=-1):
        self.crop_length = crop_length

    def __call__(self, batch):
        encoded_list = [(torch.tensor(tokens, dtype=torch.int64)) for tokens in batch]
        encoded = torch.nn.utils.rnn.pad_sequence(encoded_list, batch_first=True, padding_value=0)
        lengths = torch.tensor([x.shape[0] for x in encoded_list])
        if 0 < self.crop_length < encoded.shape[1]:
            encoded = encoded[:, :self.crop_length]
            lengths = torch.minimum(lengths, torch.tensor(self.crop_length))
        return encoded, lengths


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, rotary_embedding=None):
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
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view((q.shape[0], -1, self.num_heads, self.head_dim)).transpose(2, 1)
        k = k.view((k.shape[0], -1, self.num_heads, self.head_dim)).transpose(2, 1)
        v = v.view((v.shape[0], -1, self.num_heads, self.head_dim))

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        score = torch.einsum('nhqd,nhkd->nhqk', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        out = torch.einsum('nhqk,nkhd->nqhd', score.softmax(dim=-1), v)
        out = out.reshape((q.shape[0], -1, self.model_dim))
        return self.w_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, drop_prob=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            rotary_embedding=RotaryEmbedding(dim=32)
        )
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=drop_prob)
        )
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask, betas=(0.0, 0.0), gammas=(0.0, 0.0)):
        res = x
        x = self.norm1(x)
        x = gammas[0] * x + betas[0]
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = res + self.dropout1(x)

        res = x
        x = self.norm2(x)
        x = gammas[1] * x + betas[1]
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
        return torch.cat([x.unsqueeze(-1), freq.sin(), freq.cos()], dim=-1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, model_dim, num_layers=4, learned_sinusoidal_dim=128, dropout_prob=0.0):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(learned_sinusoidal_dim + 1, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 4 * num_layers),
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

        scaling_weights = time_emb.unsqueeze(2).split(1, dim=1)
        for i, layer in enumerate(self.encoder_layers):
            betas = scaling_weights[4 * i:4 * i + 2]
            gammas = scaling_weights[4 * i + 2:4 * i + 4]
            x = layer(x, length_mask, betas=betas, gammas=gammas)

        return self.out(x)


class Diffusion:
    def __init__(self, estimator: nn.Module, self_conditioning=True, normalize=False, sampling_method='ddpm'):
        super(Diffusion).__init__()
        self.estimator = estimator
        self.self_conditioning = self_conditioning
        self.normalize = normalize
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
    def reverse_diffusion(self, x_T, steps, td=0.0):
        x_t = x_T
        x_estimation = torch.zeros_like(x_t)

        for step in range(steps):
            t_now = 1 - step / steps
            t_next = max(1 - (step + 1 + td) / steps, 0)
            t_now = torch.tensor(t_now, device=x_t.device).repeat(x_T.shape[0])
            t_next = torch.tensor(t_next, device=x_t.device).repeat(x_T.shape[0])

            if not self.self_conditioning:
                x_estimation = torch.zeros_like(x_t)

            if self.normalize:
                x_t = x_t / x_t.std(dim=-1, keepdim=True)

            x_estimation = self.estimator(torch.cat([x_t, torch.zeros_like(x_t), x_estimation], dim=-1), t_now)

            if self.sampling_method == 'ddim':
                x_t = self.ddim_step(x_t, x_estimation, t_now, t_next)
            elif self.sampling_method == 'ddpm':
                x_t = self.ddpm_step(x_t, x_estimation, t_now, t_next)

        return x_t

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

    def loss_t(self, x_0, t, len_mask, cond_mask):
        x_t, z, std = self.forward_diffusion(x_0, t)

        if self.normalize:
            x_t = x_t / x_t.std(dim=-1, keepdim=True)

        x_noised = torch.where(cond_mask.unsqueeze(-1), torch.zeros_like(x_t), x_t)
        x_cond = torch.where(cond_mask.unsqueeze(-1), x_0, torch.zeros_like(x_0))

        x_estimation = torch.zeros_like(x_t)
        if self.self_conditioning and random.uniform(0, 1) < 0.5:
            x_estimation = self.estimator(torch.cat([x_noised, x_cond, x_estimation], dim=-1), t, len_mask)
            x_estimation = torch.where(cond_mask.unsqueeze(-1), torch.zeros_like(x_estimation), x_estimation)
            x_estimation = x_estimation.detach()

        x_0_estimation = self.estimator(torch.cat([x_noised, x_cond, x_estimation], dim=-1), t, len_mask)

        loss = (x_0 - x_0_estimation) ** 2.0
        return loss, x_0_estimation, t

    def compute_loss(self, x_0, len_mask, cond_mask, offset=1e-5):
        t = torch.rand(x_0.shape[0], dtype=x_0.dtype, device=x_0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        loss, x_0_estimation, t = self.loss_t(x_0, t, len_mask, cond_mask)
        return loss, x_0_estimation


class DiffusionLM(nn.Module):
    def __init__(self, num_embeddings=1000, embedding_dim=64, model_dim=512, num_layers=4, dropout_prob=0.2):
        super(DiffusionLM, self).__init__()
        self.num_embedding = num_embeddings
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embedding,
            embedding_dim=self.embedding_dim
        )
        nn.init.normal_(self.embedding.weight, std=0.001)

        self.embedding_grad_scale = 1.0

        self.estimator = TransformerModel(
            input_dim=self.embedding_dim * 3,
            target_dim=self.embedding_dim,
            model_dim=self.model_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        self.diffusion = Diffusion(estimator=self.estimator)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.lm_head = nn.Linear(self.embedding_dim, num_embeddings)

        self.loss_ce = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, w, lengths):
        x = self.embedding(w)
        x = F.normalize(x, dim=-1) * math.sqrt(self.embedding_dim)
        x = self.embedding_grad_scale * x + (1.0 - self.embedding_grad_scale) * x.detach()

        len_mask = torch.arange(w.shape[1], device=w.device).unsqueeze(0) < lengths.unsqueeze(1)
        # conditional masking could be much better. if true, use original embeddings
        cond_mask = torch.rand(w.shape, device=w.device) < 0.1
        diff_mask = len_mask & torch.logical_not(cond_mask)

        loss_diff, x_0_estimation = self.diffusion.compute_loss(x, len_mask, cond_mask)
        loss_diff = loss_diff[diff_mask].mean(-1)

        logits = self.lm_head(self.dropout(x_0_estimation))
        w = w.masked_fill(torch.logical_not(diff_mask), -100)
        loss_reconstruction = self.loss_ce(logits.transpose(2, 1), w)

        accuracy = (logits.argmax(dim=-1) == w).float().sum() / diff_mask.sum()

        loss_diff = loss_diff.mean()
        loss_reconstruction = loss_reconstruction.sum() / diff_mask.sum()
        loss = loss_diff + loss_reconstruction

        return loss, loss_diff, loss_reconstruction, accuracy

    def forward(self, z):
        x_0 = self.diffusion.reverse_diffusion(z, 100, 0.5)
        return self.lm_head(x_0).argmax(dim=-1)


def linear_decay_with_warmup(step, max_learning_rate, warmup_steps, hold_steps, decay_steps, min_learning_rate=1e-8):
    if step < warmup_steps:
        return max_learning_rate * (step / warmup_steps)
    elif step < warmup_steps + hold_steps:
        return max_learning_rate
    else:
        offset = warmup_steps + hold_steps
        scale = 1 - (step - offset) / (decay_steps - offset)
        return max(max_learning_rate * scale, min_learning_rate)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-decs', '--decay_steps', type=int, default=200000)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=4)

    parser.add_argument('-edim', '--embedding_dim', type=int, default=64)
    parser.add_argument('-mdim', '--model_dim', type=int, default=512)
    parser.add_argument('-numl', '--num_layers', type=int, default=4)
    parser.add_argument('-do', '--dropout_prob', type=float, default=0.1)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-spm', '--spm_model', type=str, required=True)
    parser.add_argument('-cl', '--crop_length', type=int, default=32)
    parser.add_argument('-ngen', '--num_examples', type=int, default=8)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = SentencePieceTokenizer(args.spm_model)

    model = DiffusionLM(
        num_embeddings=len(tokenizer),
        embedding_dim=args.embedding_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout_prob
    )
    model.to(device)

    if os.path.exists(args.checkpoint):
        print(f"Restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"Starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
        outputs = model(x_T).tolist()
        [print(text) for text in tokenizer.decode(outputs)]

    dataset = TextDataset(path=args.data_path, tokenizer=tokenizer)
    collate = Collate(crop_length=args.crop_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate
    )

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay
    )
    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    num_updates = checkpoint.get('num_updates', 0)
    lr_lambda = lambda step: linear_decay_with_warmup(step, args.learning_rate, 1000, 0, args.decay_steps)

    for ep in range(0, args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        for idx, (w, lengths) in enumerate(pbar):
            w = w.to(device)
            lengths = lengths.to(device)
            loss, loss_diff, loss_reconstruction, accuracy = model.compute_loss(w, lengths)

            pbar.set_description(f"epoch: {ep}")
            pbar.set_postfix({
                "loss": loss.item(),
                "mse": loss_diff.item(),
                "ce": loss_reconstruction.item(),
                "accuracy": accuracy.item(),
            })

            (loss / args.accumulation_steps).backward()
            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.param_groups[0]['lr'] = lr_lambda(num_updates)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optim.step()
                optim.zero_grad()
                torch.cuda.empty_cache()
                num_updates += 1

            if ((idx + 1) % 500 == 0) or (idx + 1 == len(dataloader)):
                checkpoint = {
                    'num_updates': num_updates,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }
                torch.save(checkpoint, args.checkpoint)

        model.eval()
        with torch.no_grad():
            x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
            outputs = model(x_T).tolist()
            [print(text) for text in tokenizer.decode(outputs)]


if __name__ == "__main__":
    train()
