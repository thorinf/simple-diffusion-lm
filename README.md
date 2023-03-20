# simple-diffusion-lm

A simple, introductory approach to Diffusion models for a discrete, text generation task.

## What are the aims of this repository?

To create a minimal implementation for training a Diffusion model for a text generation task,
and the accompanying generation process.
The goal is a modestly performing model with a minimalistic implementation.
It is not state-of-the-art techniques, but instead to show something that's _hopefully_ easy to understand
and build from.
There's accompanying notes below to explain design decisions as well as references.

## How to Run:

### Environment Setup:

Aside from PyTorch, the training script requires two other packages;
[SentencePiece](https://github.com/google/sentencepiece) and
[RotaryEmbedding](https://github.com/lucidrains/rotary-embedding-torch).
They can be installed with the following commands:

```commandline
pip install sentencepiece
pip install rotary-embedding-torch
pip install tqdm
```

### Pre-Processing:

First generate a `.txt` corpus where each line is an example.
It's recommended to apply some normalisation on the text so the data is quite clean for the next step and training, e.g.
lower-case, change numbers to words, removing unnecessary symbols.
The training script won't perform these normalisations, so data should be cleaned externally.

With a clean text corpus, the SentencePiece model can then be trained.
Follow the guides on [their repository](https://github.com/google/sentencepiece)
or [here on PyPI](https://pypi.org/project/sentencepiece/).
If the text corpus is very large, then creating a subset of the text can get around memory issues.

### Training:

The model can be trained with the command:

```commandline
python train.py -d=TXT_CORPUS -spm=SPM_MODEL -ckpt=CHECKPOINTING_PATH
```

There's a bunch of other arguments which can be altered, but above is enough to get the model working.

## Design & Implementation Notes:

The implementation is mostly based off of a small number of papers:

* [Analog Bits: Generating discrete data using diffusion models with self-conditioning][1]
* [Difformer: Empowering Diffusion Models on the Embedding Space for Text Generation][2]
* [Continuous diffusion for categorical data][3]
* [Diffusion-LM Improves Controllable Text Generation][4]

There's a lot of research out there describing the forward and reverse diffusion processes;
DDIM, DDPM, Score-based, Continuity, EDM, etc.
After reading a few papers the similarities do become clear, but the derivations can make it hard spot
how to actually implement something.
[Analog Bits][1] is very light on mathematical derivations,
but is incredibly clear when it comes to the pseudocode.
Because of this clarity, and brevity, it is the basis of the `Diffusion` class.
Additionally, this paper introduces some mechanisms, such as self-conditioning,
making it a very good paper to look at - highly recommended.

### Forward & Reverse Diffusion Explanation:

This is a very high level simplification, but might be useful as an introductory description.

When thinking about the forward process you can forget Markov Chains, Beta and Alpha Cumulative Schedules,
and instead think of it as the annealing of the desired data to a normal distribution.
At `t=0` the forward diffusion process results in the original data, and at `t=1` a normal distribution.
Between these `t` values the corrupted data is a mixture of both distributions governed by an annealing schedule.

In the reverse (generative) process, for a given number of steps moving from `t=1` to `t=0`,
the model will predict the de-noised data.
The start of the iterative process `t=1` will be a normal distribution,
and `t=0` will _hopefully_ be data which could be sampled from the training datas underlying distribution.
At each iteration there is a weighted sum between the new prediction and the previous one,
such that a small amount of noise if being shaved off each iteration.

### Losses:

The model is trained with two losses, like the [Difformer][2]. The inner model, which is named `Estimator` here,
predicts the original embeddings x̂<sub>0</sub> directly, rather than the noise added. Like [Analog Bits][1],
the Mean-Square-Error is taken between the ground-truth latent and the predicted latent.
The Cross-Entropy loss is taken between a projection of x̂<sub>0</sub> and the target tokens.

The loss is masked out for indices that are outside the original sequence length or are conditional positions.
There are more details on conditioning below.

### Normalisation:

As noted in [Difformer][2], [CDCD][3], and [Diffusion-LM][4],
the MSE training objective can be trivialised by collapsing the embedding space.
This is countered by some form of normalisation that is applied to the embeddings before `forward_diffusion`.

[Difformer][2] suggests in 3.2 that imbalanced scales of embeddings (i.e. L2 norms) -
where more frequent tokens have a smaller scale than less frequent -
could change the de-noising process. In their implementation they use Layer Normalisation, but interestingly
they allow element-wise scaling of the parameters.
With this scaling enabled, the MSE alone could be trivialised - perhaps the support of the CE loss prevents this.

[CDCD][3] has the opposite problem but a similar solution.
Instead of using the MSE loss, they use 'score interpolation' and train directly on CE only.
This would cause the embeddings to expand without normalisation, rather than collapse.
They choose to use L2 normalisation, and scale the embeddings by `sqrt(embedding_dim)`,
which keeps the standard deviation at 1.
This does also solve the possible imbalance issue noted above,
although they do not suggest this normalisation was chosen specifically to solve it.

When choosing how to normalise there is a general goal - keep the embedding target similar to the prior distribution
of the Diffusion process (i.e. a standard normal distribution).
Initial investigations of this code base experimented with `BatchNorm`, but without the element-wise scaling enabled.
However, the model did over-predict more common tokens at inference, so keeping embedding scale the same does
appear to be important. For this reason it's suggested to use a scaled L2 normalisation or `LayerNorm`.

### Estimator:

#### Transformer Encoder:

The architecture used for the Transformer is based off of the [CDCD][3] framework. The encoder layers use a pre-norm
`LayerNorm`, and the `Attention` is using [rotary positional embedding][6] on the queries and keys.
Inspired by [FiLM][7], the [CDCD][3] framework uses the temporal embeddings to scale the latent following
each `LayerNorm`.
Each `LayerNorm` in every `TransformerEncoderLayer` has its own scaling for both `beta` and `gamma`,
a total of 4 scaling parameters per layer.

#### Interpolation:

Interpolation is used in [CDCD][3] to re-estimate the embeddings using the probability of each token.
The interpolated embeddings are a weighted combination of the embedding vectors,
where weights are the softmax distribution. 
There is no training loss applied directly to the interpolated embeddings,
however interpolated embedding are used in training when using self-conditioning.


### Diffusion:

#### Conditioning:

In the context of this implementation, conditioning can be thought of as impainting where an initial set of token
embeddings are fed to the model and the gaps filled in.
As noted in [CDCD][3] section 4.1, it is possible to re-inject certain tokens when sampling,
however this is less effective than training the model to support conditional sampling.
The `simple-diffusion-lm` implementation takes the same approach as the [CDCD][3] framework,
but chooses to not concatenate the conditioning mask with the other embeddings.
There are many ways to generate masks for conditional masking and the [CDCD][3] framework explores these thoroughly.
However, this implementation only use a low probability, fully random, conditional mask.

#### Self-Conditioning:

Self-conditioning was introduced in [Analog Bits][1]. The idea is to pass a previous prediction into the `Estimator`
to improve prediction accuracy. At sampling time this comes at a negligible cost,
and only increase training time by a small factor (under 25%, according to the paper).
Note that like [CDCD][3], the self-conditioning vectors are set to zero for conditional positions.


[1]: <https://arxiv.org/abs/2208.04202> "Analog Bits: Generating discrete data using diffusion models with self-conditioning"

[2]: <https://arxiv.org/abs/2212.09412> "Difformer: Empowering Diffusion Models on the Embedding Space for Text Generation"

[3]: <https://arxiv.org/abs/2211.15089> "Continuous diffusion for categorical data"

[4]: <https://arxiv.org/abs/2205.14217> "Diffusion-LM Improves Controllable Text Generation"

[5]: <https://arxiv.org/abs/2301.10972> "On the Importance of Noise Scheduling for Diffusion Models"

[6]: <https://arxiv.org/abs/2104.09864> "RoFormer: Enhanced Transformer with Rotary Position Embedding"

[7]: <https://arxiv.org/abs/1709.07871> "FiLM: Visual Reasoning with a General Conditioning Layer"
