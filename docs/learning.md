# Reading NanoChat from text to predictions

This example teaches a small language model and how to express its training loop
in Haskell. You need basic familiarity with records, lists, `Maybe`, and `IO`.
The tensor shapes below supply information that Hasktorch's untyped `Tensor`
type does not carry in Haskell's type system.

Start with the [small runnable experiment](../examples/tiny/README.md). It needs
no dataset download. Then read the code in this order:

| Stage | Code to read | Question it answers |
|---|---|---|
| Words and IDs | [`buildTokenizer`, `tokenToId`](../src/NanoChat/Tokenizer.hs) | How does text become numbers? |
| Training examples | [`sampleBatch`](../src/NanoChat/Training.hs) | What should the model predict at each position? |
| Parameters | [`NanoChatModelSpec`, `NanoChatModel`](../src/NanoChat/Transformer.hs) | What is initialized and learned? |
| Forward pass | [`modelForwardWith`, `blockForward`, `headForward`](../src/NanoChat/Transformer.hs) | How do IDs become predictions? |
| Learning | [`trainStep`, `trainModel`](../src/NanoChat/Training.hs) | How does an error change the parameters? |
| Evaluation | [`evaluateLoss`](../src/NanoChat/Training.hs) | Does the model predict unseen text better? |
| Text generation | [`generate`](../src/NanoChat/Transformer.hs) | How does a next-token predictor write a response? |
| Persistence | [`saveTrainingCheckpoint`, `loadTrainingCheckpoint`](../src/NanoChat/Checkpoint.hs) | What must survive between runs? |
| Putting it together | [training entry point](../app/train/Main.hs), [chat entry point](../app/chat/Main.hs) | How are these functions used? |

The generated Haddock pages provide argument and shape contracts for these
functions. Build them with:

```sh
cabal haddock lib:nanochat --disable-documentation
```

The flag avoids requesting documentation for every dependency; documentation for
this explicit target is still generated. Cabal prints the output location.
Links to dependency symbols need installed dependency documentation, so a local
build can report missing Hasktorch links even though this project's pages build.

## 1. Words become IDs

The tokenizer splits on whitespace. It reserves four IDs:

```text
0 = [UNK]    1 = [USER]    2 = [BOT]    3 = [END]
```

Other distinct words get IDs in sorted text order. For `hello hello world`,
`hello` gets ID 4 and `world` gets ID 5. An ID is an index, not a numeric measure
of a word's meaning. Repetition does not create more vocabulary entries.

Capitalization and punctuation matter: `Hello`, `hello`, and `hello.` are three
different tokens. The tokenizer does not learn subword pieces. An unseen word
maps to `[UNK]`.

Build the vocabulary from training data only. Validation, chat, and resumed
training reuse it. Rebuilding the vocabulary could make the same ID refer to a
different word, destroying the meaning of the learned embedding rows.

## 2. IDs become prediction examples

Suppose a window of the token stream is:

```text
[USER] hello [BOT] hi [END]
```

A training example of context length four is:

```text
input:  [USER] hello [BOT] hi
target: hello  [BOT] hi    [END]
```

Every input position predicts the next token. `sampleBatch` chooses several
starting positions at random and stacks their windows. It returns **one batch
per call**. Starting positions are sampled with replacement: windows can overlap
or recur. The training loop does not perform complete epochs through the data.

With batch size eight and context length 128, both input and target tensors have
shape `[8,128]`, giving 1,024 prediction targets per update. `Text.words` removes
line boundaries, so windows can cross conversations. All targets contribute to
loss, including user text and special markers.

## 3. Specifications become parameters

```haskell
model <- T.sample (NanoChatModelSpec cfg)
```

`T` is the local alias for `Torch`. `sample` is a method of `Randomizable`.
`NanoChatModelSpec` describes dimensions; the resulting `NanoChatModel` contains
initialized parameters. This use of “sample” concerns initialization, not
selecting a training batch or generating a word.

`sample` returns an `IO` action because initialization uses randomness. The
instance in `Transformer.hs` constructs the nested records. The `<$>` and `<*>`
operators assemble those records from initialization actions; `replicateM`
runs an action repeatedly to create distinct heads and blocks.

A `Tensor` can hold data, activations, or weights. A `Parameter` identifies a
trainable leaf for automatic differentiation. Our embedding and normalization
initializers use `makeIndependent` to create these leaves. `toDependent` exposes
a parameter as a tensor for calculations, allowing those calculations to depend
on it in the gradient graph.

`Parameterized` tells Hasktorch how to find parameters inside a model. The
`Generic` instances support the traversal of these nested records.
`flattenParameters` returns them in a consistent order for Adam and serialization;
it does **not** concatenate all weights into one tensor.

For another small initialization/optimization example, see Hasktorch's
[linear regression tutorial](https://hasktorch.github.io/tutorial/06-linear-regression.html).
This repository's code targets Hasktorch 0.2.2.x; tutorial API details may differ.

## 4. Follow the shapes through the transformer

Use these names throughout:

| Symbol | Meaning | Default | Tiny example |
|---|---|---:|---:|
| B | Batch size | 8 | 8 |
| T | Input sequence length during training | 128 | 16 |
| D | Embedding width | 128 | 32 |
| H | Heads per block | 4 | 2 |
| K | Width per head, D / H | 32 | 16 |
| V | Vocabulary size | Derived from training text | Derived from training text |

The forward pass follows this path:

```text
IDs [B,T]
  -> token embedding lookup [B,T,D]
  +  position embeddings [T,D], broadcast across B
  -> transformer blocks [B,T,D]
  -> final layer normalization [B,T,D]
  -> output projection [B,T,V]
```

`embeddingForward` selects a row of the embedding table for each ID. Those rows
are learned vectors. Position embeddings give the model a representation of
where a token appears within the current window.

### One attention head

From the normalized input `x`, three learned projections form queries, keys,
and values, each `[B,T,K]`. These are three different transformations of the
same input, not three different datasets.

```text
Q [B,T,K] multiplied by transpose(K) [B,K,T]
    -> scores [B,T,T]
    -> divide by sqrt(head width)
    -> add causal mask [T,T]
    -> softmax along the final axis
    -> dropout during training
    -> multiply by V [B,T,K]
    -> head output [B,T,K]
```

Here `K` in `transpose(K)` denotes the key tensor; “head width” is the scalar
`D / H`. Each score row asks which input positions matter for a particular query
position. The final multiplication forms a weighted combination of value vectors.

The additive mask for three positions is:

```text
          key position
           0    1    2
query 0 [  0   -∞   -∞ ]
      1 [  0    0   -∞ ]
      2 [  0    0    0 ]
```

The current position and earlier positions remain visible. Future scores become
negative infinity, so their softmax weights are zero. Setting future scores to
zero would not work, because `exp 0 = 1`.

This prevents a position from looking at its target in the next input position,
even though training computes all positions in parallel. At inference time only
the available prompt and previously generated tokens are provided.

### A complete block

`layerNormForward` normalizes the final embedding axis separately for each token.
It subtracts the component mean, divides by the square root of variance plus a
small constant, and applies learned gain and bias. It does not normalize across
time or across different examples.

`blockForward` computes that normalized input once for all heads. The heads have
different projections, so sharing the input does not make their outputs equal.
Their `[B,T,K]` outputs are concatenated to `[B,T,D]` and projected to width `D`.

The two sublayers are:

```text
residual = x + attention(layerNorm1(x))
output   = residual + feedForward(layerNorm2(residual))
```

Dropout is applied at the sites shown in the implementation. The feed-forward
network expands width `D` to `4*D`, applies ReLU, and contracts to `D`. Each
residual addition preserves an unchanged path from its input. Normalization is
before each sublayer: this is called pre-layer normalization.

### Relation to the paper

Attention follows equation 1 and the masking discussion in section 3.2.3 of
[Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762).
The feed-forward form corresponds to section 3.3. The paper's main model uses an
encoder and decoder, post-layer normalization, and sinusoidal positions. This
example uses only causal self-attention blocks, pre-layer normalization, and
learned positions. It is not a reproduction of the full translation model.

## 5. Predictions become loss, and loss changes weights

The final linear layer produces **logits**, unnormalized scores for all `V`
possible next tokens at each `[B,T]` position. A larger vocabulary enlarges this
layer's work; input embedding lookup alone is not the main vocabulary-dependent
matrix multiplication.

For the loss, the code reshapes logits to `[B*T,V]` and targets to `[B*T]`.
`logSoftmax` converts each score row into log probabilities. `nllLoss'` takes the
negative log probability assigned to the correct next token and averages across
positions. Together these implement mean cross-entropy, in natural-log units
(nats) per target token. Lower loss means more probability assigned to the
observed targets on average.

`modelForwardTrain` computes that loss but does not change parameters.
`trainStep` passes it to `Optim.runStep`, which computes gradients and returns
updated parameters plus updated Adam state. Adam keeps a running estimate of
gradients and squared gradients for each parameter, as well as a step counter.
The learning rate controls the update scale.

`trainModel` uses `foldM` to carry the returned triple through the steps:

```text
(model0, adam0, loss0)
  -> iteration 1 -> (model1, adam1, loss1)
  -> iteration 2 -> (model2, adam2, loss2)
  -> ...
```

There is no reassignment to the original `optimizer` variable. The new optimizer
in one result becomes the optimizer argument of the next iteration. The logged
batch loss was measured before that step's parameter update.

### Why both Identity and IO?

`modelForwardWith` accepts a function `Tensor -> m Tensor`. Training passes a
random dropout action in `IO`. Evaluation passes `Identity`, which wraps a tensor
without changing it; `runIdentity` unwraps the result. The `Monad m` constraint
allows both uses of the same forward-pass code.

Evaluation therefore disables dropout without duplicating the attention code.
It does not turn generation into a deterministic operation: `generate` still
randomly samples the next token from the output distribution.

## Measuring learning

Add `--validation FILE` before the fresh/resume command to evaluate both datasets
before and after training. For example:

```sh
cabal run nanochat-train -- --tiny --validation examples/tiny/validation.txt models/learning.model examples/tiny/train.txt 200
```

The vocabulary is built from training text only. Validation words that are absent
from it become `[UNK]`; the command prints their count. Heavy use of `[UNK]` limits
what the score can tell you about the original text. Validation never changes
vocabulary, weights, or optimizer state.

`evaluateLoss` disables dropout and traverses each dataset in a fixed order.
It scores every token after the first exactly once. Input windows begin at
positions `0, blockSize, 2*blockSize, ...`; a final shorter window is included.
Target windows are shifted forward one token. Thus adjacent input windows do not
overlap, although the next window's first input was the previous window's last
target. Context resets at each window boundary. This is a fixed-window score,
not evaluation with the longest possible sliding context at every position.

Batch means are weighted by their numbers of targets before averaging. Otherwise
a short final batch would have too much influence. Changing evaluation batch size
changes grouping, not window boundaries; tiny floating-point differences may
still occur. Evaluating a large corpus adds extra forward passes and can be slow.

The training-set and validation scores are measured in the same mode. Compare
those scores with each other, rather than comparing a random dropout-on batch
with the whole held-out corpus. Falling training loss with flat or rising
validation loss is evidence that fitting the training set is not improving the
held-out predictions. It is not guaranteed to happen in every tiny experiment.
Keep a separate test set for a final assessment after choosing settings.

The current checkpoint retains the last training batch's loss. Evaluation scores
are printed to the terminal and are not stored there. Both losses include user
text and markers, not just assistant responses.

## 6. Predicting repeatedly produces text

`generate` takes the logits at the final input position, applies softmax, samples
one token, appends it, and repeats. Once the sequence is longer than `blockSize`,
only the most recent window is evaluated. Learned positions restart at zero in
that window. There is no key/value cache, so the window is recomputed each step.

The chat executable requests up to 30 new tokens and displays only the portion
before the first `[END]`. `generate` itself still produces the requested count.
Each user message starts a fresh prompt; the chat loop does not retain earlier
turns. A small synthetic dataset is useful for inspecting behavior but cannot
teach general conversation.

## 7. What a checkpoint preserves

A version-2 archive stores configuration, vocabulary order, model parameters,
Adam's two moment tensors per parameter, its betas and counter, and training
metadata. The moments roughly triple the tensor storage compared with weights
alone. The model and optimizer use matching parameter order.

Resume loads those values and trains for an additional number of steps. A new
output filename preserves the input checkpoint. Version-1 archives can still be
used for chat, but contain no Adam state. Checkpoints do not store datasets or RNG
state, and they are written only after a successful run. These limits are described
in [the README](../readme.md) and the Haddock for `NanoChat.Checkpoint`.
