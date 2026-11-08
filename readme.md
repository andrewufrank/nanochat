# NanoChat in Haskell

A small CPU language-model experiment inspired by
[karpathy/nanochat](https://github.com/karpathy/nanochat). This is an educational
transformer implementation, not a faithful port of upstream NanoChat or a loader
for its checkpoints. It uses learned positional embeddings, pre-layer normalization,
causal multi-head attention, a ReLU feed-forward network, and Adam. Upstream uses a
different architecture and a much larger training/tokenization pipeline.

## Learn by reading and running

Start with the [guided walkthrough](docs/learning.md), which follows text through
batching, attention, loss, Adam, generation, and checkpoints. It explains tensor
shapes and the Haskell machinery alongside links to the implementation.

The [tiny CPU experiment](examples/tiny/README.md) includes synthetic training and
validation text and uses the `--tiny` preset (context 16, width 32, two heads, one
block). It needs no dataset download. After building, run from the repository root:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cabal run nanochat-train -- --tiny --validation examples/tiny/validation.txt models/tiny-200.model examples/tiny/train.txt 200
cabal run nanochat -- models/tiny-200.model
```

The source modules contain Haddock explanations of each function's role, tensor
shapes, and assumptions. Generate the local API pages with:

```sh
cabal haddock lib:nanochat --disable-documentation
```

Cabal prints the output directory. This builds this target's documentation without
requesting documentation for every dependency; links to Hasktorch types require
its dependency documentation to be available.

## Build and run

The package uses Hpack and the untyped API of Hasktorch 0.2.2.x. Tested with
GHC 9.10.3 and Hasktorch 0.2.2.0. Hasktorch also requires its matching native LibTorch
libraries at runtime.

```sh
hpack
cabal build all
cabal test --test-show-details=direct
cabal run nanochat-train -- models/experiment-01.model chat_dataset.txt 500
cabal run nanochat -- models/experiment-01.model
```

Training takes a model filename, a dataset path, and an optional positive step
count (default 500). Optional `--tiny` and `--validation FILE` flags precede
these arguments. `--tiny` selects `tinyConfig` for new runs; resume always uses
the saved configuration. Chat takes only a saved model filename and does not need the
original dataset. The dataset must contain at least 129
whitespace-separated tokens for the default context size of 128 (17 tokens for
the tiny context size of 16). For dialogue
experiments, use text such as `[USER] hello [BOT] hi there [END]`, repeated across
your examples. The tokenizer is a simple word tokenizer with reserved `[UNK]`,
`[USER]`, `[BOT]`, and `[END]` tokens; vocabulary size is calculated from distinct
words plus those markers. It does not use BPE or tiktoken.

Training runs on CPU with batches of eight. Adjust `defaultConfig` in
`src/NanoChat/Transformer.hs` for smaller experiments; embedding width must be divisible by
head count. Dropout is enabled only during training. Training saves the model
and exits; run the separate chat executable to enter messages. A blank line or
EOF exits chat. Output is capped at 30 new tokens and is displayed up to the first
`[END]`. Tiny datasets and short training runs are useful for experiments, but
will not produce a capable chatbot.

Each `.model` is a single LibTorch archive containing model weights and versioned
JSON metadata (stored as a byte tensor): the exact configuration, vocabulary in
token-ID order, training-step count, dataset path, batch size, learning rate, and
last batch's training loss. Loading uses the saved configuration, even if you
have since changed `defaultConfig`. The last batch loss is not a validation score.
New training runs save version-2 checkpoints containing Adam's first and second
moment tensors, beta settings, and step counter as well. Their tensor payload is
approximately three times the size of model weights alone.

Use a different filename for each experiment; existing files are refused before
training starts. Parent directories are created automatically, and saving uses a
temporary file before publishing the completed archive. For example:

```sh
cabal run nanochat-train -- models/dialogue-2k.model data/dailydialog/train.txt 2000
cabal run nanochat-train -- models/dialogue-10k.model data/dailydialog/train.txt 10000
cabal run nanochat -- models/dialogue-2k.model
cabal run nanochat -- models/dialogue-10k.model
```

The commands above start independent training runs. To continue an existing run,
provide an input checkpoint, a new output filename, the dataset, and the number
of **additional** steps (default 500):

```sh
cabal run nanochat-train -- --resume models/dialogue-2k.model models/dialogue-resumed-10k.model data/dailydialog/train.txt 8000
cabal run nanochat -- models/dialogue-resumed-10k.model
```

Resume restores the saved model configuration, vocabulary, batch size, learning
rate, and Adam state. Step numbers continue from the checkpoint, so this example
finishes at 10,000 total steps. The input checkpoint is preserved; existing output
files are refused. You may supply a different dataset, but vocabulary IDs remain
fixed and new words map to `[UNK]`. The new checkpoint records the latest dataset
path. The dataset itself is not stored in the archive.

Version-1 files remain usable for chat, but cannot resume with Adam history because
they never stored it; the resume command reports this instead of silently creating
a fresh optimizer. Random-generator state is not saved, so random batches and
dropout after resuming will differ from an uninterrupted run. Saving still happens
only at the end of a successful run; periodic checkpoints and recovery of progress
from an interrupted run are not implemented. The files are for this Haskell model
format, not upstream NanoChat/nanoGPT checkpoints.

## Validation loss

Pass a held-out text file with `--validation FILE`, before the fresh or resume
arguments. The trainer reports **training-set evaluation loss and validation
loss before and after training**, both with dropout disabled. For example:

```sh
cabal run nanochat-train -- --validation data/dailydialog/validation.txt models/dialogue-validation.model data/dailydialog/train.txt 500
```

The training tokenizer is reused; validation cannot add words to the vocabulary.
Unseen words map to `[UNK]` and their count is reported. Validation requires at
least two tokens. `evaluateLoss` scores every token after the first exactly once
using consecutive windows, includes the final short window, and weights loss by
target count. Context resets at each window boundary. Evaluation batch size does
not change those boundaries.

These scores are mean cross-entropy in nats per token, over all tokens including
user text and markers. They are printed, not saved in checkpoint metadata.
Training progress still reports a random batch's dropout-on loss before its
update; it is a different measurement. Full evaluation adds two passes over each
dataset per invocation, so it can be costly on large files. Without `--validation`,
the existing training-only behavior is retained.

Compare both evaluation scores over successive runs: falling training loss alone
does not show that the model predicts held-out text better. See the
[walkthrough](docs/learning.md#measuring-learning) for the evaluation procedure.

The training entry point is `app/train/Main.hs`, chat is `app/chat/Main.hs`,
the transformer is in `src/NanoChat/Transformer.hs`,
tokenization is in `src/NanoChat/Tokenizer.hs`, training is in
`src/NanoChat/Training.hs`, and persistence is in `src/NanoChat/Checkpoint.hs`.

This workspace's `cabal.project.local` embeds the installed LibTorch library
directory in the executable and test binaries' runtime search path. Normal
`cabal build` and `cabal test` therefore work without `LD_LIBRARY_PATH`. This is
a machine-local setting: update the absolute path when moving the project or
changing the LibTorch installation, then rebuild.

On another machine, if the loader reports `libc10.so` or `libtorch_cpu.so` missing, add the `lib`
directory of the LibTorch installation used by Hasktorch to `LD_LIBRARY_PATH`.
On the machine used for validation, that was:

```sh
export LD_LIBRARY_PATH="$HOME/.cache/libtorch/2.9.1/linux-x86_64/cpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For these small CPU workloads, start with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
in the environment; benchmark larger thread counts for your machine/model.

The test suite checks vocabulary handling, shifted training batches, invalid
configuration/data, causal masking, loss reduction with Adam, dropout/evaluation,
batched generation beyond the context window, and model save/load (including
exact logits, Unicode vocabulary, overwrite protection, and incomplete archives).
It also checks Adam checkpoint validation, version-1 compatibility, and exact
agreement of resumed and uninterrupted training on deterministic batches with
dropout disabled, including non-default learning rate and Adam beta settings.
Validation checks cover fixed windows, token-weighted averaging of the short
final window, deterministic evaluation with dropout configured, and invalid inputs.


## Download DailyDialog with turn annotations

The preparation script uses Python 3's standard library (no pip packages):

```sh
python3 scripts/prepare_dailydialog.py
cabal run nanochat-train -- models/dailydialog.model data/dailydialog/train.txt 500
cabal run nanochat -- models/dailydialog.model
```

It downloads a pinned revision of [roskoN/dailydialog](https://huggingface.co/datasets/roskoN/dailydialog)
and writes separate `train`, `validation`, and `test` files in `data/dailydialog`:

- `.txt`: one conversation per line, with alternating `[USER]` and `[BOT]` turns.
  `[END]` follows each bot response and terminates an unpaired final user turn.
- `.jsonl`: complete conversations with one-based turn numbers, speaker A/B,
  assigned user/assistant role, text, and the original numeric act/emotion labels.
- `manifest.json`: source revision, archive hashes, counts, and license attribution.

For example (illustrative text):

```text
[USER] Hello . [BOT] Hi ! [END] [USER] How are you ? [BOT] Fine , thanks . [END]
```

The source already separates utterances; speaker roles are assigned by alternating
turn position, not inferred from dialogue acts or emotions. These are human-human
conversations, so the assistant label is a training convention. All turns,
including an odd final turn, are retained. Existing converted files are protected;
use `--force` to replace them or `--output-dir PATH` for a different directory.

Train only on `train.txt`; use `--validation data/dailydialog/validation.txt` to
measure held-out loss, and keep `test.txt` for a final assessment. The trainer does
not mask loss to assistant tokens. It reads text as a continuous stream, so random windows can cross
conversation boundaries. JSONL keeps the full structure for a future dialogue-aware
loader. The dataset's published license is CC BY-NC-SA 4.0, per its
[loader metadata](https://huggingface.co/datasets/roskoN/dailydialog/blob/main/dailydialog.py).

# my notes and useful links
on Hasktorch:
    https://www.stackbuilders.com/insights/hasktorch-libtorch-haskell-bindings-for-deep-learning-using-ffi/
on transformer: 
    https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
    https://jalammar.github.io/illustrated-transformer/
    https://nlp.seas.harvard.edu/annotated-transformer/#encoder-and-decoder-stacks
    https://github.com/tensorflow/nmt

state space models 
    https://huggingface.co/blog/lbourdois/get-on-the-ssm-train
    https://acingai.com/articles/state-space-models-2026

a dataset with math examples (high school level)
    https://huggingface.co/datasets/openai/gsm8k

    https://github.com/mlabonne/llm-datasets

# cost in function of vocabulary size
Let:
- \(V\) = vocabulary size
- \(D\) = embedding width
- \(B\) = batch size
- \(T\) = sequence length
The final layer multiplies a \((BT)\times D\) matrix by a \(D\times V\) matrix. Counting a multiplication and addition as two FLOPs:
\[
\text{forward cost} \approx 2BTDV
\]Including backpropagation through that layer:
\[
\text{training cost} \approx 6BTDV
\]With your defaults, \(B=8,\ T=128,\ D=128\):
Vocabulary size	Final layer FLOPs per training step, including backward
- 1,000	    0.79 billion
- 10,000	7.9 billion
- 50,000	39 billion
