# NanoChat in Haskell

A small CPU language-model experiment inspired by
[karpathy/nanochat](https://github.com/karpathy/nanochat). This is an educational
transformer implementation, not a faithful port of upstream NanoChat or a loader
for its checkpoints. It uses learned positional embeddings, pre-layer normalization,
causal multi-head attention, a ReLU feed-forward network, and Adam. Upstream uses a
different architecture and a much larger training/tokenization pipeline.

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
count (default 500). Chat takes only a saved model filename and does not need the
original dataset. The dataset must contain at least 129
whitespace-separated tokens for the default context size of 128. For dialogue
experiments, use text such as `[USER] hello [BOT] hi there [END]`, repeated across
your examples. The tokenizer is a simple word tokenizer with reserved `[UNK]`,
`[USER]`, `[BOT]`, and `[END]` tokens; vocabulary size is calculated from distinct
words plus those markers. It does not use BPE or tiktoken.

Training runs on CPU with batches of eight. Adjust `defaultConfig` in
`src/NanoChat.hs` for smaller experiments; embedding width must be divisible by
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

Use a different filename for each experiment; existing files are refused before
training starts. Parent directories are created automatically, and saving uses a
temporary file before publishing the completed archive. For example:

```sh
cabal run nanochat-train -- models/dialogue-2k.model data/dailydialog/train.txt 2000
cabal run nanochat-train -- models/dialogue-10k.model data/dailydialog/train.txt 10000
cabal run nanochat -- models/dialogue-2k.model
cabal run nanochat -- models/dialogue-10k.model
```

These are independent training runs. Saving currently happens at the end of a
successful run; interrupted runs are not checkpointed. Adam state is not saved,
and resuming training remains a future feature. The files are for this Haskell
model format, not upstream NanoChat/nanoGPT checkpoints.

The training entry point is `app/train/Main.hs`, chat is `app/chat/Main.hs`,
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

Train only on `train.txt`; keep validation/test separate. The current Haskell
trainer does not yet evaluate those held-out splits or mask loss to assistant
tokens. It reads text as a continuous stream, so random windows can cross
conversation boundaries. JSONL keeps the full structure for a future dialogue-aware
loader. The dataset's published license is CC BY-NC-SA 4.0, per its
[loader metadata](https://huggingface.co/datasets/roskoN/dailydialog/blob/main/dailydialog.py).

# my notes and useful links
on Hasktorch:
    https://www.stackbuilders.com/insights/hasktorch-libtorch-haskell-bindings-for-deep-learning-using-ffi/
from stella: 
    https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
    https://jalammar.github.io/illustrated-transformer/

    https://github.com/tensorflow/nmt

state space models 
    https://huggingface.co/blog/lbourdois/get-on-the-ssm-train
    https://acingai.com/articles/state-space-models-2026

a dataset with math examples (high school level)
    https://huggingface.co/datasets/openai/gsm8k

    https://github.com/mlabonne/llm-datasets