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
cabal run nanochat -- chat_dataset.txt 500
```

The optional arguments are the dataset path (default `chat_dataset.txt`) and a
positive training-step count (default 500). The dataset must contain at least 129
whitespace-separated tokens for the default context size of 128. For dialogue
experiments, use text such as `[USER] hello [BOT] hi there [END]`, repeated across
your examples. The tokenizer is a simple word tokenizer with reserved `[UNK]`,
`[USER]`, `[BOT]`, and `[END]` tokens; vocabulary size is calculated from distinct
words plus those markers. It does not use BPE or tiktoken.

Training runs on CPU with batches of eight. Adjust `defaultConfig` in
`src/NanoChat.hs` for smaller experiments; embedding width must be divisible by
head count. Dropout is enabled only during training. After training, enter chat
messages; a blank line or EOF exits. Output is capped at 30 new tokens and is
displayed up to the first `[END]`. Every run trains from scratch; weights and the
tokenizer are not yet saved. Tiny datasets and short training runs are useful for
experiments, but will not produce a capable chatbot.

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
and batched generation beyond the context window.


# my notes
on Hasktorch:
    https://www.stackbuilders.com/insights/hasktorch-libtorch-haskell-bindings-for-deep-learning-using-ffi/
 