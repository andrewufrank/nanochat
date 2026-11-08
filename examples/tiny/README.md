# A small CPU experiment

These are synthetic examples written for this project. No download is needed.
Each input asks for a description such as `describe a red cup`; the target
response is `the cup is red`. Training contains twelve color/object combinations.
Validation holds out four different combinations while using the same vocabulary.
This is an exercise in learning a small pattern, not a realistic chatbot benchmark.

Run these commands from the repository root after completing the build setup in
[the main README](../../readme.md):

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cabal run nanochat-train -- --tiny --validation examples/tiny/validation.txt models/tiny-200.model examples/tiny/train.txt 200
cabal run nanochat -- models/tiny-200.model
```

At the chat prompt, try `describe a red boat`. This combination is held out from
training. Press Enter on an empty line to leave chat. Generated text is sampled,
so a correct response is not guaranteed, even after training loss decreases.

`--tiny` selects `tinyConfig`: context length 16, embedding width 32, two heads,
one block, and training dropout 0.1. Batch size is eight and learning rate is
0.001. The vocabulary is built from `train.txt` only. All validation words are in
that vocabulary, so the validation `[UNK]` count should be zero.

Continue the same model for 200 more steps:

```sh
cabal run nanochat-train -- --validation examples/tiny/validation.txt --resume models/tiny-200.model models/tiny-400.model examples/tiny/train.txt 200
cabal run nanochat -- models/tiny-400.model
```

Resume restores the tiny configuration automatically; do not pass `--tiny` when
resuming. The output file must have a new name. To repeat an experiment, choose
new filenames rather than overwriting earlier checkpoints.

## What to look for

The trainer prints two kinds of measurements:

- `Step ... | loss ...` is one randomly sampled training batch, with dropout on,
  measured before that step's update. It can rise or fall from one report to the next.
- `Before training` and `After training` report mean training-set evaluation loss
  and validation loss, both with dropout off. These are the scores to compare.

Both evaluation passes use fixed windows and include all target tokens after the
first. They include user words and markers as well as bot words; they are not
scores of bot answers alone. Context resets every 16 input positions, and windows
can cross line boundaries. See the [walkthrough](../../docs/learning.md#measuring-learning)
for the exact procedure.

Training-set evaluation loss should generally decrease as the model learns this
small corpus. Validation loss may improve too, but need not track it. If training
loss keeps falling while validation loss stalls or rises, the model is fitting
training-specific patterns without improving on the held-out combinations. A
particular 200-step run is not guaranteed to show overfitting.

Initialization, training batches, dropout, and generated tokens are random.
Expect different numbers on different runs. Within the same environment,
evaluating the same saved model on the same data is deterministic. RNG state is
not stored in checkpoints, so resuming does not reproduce an uninterrupted run's
random choices. Validation scores are printed, not stored in checkpoint metadata.

## Small exercises

1. Resume for another 200 steps and compare the two evaluation losses.
2. Change a word in a copy of the validation file to an unseen word. Observe the
   `[UNK]` report; the model cannot learn a new vocabulary entry from validation.
3. Compare a held-out prompt with one copied from training. Explain why a lower
   training loss alone is insufficient evidence of generalization.
4. Inspect `sampleBatch`: eight sequences of sixteen tokens make 128 next-token
   predictions per update, not eight words.
