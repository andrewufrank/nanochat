#!/usr/bin/env python3
"""Download DailyDialog and annotate its existing turns; Python stdlib only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

SOURCE = "https://huggingface.co/datasets/roskoN/dailydialog"
REVISION = "5214b2a66405abf87fd229e5c1007985501ffe3e"
SPLITS = ("train", "validation", "test")


def read_dialogues(archive, split):
    with ZipFile(io.BytesIO(archive)) as zipped:
        def lines(kind):
            names = [n for n in zipped.namelist() if Path(n).name == f"dialogues{kind}_{split}.txt"]
            if len(names) != 1:
                raise ValueError(f"Expected one dialogues{kind}_{split}.txt; found {names}")
            return zipped.read(names[0]).decode("utf-8").splitlines()

        utterances, acts, emotions = lines(""), lines("_act"), lines("_emotion")
        if not len(utterances) == len(acts) == len(emotions):
            raise ValueError(f"{split}: annotation row counts differ")
        for number, (line, act_line, emotion_line) in enumerate(zip(utterances, acts, emotions)):
            pieces = line.split("__eou__")
            if not pieces[-1].strip():
                pieces.pop()
            texts = [" ".join(piece.split()) for piece in pieces]
            act_ids = list(map(int, act_line.split()))
            emotion_ids = list(map(int, emotion_line.split()))
            if not texts or not all(texts) or not len(texts) == len(act_ids) == len(emotion_ids):
                raise ValueError(f"{split} row {number}: empty turn or mismatched annotations")
            turns = []
            for index, (text, act, emotion) in enumerate(zip(texts, act_ids, emotion_ids)):
                if any(marker in text for marker in ("[USER]", "[BOT]", "[END]", "[UNK]")):
                    raise ValueError(f"{split} row {number}: reserved marker in source text")
                turns.append({"turn": index + 1, "speaker": "A" if index % 2 == 0 else "B",
                              "role": "user" if index % 2 == 0 else "assistant",
                              "text": text, "act": act, "emotion": emotion})
            yield {"id": f"{split}_{number}", "split": split, "turns": turns}


def training_text(dialogue):
    parts = []
    turns = dialogue["turns"]
    for index, turn in enumerate(turns):
        parts.extend(("[USER]" if turn["role"] == "user" else "[BOT]", turn["text"]))
        # Match the CLI's '[USER] ... [BOT]' prompt. End each bot response,
        # and also terminate an unpaired final user turn at the dialogue boundary.
        if turn["role"] == "assistant" or index == len(turns) - 1:
            parts.append("[END]")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/dailydialog"))
    parser.add_argument("--force", action="store_true", help="Replace existing converted files")
    args = parser.parse_args()
    out = args.output_dir
    targets = [out / f"{split}.{suffix}" for split in SPLITS for suffix in ("txt", "jsonl")]
    targets.append(out / "manifest.json")
    if not args.force and any(path.exists() for path in targets):
        parser.error("Output files already exist; use a new directory or --force.")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE, "revision": REVISION, "license": "CC BY-NC-SA 4.0",
                "roles": "Alternating speakers A/B mapped to user/assistant; not original assistant labels",
                "splits": {}}
    for split in SPLITS:
        url = f"{SOURCE}/resolve/{REVISION}/{split}.zip"
        print(f"Downloading {split}...", flush=True)
        with urlopen(url, timeout=60) as response:
            archive = response.read()
        dialogues = list(read_dialogues(archive, split))
        with (out / f"{split}.txt").open("w", encoding="utf-8") as text_file, \
             (out / f"{split}.jsonl").open("w", encoding="utf-8") as json_file:
            for dialogue in dialogues:
                text_file.write(training_text(dialogue) + "\n")
                json_file.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
        count = sum(len(dialogue["turns"]) for dialogue in dialogues)
        manifest["splits"][split] = {"url": url, "sha256": hashlib.sha256(archive).hexdigest(),
                                     "dialogues": len(dialogues), "turns": count}
        print(f"  {len(dialogues):,} dialogues, {count:,} turns -> {out / (split + '.txt')}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
