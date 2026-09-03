"""COCO captions: loading, splitting, and the caption dataset.

Only val2014 is used.  The original project trained on train2014 (13 GB); this
study needs a controlled comparison of decoding strategies rather than the best
possible captioner, and val2014's 40k images with five references each is ample
for that while staying reproducible on a laptop.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .vocab import Vocabulary


@dataclass(frozen=True)
class Split:
    name: str
    image_ids: list[int]


def load_annotations(path: Path) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Return image_id -> file name, and image_id -> its reference captions."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    file_names = {image["id"]: image["file_name"] for image in payload["images"]}
    captions: dict[int, list[str]] = {}
    for annotation in payload["annotations"]:
        captions.setdefault(annotation["image_id"], []).append(annotation["caption"])
    return file_names, captions


def make_splits(image_ids: list[int], seed: int = 42,
                fractions: tuple[float, float] = (0.79, 0.10)) -> dict[str, Split]:
    """Split by IMAGE, never by caption.

    Each image carries five captions of the same scene.  Splitting on captions
    would put four references in training and the fifth in test, so the model
    would be scored on a sentence it had all but memorised.
    """
    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * fractions[0])
    val_end = train_end + int(n * fractions[1])
    return {
        "train": Split("train", shuffled[:train_end]),
        "val": Split("val", shuffled[train_end:val_end]),
        "test": Split("test", shuffled[val_end:]),
    }


class CaptionDataset(Dataset):
    """(image feature, caption ids) pairs - one row per caption."""

    def __init__(self, features: torch.Tensor, feature_index: dict[int, int],
                 captions: dict[int, list[str]], image_ids: list[int],
                 vocab: Vocabulary, max_len: int = 20) -> None:
        self.features = features
        self.vocab = vocab
        self.max_len = max_len
        self.pairs: list[tuple[int, str]] = [
            (feature_index[image_id], caption)
            for image_id in image_ids
            if image_id in feature_index
            for caption in captions.get(image_id, [])
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        row, caption = self.pairs[index]
        ids = self.vocab.encode(caption, self.max_len)
        return self.features[row], torch.tensor(ids, dtype=torch.long)


def collate(batch, pad_id: int):
    """Pad a batch of variable-length captions to the longest in the batch."""
    features = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    longest = max(len(c) for c in captions)
    padded = torch.full((len(captions), longest), pad_id, dtype=torch.long)
    for row, caption in enumerate(captions):
        padded[row, : len(caption)] = caption
    return features, padded
