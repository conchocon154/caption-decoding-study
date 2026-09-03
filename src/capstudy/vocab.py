"""Caption vocabulary.

Built from the training split only.  Building it over the whole dataset would
leak test captions into the model's output space: words that only ever appear in
test references would become predictable, and every metric would be flattered.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(caption: str) -> list[str]:
    """Lowercase and split into word tokens, dropping punctuation."""
    return _TOKEN.findall(caption.lower())


class Vocabulary:
    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {token: index for index, token in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def start_id(self) -> int:
        return self.stoi[START]

    @property
    def end_id(self) -> int:
        return self.stoi[END]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @classmethod
    def build(cls, captions: list[str], min_count: int = 5) -> "Vocabulary":
        counts = Counter(token for caption in captions for token in tokenize(caption))
        kept = sorted(word for word, count in counts.items() if count >= min_count)
        return cls(SPECIALS + kept)

    def encode(self, caption: str, max_len: int) -> list[int]:
        """`<start> tokens... <end>`, truncated to `max_len` including both."""
        tokens = tokenize(caption)[: max_len - 2]
        ids = [self.start_id] + [self.stoi.get(t, self.unk_id) for t in tokens] + [self.end_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Ids back to a sentence, stopping at `<end>` and dropping specials."""
        words = []
        for index in ids:
            token = self.itos[index]
            if token == END:
                break
            if token in (PAD, START):
                continue
            words.append(token)
        return " ".join(words)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))
