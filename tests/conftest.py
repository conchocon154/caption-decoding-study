import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capstudy.vocab import Vocabulary  # noqa: E402


@pytest.fixture(scope="session")
def vocab() -> Vocabulary:
    captions = [
        "a man riding a wave on a surfboard",
        "a cat sitting on a wooden table",
        "two dogs running across a green field",
        "a plate of food on a table",
        "a man riding a horse on a beach",
    ] * 5   # repeated so every word clears min_count
    return Vocabulary.build(captions, min_count=2)


@pytest.fixture(scope="session")
def tiny_model(vocab):
    """A small untrained decoder - enough to exercise the decoding code paths."""
    from capstudy.model import CaptionDecoder

    torch.manual_seed(0)
    return CaptionDecoder(feature_dim=16, embed_size=12, hidden_size=16,
                          vocab_size=len(vocab), num_layers=1, dropout=0.0).eval()
