"""Train the caption decoder on cached image features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CaptionDataset, collate
from .features import pick_device
from .model import CaptionDecoder
from .vocab import Vocabulary


@dataclass
class TrainConfig:
    embed_size: int = 256
    hidden_size: int = 512
    num_layers: int = 1
    dropout: float = 0.3
    batch_size: int = 256
    learning_rate: float = 3e-4
    epochs: int = 12
    patience: int = 2
    max_len: int = 20
    seed: int = 42


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, captions in loader:
            features = features.to(device)
            captions = captions.to(device)
            logits = model(features, captions)
            # Targets are the caption shifted by one: predict word t+1 at step t.
            loss = criterion(logits.reshape(-1, logits.size(-1)), captions.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            tokens = int((captions != criterion.ignore_index).sum())
            total_loss += float(loss) * tokens
            total_tokens += tokens
    return total_loss / max(total_tokens, 1)


def train_decoder(features, feature_index, captions, splits, vocab: Vocabulary,
                  config: TrainConfig, checkpoint: Path, progress=None):
    """Fit the decoder, keeping the weights from the best validation epoch."""
    torch.manual_seed(config.seed)
    device = pick_device()

    loaders = {}
    for name in ("train", "val"):
        dataset = CaptionDataset(features, feature_index, captions,
                                 splits[name].image_ids, vocab, config.max_len)
        loaders[name] = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=(name == "train"),
            collate_fn=lambda batch: collate(batch, vocab.pad_id),
        )

    model = CaptionDecoder(features.size(1), config.embed_size, config.hidden_size,
                           len(vocab), config.num_layers, config.dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history, best_loss, waited = [], float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4)})
        if progress is not None:
            progress(epoch, train_loss, val_loss)

        if val_loss < best_loss - 1e-4:
            best_loss, waited = val_loss, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "feature_dim": features.size(1),
                        "vocab_size": len(vocab)}, checkpoint)
        else:
            waited += 1
            # Stop once validation stops improving: further epochs only sharpen
            # the model on the training captions, which would change what the
            # decoding comparison is measuring.
            if waited >= config.patience:
                break

    return model, history, best_loss


def load_decoder(checkpoint: Path, device=None) -> CaptionDecoder:
    device = device or pick_device()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["config"]
    model = CaptionDecoder(payload["feature_dim"], config["embed_size"],
                           config["hidden_size"], payload["vocab_size"],
                           config["num_layers"], config["dropout"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
