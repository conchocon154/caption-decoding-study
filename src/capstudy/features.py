"""Extract and cache frozen ResNet-50 image features.

The encoder never trains in this study, so its output for a given image never
changes.  Running it once and caching the 2048-d vectors turns every later
experiment from an image-loading job into a matrix lookup: the decoder trains in
minutes instead of hours, and the four decoding strategies are compared on
identical inputs by construction.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

FEATURE_DIM = 2048


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, file_names: list[str]) -> None:
        self.root = root
        self.file_names = file_names
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.transform = weights.transforms()

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int):
        path = self.root / self.file_names[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), index


def build_encoder(device: torch.device) -> nn.Module:
    """ResNet-50 with the classifier removed, leaving the pooled 2048-d vector."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    encoder = nn.Sequential(*list(model.children())[:-1])
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


@torch.no_grad()
def extract(image_root: Path, image_ids: list[int], file_names: dict[int, str],
            batch_size: int = 64, workers: int = 4, progress=None
            ) -> tuple[torch.Tensor, dict[int, int]]:
    """Return the feature matrix and a map from image id to its row."""
    device = pick_device()
    encoder = build_encoder(device)

    usable = [image_id for image_id in image_ids if image_id in file_names]
    names = [file_names[image_id] for image_id in usable]
    loader = DataLoader(
        ImageFolderDataset(image_root, names),
        batch_size=batch_size, num_workers=workers, shuffle=False,
    )

    features = torch.zeros(len(usable), FEATURE_DIM, dtype=torch.float32)
    for images, indices in loader:
        batch = encoder(images.to(device)).flatten(1).float().cpu()
        features[indices] = batch
        if progress is not None:
            progress.update(len(indices))

    return features, {image_id: row for row, image_id in enumerate(usable)}


def save(features: torch.Tensor, index: dict[int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "index": index}, path)


def load(path: Path) -> tuple[torch.Tensor, dict[int, int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["features"], payload["index"]
