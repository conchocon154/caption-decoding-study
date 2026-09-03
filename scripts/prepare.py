"""Unpack COCO, build the splits and vocabulary, and cache image features."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm  # noqa: E402

from capstudy import features as feature_tools  # noqa: E402
from capstudy.data import load_annotations, make_splits  # noqa: E402
from capstudy.vocab import Vocabulary  # noqa: E402

DATA = ROOT / "data"


def unzip_images(archive: Path, target: Path) -> Path:
    images = target / "val2014"
    if images.exists() and any(images.iterdir()):
        return images
    if not archive.exists():
        raise SystemExit(f"{archive} not found - run scripts/download.sh first")
    print(f"unpacking {archive.name} ...")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=5,
                        help="drop words appearing fewer times in the training split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    annotations = DATA / "annotations" / "captions_val2014.json"
    if not annotations.exists():
        raise SystemExit(f"{annotations} not found - unzip the annotations first")

    file_names, captions = load_annotations(annotations)
    splits = make_splits(list(captions), seed=args.seed)
    print(f"{len(file_names):,} images, {sum(len(c) for c in captions.values()):,} captions")
    for name, split in splits.items():
        print(f"  {name:5} {len(split.image_ids):6,} images")

    # Vocabulary comes from training captions only - see capstudy/vocab.py.
    training_captions = [c for image_id in splits["train"].image_ids
                         for c in captions[image_id]]
    vocab = Vocabulary.build(training_captions, min_count=args.min_count)
    print(f"vocabulary: {len(vocab):,} tokens (min_count={args.min_count})")

    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    vocab.save(processed / "vocab.json")
    (processed / "splits.json").write_text(
        json.dumps({name: split.image_ids for name, split in splits.items()}),
        encoding="utf-8")

    feature_path = processed / "features.pt"
    if feature_path.exists():
        print(f"features already cached at {feature_path}")
        return

    images = unzip_images(DATA / "raw" / "val2014.zip", DATA)
    all_ids = [i for split in splits.values() for i in split.image_ids]
    print(f"extracting ResNet-50 features for {len(all_ids):,} images ...")
    with tqdm(total=len(all_ids), unit="img") as bar:
        matrix, index = feature_tools.extract(
            images, all_ids, file_names,
            batch_size=args.batch_size, workers=args.workers, progress=bar)
    feature_tools.save(matrix, index, feature_path)
    print(f"features {tuple(matrix.shape)} -> {feature_path}")


if __name__ == "__main__":
    main()
