#!/usr/bin/env bash
# Fetch COCO 2014 val images and captions. About 6.5 GB.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"

echo "annotations (253 MB) ..."
[ -f annotations_trainval2014.zip ] || \
  curl -SL -o annotations_trainval2014.zip \
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
unzip -q -o annotations_trainval2014.zip -d ..

echo "val2014 images (6.2 GB) ..."
[ -f val2014.zip ] || \
  curl -SL -o val2014.zip "http://images.cocodataset.org/zips/val2014.zip"

echo "done. Next: python scripts/prepare.py"
