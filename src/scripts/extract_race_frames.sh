#!/usr/bin/env bash
set -e

mkdir -p data/race_frames

python src/data/extract_frames.py --outdir data/race_frames

echo "Done: frames extracted to data/race_frames/"
