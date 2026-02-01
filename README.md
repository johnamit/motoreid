# 🏍️ MotoGP Team Detection & Re-ID

<div align="center">
<img src="assets/mgp-detect-banner.png" alt="MotoGP Team Detection" width="900">

<a href="https://github.com/johnamit/mgp-detect">
  <img src="https://img.shields.io/badge/Status-Active_Development-orange?style=for-the-badge" alt="Status">
</a>
</div>

A deep learning pipeline for **real-time MotoGP team detection, tracking, and re-identification** from race broadcast footage. This system combines **YOLOv8** for robust object localization with **DINOv3** (Vision Transformer) embeddings for semantic team classification. It addresses specific challenges in high-speed sports computer vision: persistent identity tracking across extreme occlusions, rapid camera cuts, and motion blur.

<p align="center">
  <a href="#overview"><img src="https://img.shields.io/badge/Overview-111111?style=for-the-badge" alt="Overview"></a>
  <a href="#dataset-construction--training"><img src="https://img.shields.io/badge/Data_Pipeline-111111?style=for-the-badge" alt="Dataset"></a>
  <a href="#pipeline-architecture"><img src="https://img.shields.io/badge/Architecture-111111?style=for-the-badge" alt="Pipeline"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Install-111111?style=for-the-badge" alt="Installation"></a>
  <a href="#usage"><img src="https://img.shields.io/badge/Usage-111111?style=for-the-badge" alt="Usage"></a>
  <a href="#teams"><img src="https://img.shields.io/badge/Teams-111111?style=for-the-badge" alt="Teams"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-111111?style=for-the-badge" alt="Citation"></a>
</p>

---

## Development Status

This project is currently in **active development**.  
Current objectives: 
- Finalising the REID memory bank strategy for long-term occlusion handling.
- Improving "Hard Negative" performance on rear-view angles and implementing Kalman Filter integration for trajectory smoothing.
- Demonstration video to be added soon.

---

## Overview

This project implements a multi-stage perception pipeline designed for high-velocity agents:

1. **Detection** — **YOLOv8** (fine-tuned) localizes motorcycles in each frame.
2. **Feature Extraction** — **DINOv3 ViT-S/16** extracts dense semantic features from detected regions, leveraging its self-supervised understanding of object geometry.
3. **Classification** — A lightweight Logistic Regression head predicts team identity from the high-dimensional DINO embeddings.
4. **Re-ID & Tracking** — **ByteTrack** handles short-term association, while a **Cosine Similarity Memory Bank** enables long-term re-identification after occlusions.
5. **State Estimation** — EMA embedding smoothing and velocity-based position prediction reduce ID switching during glare or blur.

**Key Features:**
- Label locking after high-confidence agreement
- Re-ID matching via combined visual + spatial similarity
- Position trajectory tracking with velocity prediction
- Global constraint enforcement (max 2 bikes per team)
- Visual debug overlay (trajectories, Re-ID events)

---

## Dataset Construction & Training

Unlike general-purpose object detectors, this project relies on a highly specialized, manually curated dataset to handle the specific livery variations of the 2025 MotoGP grid.

### 1. Data Ingestion (YouTube)

Raw footage was sourced from high-definition broadcast highlights (1080p/60fps) on YouTube. We developed custom `ffmpeg` scripts (`src/extract_frames.py`) to extract frames at specific intervals, ensuring a diverse range of lighting conditions (sunny, overcast) and camera angles (onboard, trackside, helicopter).

### 2. Annotation (Roboflow)

To train the YOLOv8 detector, we created a custom bounding box dataset:

| Aspect | Details |
|--------|---------|
| **Platform** | [Roboflow](https://app.roboflow.com/amitjohnworkspace/motogp-team-detection/models) for annotation management and augmentation |
| **Labeling** | Manually annotated 501 instances (404 motorcycles, 97 null), specifically filtering out "soft" targets (e.g., pit lane scooters, pedestrians) to focus on the riders and their racing prototypes |
| **Augmentation** | Applied random rotations, exposure adjustments, and noise injection to simulate broadcast compression artifacts |
| **Export** | YOLOv8-compatible format with train/valid/test splits |

### 3. Identity Dataset

For the DINOv3 classifier, we built a reference library of **~700 high-quality crops**:

| Stage | Tool | Description |
|-------|------|-------------|
| **Harvesting** | `src/harvest_bikes.py` | Auto-crop bikes from video stream using trained YOLO model |
| **Cleaning** | Manual curation | Remove motion blur, ensure balanced class distribution across all 11 teams |
| **Hard Negatives** | Targeted mining | Specifically captured "difficult" angles (direct rear view, extreme lean angles) to force the model to learn geometric features beyond just side-fairing logos |

### 4. Training Pipeline

```
YouTube Highlights (1080p/60fps)
        │
        ▼
┌───────────────────┐
│  Frame Extraction │ ──► extract_frames.py
│  (ffmpeg @ 5fps)  │
└───────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌───────────────────┐        ┌───────────────────┐
│  Roboflow         │        │  harvest_bikes.py │
│  (BBox Annotation)│        │  (Auto-crop)      │
└───────────────────┘        └───────────────────┘
        │                              │
        ▼                              ▼
┌───────────────────┐        ┌───────────────────┐
│  YOLOv8           │        │  Manual Sorting   │
│  Fine-tuning      │        │  (11 Team Folders)│
└───────────────────┘        └───────────────────┘
        │                              │
        ▼                              ▼
┌───────────────────┐        ┌───────────────────┐
│  Motorcycle       │        │  DINOv3 + LogReg  │
│  Detector         │        │  Team Classifier  │
└───────────────────┘        └───────────────────┘
```

---

## Prerequisites

- **Python** 3.10+
- **PyTorch** 2.4+ (Required for DINOv3 compatibility)
- **CUDA** 12.x (Highly recommended for real-time performance)
- **Operating System**: Linux (tested on Ubuntu 22.04)

**Tested on:** NVIDIA RTX 3090 • AMD Ryzen 7 • 32GB RAM

---

## Project Structure

```
MotoGP-Team-Detection/
├── assets/                     # Banner images and visual assets
├── data/
│   ├── input/
│   │   ├── race_highlights/    # Source video files
│   │   ├── race_frames/        # Extracted frames for training
│   │   │   ├── 2025_spanish_gp/
│   │   │   ├── 2025_italian_gp/
│   │   │   ├── 2025_qatar_gp/
│   │   │   └── 2025_german_gp_sprint/
│   │   └── null_samples/       # Background/non-bike samples
│   ├── output/                 # Annotated video results
│   │   ├── annotated_races_yolo/
│   │   └── annotated_races_yolodino/
│   └── teams/                  # The Identity Dataset (Sorted by Team)
│       ├── aprilia_factory/
│       ├── aprilia_trackhouse/
│       ├── ducati_lenovo/
│       ├── ducati_gresini/
│       ├── ducati_vr46/
│       ├── honda_hrc/
│       ├── honda_lcr/
│       ├── ktm_factory/
│       ├── ktm_tech3/
│       ├── yamaha_monster/
│       └── yamaha_pramac/
├── models/
│   ├── DINO/                   # DINOv3 backbone weights
│   └── YOLO/                   # Fine-tuned YOLOv8 weights
├── notebooks/
│   └── teams_dist.ipynb        # Team distribution analysis
├── runs/
│   ├── classifier/             # Trained Scikit-Learn team classifiers
│   └── detect/                 # YOLO training logs
├── src/
│   ├── main.py                 # CORE PIPELINE (Detection + Re-ID Logic)
│   ├── detector.py             # YOLO-only inference check
│   ├── train_detector.py       # YOLO fine-tuning script
│   ├── train_identity_model.py # DINOv3 feature extraction & classifier training
│   ├── extract_frames.py       # Frame extraction from videos
│   ├── harvest_bikes.py        # Dataset creation tool (auto-crop)
│   └── collect_nulls.py        # Collect background/null samples
├── mgp_env.yaml                # Conda environment definition
└── requirements.txt            # Python dependencies
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/johnamit/mgp-detect.git
cd mgp-detect
```

### 2. Create Environment

**Using Conda (Recommended):**

```bash
conda env create -f mgp_env.yaml
conda activate mgp_env
```

### 3. Setup Dependencies

This project uses **DINOv3**, which requires a local clone of the official repository:

```bash
# Clone DINOv3 into the project root
git clone https://github.com/facebookresearch/dinov3.git

# Download Pre-trained Weights (ViT-Small)
mkdir -p models/DINO
wget -O models/DINO/dinov3_vits16_pretrain_lvd1689m.pth \
    https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/resolve/main/dinov3_vits16_pretrain_lvd1689m.pth
```

---

## Usage

### Full Pipeline (Detection + Re-ID)

Run the complete pipeline. This script initializes the models, processes the video with temporal smoothing, and outputs annotated results.

```bash
python src/main.py \
    --source data/input/race_highlights/spanish_gp_2025.mp4 \
    --output data/output/annotated_races_yolodino/spanish_gp.mp4 \
    --yolo_weights models/YOLO/best.pt \
    --classifier_path runs/classifier/dinov3_identity_model.pkl \
    --dino_weights models/DINO/dinov3_vits16_pretrain_lvd1689m.pth \
    --conf_thresh 0.25 \
    --draw_trajectory \
    --draw_reid_events
```

**Key Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | Required | Path to input video |
| `--output` | str | `data/output/...` | Path for annotated output |
| `--conf_thresh` | float | `0.25` | YOLO confidence threshold |
| `--lock_threshold` | float | `0.85` | Agreement ratio to lock team label |
| `--reid_visual_thresh` | float | `0.80` | Cosine similarity for Re-ID matching |
| `--draw_trajectory` | flag | - | Draw position history trails |
| `--draw_reid_events` | flag | - | Flash Re-ID match events |

### YOLO-Only Detection

For simple detection without team classification:

```bash
python src/detector.py \
    --video data/input/race_highlights/race.mp4 \
    --output data/output/annotated_races_yolo/race_annotated.mp4
```

### Train Team Classifier

If you add new images to `data/teams/`, retrain the identity head:

```bash
python src/train_identity_model.py \
    --data_dir data/teams \
    --model_dir runs/classifier \
    --model_weights models/DINO/dinov3_vits16_pretrain_lvd1689m.pth \
    --batch_size 64
```

### Extract Frames from Video

```bash
python src/extract_frames.py \
    --video data/input/race_highlights/race.mp4 \
    --output data/input/race_frames/race_name \
    --fps 5
```

### Harvest Bike Crops

Auto-crop detected bike regions for building the identity dataset:

```bash
python src/harvest_bikes.py \
    --frames_dir data/input/race_frames/race_name \
    --output_dir data/teams/unlabeled \
    --yolo_weights models/YOLO/yolov8m.pt
```

---

## Pipeline

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT VIDEO FRAME                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    YOLOv8 DETECTION                                 │
│                    (Motorcycle Localization)                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DINOv3 ViT-S/16                                  │
│                    (Feature Extraction)                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────┐     ┌───────────────────────────────────┐
│   TEAM CLASSIFIER         │     │   BYTETRACK + MEMORY BANK         │
│   (Logistic Regression)   │     │   (Short + Long Term Tracking)    │
└───────────────────────────┘     └───────────────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LABEL VOTING + LOCKING                           │
│                    (Confidence-based agreement)                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ANNOTATED OUTPUT                                 │
│                    (Team labels, trajectories, Re-ID events)        │
└─────────────────────────────────────────────────────────────────────┘
```

### Re-ID Strategy

The system maintains identity through:

1. **EMA Embedding Smoothing** — Running average of appearance features reduces noise from motion blur
2. **Velocity Prediction** — Extrapolate position when occluded based on historical trajectory
3. **Memory Bank** — Store embeddings of lost tracks for later matching (up to 300 frames)
4. **Combined Similarity** — Visual (cosine) + Spatial (L2 distance) matching with configurable weights

---

## Teams

The system identifies all 11 teams from the 2025 MotoGP grid:

| Manufacturer | Teams | Color Code |
|--------------|-------|------------|
| **Ducati** | Lenovo, Gresini, VR46 | Red / Orange-Red / Yellow |
| **Aprilia** | Factory, Trackhouse | Green / Teal |
| **KTM** | Factory, Tech3 | Orange / Dark Orange |
| **Yamaha** | Monster, Pramac | Blue / Light Blue |
| **Honda** | HRC, LCR | Black / Silver |
```python
TEAM_COLORS = {
    'aprilia_factory':    (0, 255, 0),      # Green
    'aprilia_trackhouse': (0, 200, 100),    # Teal
    'ducati_lenovo':      (0, 0, 255),      # Red
    'ducati_gresini':     (0, 100, 255),    # Orange-Red
    'ducati_vr46':        (0, 255, 255),    # Yellow
    'honda_hrc':          (0, 0, 0),        # Black
    'honda_lcr':          (200, 200, 200),  # Light Gray
    'ktm_factory':        (0, 165, 255),    # Orange
    'ktm_tech3':          (0, 140, 255),    # Dark Orange
    'yamaha_monster':     (255, 0, 0),      # Blue
    'yamaha_pramac':      (255, 100, 100),  # Light Blue
}
```

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{mgp-detect,
  author = {Amit John},
  title = {MotoGP Team Detection: Deep Re-ID for High-Velocity Agents},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/johnamit/mgp-detect}
}
```

**Models Used:**

```bibtex
@article{simeoni2025dinov3,
  title={DINOv3: Learning Scalable Visual Features without Supervision},
  author={Sim{\'e}oni, Oriane and others},
  journal={arXiv preprint arXiv:2508.10104},
  year={2025}
}

@software{ultralytics2023yolov8,
  author = {Jocher, Glenn and others},
  title = {Ultralytics YOLOv8},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

## License

This project is released under the MIT License.

**Note:** MotoGP broadcast footage is copyrighted material. This project is intended for **educational and research purposes only**. All models were trained on fair-use excerpts for non-commercial analysis.
