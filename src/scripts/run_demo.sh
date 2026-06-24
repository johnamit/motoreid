#!/usr/bin/env bash
set -e

ENV="motogp_overtake"
RUN="conda run --no-capture-output -n $ENV python"

YOLO_WEIGHTS="models/YOLO/motogp_yolov8m/weights/best.pt"
CLASSIFIER="models/classifier/dinov3_identity_model.pkl"
DINO_WEIGHTS="models/DINO/dinov3_vits16_pretrain_lvd1689m.pth"

mkdir -p logs/yolo_detection
mkdir -p logs/pipeline

echo "1/6 Detector Only: 2025 British GP"
$RUN src/inference/detector.py \
    --video data/race_highlights/test/grandprix/2025_british_gp.mp4 \
    --yolo_weights $YOLO_WEIGHTS \
    --output results/yolo_detection/2025_british_gp_yolov8m.mp4 \
    --log logs/yolo_detection/2025_british_gp_yolov8m.log

echo "2/6 Detector Only: 2025 Portuguese Q2"
$RUN src/inference/detector.py \
    --video data/race_highlights/test/qualifying/2025_portuguese_q2.mp4 \
    --yolo_weights $YOLO_WEIGHTS \
    --output results/yolo_detection/2025_portuguese_q2_yolov8m.mp4 \
    --log logs/yolo_detection/2025_portuguese_q2_yolov8m.log

echo "3/6 Full Pipeline: 2025 British GP"
$RUN src/inference/main.py \
    --video data/race_highlights/test/grandprix/2025_british_gp.mp4 \
    --output results/pipeline/2025_british_gp_full_pipeline.mp4 \
    --yolo_weights $YOLO_WEIGHTS \
    --classifier_path $CLASSIFIER \
    --dino_weights $DINO_WEIGHTS \
    --log logs/pipeline/2025_british_gp_full_pipeline.log

echo "4/6 Full Pipeline: 2025 Australian GP Sprint"
$RUN src/inference/main.py \
    --video data/race_highlights/test/grandprix/2025_australian_gp_sprint.mp4 \
    --output results/pipeline/2025_australian_gp_sprint_full_pipeline.mp4 \
    --yolo_weights $YOLO_WEIGHTS \
    --classifier_path $CLASSIFIER \
    --dino_weights $DINO_WEIGHTS \
    --log logs/pipeline/2025_australian_gp_sprint_full_pipeline.log

echo "5/6 Full Pipeline: 2025 Portuguese Q2"
$RUN src/inference/main.py \
    --video data/race_highlights/test/qualifying/2025_portuguese_q2.mp4 \
    --output results/pipeline/2025_portuguese_q2_full_pipeline.mp4 \
    --yolo_weights $YOLO_WEIGHTS \
    --classifier_path $CLASSIFIER \
    --dino_weights $DINO_WEIGHTS \
    --log logs/pipeline/2025_portuguese_q2_full_pipeline.log

echo "6/6 Full Pipeline: 2025 Spanish Q2"
$RUN src/inference/main.py \
    --video data/race_highlights/test/qualifying/2025_spanish_q2.mp4 \
    --output results/pipeline/2025_spanish_q2_full_pipeline.mp4 \
    --yolo_weights $YOLO_WEIGHTS \
    --classifier_path $CLASSIFIER \
    --dino_weights $DINO_WEIGHTS \
    --log logs/pipeline/2025_spanish_q2_full_pipeline.log
