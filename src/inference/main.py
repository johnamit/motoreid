"""
MotoGP Team Detection Pipeline
===============================

Frame-by-frame detection and team classification using
YOLOv8 (detection) + DINOv3 embeddings + Logistic Regression.

Architecture:
    1. Detection + Tracking: YOLOv8 + BoT-SORT with ReID
    2. Embedding: DINOv3 ViT-S/16 (Semantic representation)
    3. Classification: Logistic Regression over DINO embeddings
"""

import os
import sys
import cv2
import torch
import joblib
import logging
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.transforms import v2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dinov3'))
from dinov3.hub.backbones import dinov3_vits16


TEAM_COLORS = {
    'aprilia_factory':    (0, 255, 0),
    'aprilia_trackhouse': (0, 200, 100),
    'ducati_lenovo':      (0, 0, 255),
    'ducati_gresini':     (0, 100, 255),
    'ducati_vr46':        (0, 255, 255),
    'honda_hrc':          (0, 0, 0),
    'honda_lcr':          (200, 200, 200),
    'ktm_factory':        (0, 165, 255),
    'ktm_tech3':          (0, 140, 255),
    'yamaha_monster':     (255, 0, 0),
    'yamaha_pramac':      (255, 100, 100),
    'unknown':            (128, 128, 128),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="results/pipeline/final_race.mp4", help="Path to save annotated video")
    parser.add_argument("--log", type=str, default="logs/pipeline/pipeline.log", help="Path to save log file")

    parser.add_argument("--yolo_weights", type=str, default="models/YOLO/motogp_yolov8m/weights/best.pt", help="YOLOv8 weights")
    parser.add_argument("--classifier_path", type=str, default="models/classifier/dinov3_identity_model.pkl", help="Path to trained team classifier")
    parser.add_argument("--dino_weights", type=str, default="models/DINO/dinov3_vits16_pretrain_lvd1689m.pth", help="Path to DINOv3 backbone weights")

    parser.add_argument("--img_size", type=int, default=224, help="DINOv3 input size")
    parser.add_argument("--tracker", type=str, default="models/YOLO/trackers/botsort_reid.yaml", help="Tracker config YAML")

    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def setup_logger(log_path):
    ensure_parent_dir(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.WARNING)


def make_transform(img_size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class MotoGPPipeline:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f"Initializing Pipeline on {self.device}")

        self._load_yolo()
        self._load_dino()
        self._load_classifier()

        self.transform = make_transform(args.img_size)

    def _load_yolo(self):
        logging.info(f"Loading YOLOv8: {self.args.yolo_weights}")
        self.yolo = YOLO(self.args.yolo_weights)

    def _load_dino(self):
        logging.info(f"Loading DINOv3: {self.args.dino_weights}")

        self.dino = dinov3_vits16(pretrained=False)
        state_dict = torch.load(
            self.args.dino_weights,
            map_location=self.device,
            weights_only=True
        )
        self.dino.load_state_dict(state_dict)
        self.dino.to(self.device).eval()

    def _load_classifier(self):
        logging.info(f"Loading Classifier: {self.args.classifier_path}")

        data = joblib.load(self.args.classifier_path)
        self.classifier = data['model']
        self.classes = data['classes']

        logging.info(f"   Classes: {self.classes}")

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        img_tensor = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.dino(img_tensor)

            if isinstance(output, dict):
                features = output['x_norm_clstoken']
            elif output.dim() == 2:
                features = output
            else:
                features = output[:, 0, :]

        return features.cpu().numpy()[0]

    def draw_detection(self, frame: np.ndarray, box: tuple, label: str, conf: float, track_id: int):
        x1, y1, x2, y2 = box
        color = TEAM_COLORS.get(label, (128, 128, 128))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        display_label = label.replace('_', ' ').upper()
        label_text = f"#{track_id} {display_label} {conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 4, y1 - 5), font,
                   font_scale, (255, 255, 255), thickness)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]

        results = self.yolo.track(
            frame,
            persist=True,
            tracker=self.args.tracker,
            verbose=False
        )

        boxes_result = results[0].boxes
        if not boxes_result or not boxes_result.is_track or boxes_result.id is None:
            return frame

        boxes = boxes_result.xyxy.cpu().numpy()
        track_ids = boxes_result.id.int().cpu().numpy()

        for idx, box in enumerate(boxes):
            track_id = int(track_ids[idx])
            x1, y1, x2, y2 = map(int, box[:4])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            features = self.extract_features(crop_rgb)

            probs = self.classifier.predict_proba(features.reshape(1, -1))[0]
            pred_idx = np.argmax(probs)
            label = self.classes[pred_idx]
            conf = probs[pred_idx]

            self.draw_detection(frame, (x1, y1, x2, y2), label, conf, track_id)

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {self.args.video}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Input: {self.args.video}")
        logging.info(f"- Resolution: {width}x{height} @ {fps:.1f}fps")
        logging.info(f"- Frames: {total_frames}")
        logging.info(f"Output: {self.args.output}")
        logging.info(f"Tracker: {self.args.tracker}")

        ensure_parent_dir(self.args.output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.args.output, fourcc, fps, (width, height))

        with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.process_frame(frame)
                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()

        logging.info(f"Complete: Processed {total_frames} frames")
        logging.info(f"Output: {self.args.output}")


def main():
    args = parse_args()
    setup_logger(args.log)

    logging.info("MotoGP Team Detection Pipeline")
    logging.info("Detection + BoT-SORT Tracking + DINOv3 Team Classification")

    pipeline = MotoGPPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
