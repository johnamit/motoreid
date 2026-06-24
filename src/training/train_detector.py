"""Train a YOLOv8 model using a dataset downloaded from Roboflow."""
import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from ultralytics import YOLO
import torch
from roboflow import Roboflow

load_dotenv()

def resolve_model_path(model: str) -> str:
    """Resolve bare model names (yolov8n.pt) to local models/YOLO/ path."""
    if os.sep in model or os.path.exists(model):
        return model
    local_path = os.path.join("models", "YOLO", model)
    if os.path.exists(local_path):
        return local_path
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="yolov8m.pt", help="YOLOv8 base model (yolov8n.pt, yolov8m.pt, yolov8l.pt)")
    parser.add_argument("--run_name", type=str, default="motogp_yolov8m", help="Training run name (saved under models/YOLO/<run_name>)")
    parser.add_argument("--log", type=str, default="logs/train_detector/training.log", help="Log file path")
    return parser.parse_args()


def download_dataset():
    """Download annotated dataset from Roboflow."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in environment variables.")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("amitjohnworkspace").project("motogp-team-detection")
    version = project.version(5)
    dataset = version.download("yolov8")
    return dataset


def train_model(args):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    logging.info(f"Device: {device}")

    print("Downloading dataset from Roboflow...")
    logging.info("Downloading dataset from Roboflow...")
    dataset = download_dataset()
    logging.info(f"Dataset downloaded to {dataset.location}")

    model_path = resolve_model_path(args.base_model)
    print(f"Loading YOLOv8 model from {model_path}...")
    logging.info(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)

    run_name = args.run_name
    print(f"Starting training (run: {run_name})...")
    logging.info(f"Starting training (run: {run_name})...")
    save_dir = os.path.abspath(os.path.join("models", "YOLO", run_name))
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        project=os.path.abspath("models/YOLO"),
        name=run_name,
        exist_ok=True,
        plots=True,
        device=device
    )

    msg = f"Training complete. Model saved to {save_dir}"
    print(msg)
    logging.info(msg)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.WARNING)
    train_model(args)

