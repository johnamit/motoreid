import os
from dotenv import load_dotenv
from ultralytics import YOLO
import torch
from roboflow import Roboflow

load_dotenv()

def download_dataset():
    """Download my annotated dataset from Roboflow"""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in environment variables.")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("amitjohnworkspace").project("motogp-team-detection")
    version = project.version(5)
    dataset = version.download("yolov8") 
    return dataset


def train_model():
    # setup device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # download dataset
    print("Downloading dataset...")
    dataset = download_dataset()

    # load the base model
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8m.pt")  # using the medium model

    # train the model
    print("Starting training...")
    model.train(
        data = f"{dataset.location}/data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = -1, # auto batch size
        project = "runs/detect",
        name = "motogp_v2",
        exist_ok = True,
        plots = True,
        device = device
    )

    print("Training complete. Model saved to runs/detect/motogp_v2")


if __name__ == "__main__":
    train_model()

