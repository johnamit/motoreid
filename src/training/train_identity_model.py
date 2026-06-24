"""Train a DINOv3-based team classifier using Logistic Regression."""
import os
import sys
import logging
import torch
import numpy as np
import joblib
import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torchvision.transforms import v2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dinov3'))
from dinov3.hub.backbones import dinov3_vits16

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder of team images")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the trained classifier")
    parser.add_argument("--dino_weights", type=str, default="models/DINO/dinov3_vits16_pretrain_lvd1689m.pth", help="Path to DINOv3 pretrained weights")
    parser.add_argument("--log", type=str, default="logs/train_identity_model/training.log", help="Log file path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for Logistic Regression")
    return parser.parse_args()


def make_transform(resize_size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def main():
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Data: {args.data_dir}")
    logging.info(f"Device: {device} | Model dir: {args.model_dir} | Weights: {args.dino_weights}")

    logging.info("Loading DINOv3 model...")
    model = dinov3_vits16(pretrained=False)
    state_dict = torch.load(args.dino_weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logging.info(f"Preparing dataset from {args.data_dir}...")
    transform = make_transform(args.img_size)
    dataset = ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    logging.info(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes: {dataset.classes}")

    feature_list = []
    label_list = []

    logging.info("Extracting features using DINOv3...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Extracting Features", disable=None):
            imgs = imgs.to(device)
            output = model(imgs)

            if isinstance(output, dict):
                features = output['x_norm_clstoken']
            elif output.dim() == 2:
                features = output
            else:
                features = output[:, 0, :]

            feature_list.append(features.cpu().numpy())
            label_list.append(labels.numpy())

    features = np.vstack(feature_list)
    labels = np.hstack(label_list)

    logging.info(f"Training classifier on {features.shape[0]} samples...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=args.random_seed
    )
    clf = LogisticRegression(max_iter=args.max_iter, solver='lbfgs')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc:.2f}")
    logging.info(f"Validation Accuracy: {acc:.2f}")
    logging.info("\n" + classification_report(y_test, preds, target_names=dataset.classes))

    logging.info("Retraining on full dataset and saving...")
    final_clf = LogisticRegression(max_iter=args.max_iter, solver='lbfgs')
    final_clf.fit(features, labels)

    save_path = os.path.join(args.model_dir, "dinov3_identity_model.pkl")
    joblib.dump({
        'model': final_clf, 
        'classes': dataset.classes,
        'embedding_dim': features.shape[1],
        'model_name': 'dinov3_vits16'
    }, save_path)

    print(f"Model saved to: {save_path}")
    logging.info(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
