import os
import sys
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

# Add dinov3 repo to path for direct import (avoids heavy hubconf.py dependencies)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dinov3'))
from dinov3.hub.backbones import dinov3_vits16

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root folder of team images")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save the trained classifier model")
    parser.add_argument("--model_weights", type=str, default="models/DINO/dinov3_vits16_pretrain_lvd1689m.pth" ,help="Path to the pre-trained DINOv3 model weights")
    parser.add_argument("--repo", type=str, default="dinov3", help="local repo for DINOv3 model")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resize", type=int, default=224, help="Image input size (must be multiple of 14/16 depending on model)")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for Logistic Regression solver")

    return parser.parse_args()


def make_transform(resize_size=224):
    """Dinov3 Pre-processing transform"""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def main():
    args = parse_args()

    # setup
    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    print(f"📂 Data: {args.data_dir}")
    print(f"🦖 Repo: {args.repo}")
    print(f"⚖️  Weights: {args.model_weights}")

    # Load DINOv3 model
    print("\nLoading DINOv3 model...")
    model = dinov3_vits16(pretrained=False)
    state_dict = torch.load(args.model_weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
    model.to(device)
    model.eval()

    # Prepare Dataset
    print(f"\n Preparing dataset from {args.data_dir}...")
    transform = make_transform(args.resize)
    dataset = ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    print(f"Dataset contains {len(dataset)} images across {len(dataset.classes)} classes.")
    print(f"Classes: {dataset.classes}")

    # Feature Extraction
    feature_list = []
    label_list = []

    print("\nExtracting features using DINOv3...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Extracting Features"):
            imgs = imgs.to(device)
            output = model(imgs)

            # Handle different output formats
            if isinstance(output, dict):
                features = output['x_norm_clstoken']
            elif output.dim() == 2:
                features = output  # Already (batch_size, feature_dim)
            else:
                features = output[:, 0, :]  # (batch_size, num_tokens, dim) -> CLS token

            feature_list.append(features.cpu().numpy())
            label_list.append(labels.numpy())

    features = np.vstack(feature_list)
    labels = np.hstack(label_list)

    # Train and Evaluate
    print(f"\nTraining classifier on {features.shape[0]} samples...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=args.random_seed
    )
    clf = LogisticRegression(max_iter=args.max_iter, solver='lbfgs')
    clf.fit(X_train, y_train)

    # validation report
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nValidation Accuracy: {acc:.2f}")
    print("-" * 60)
    print(classification_report(y_test, preds, target_names=dataset.classes))

    # Save final model
    print("\nRetraining on full dataset and saving...")
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


if __name__ == "__main__":
    main()