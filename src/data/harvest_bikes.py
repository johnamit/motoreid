"""Harvest cropped bike detections from a race video for dataset building."""
import cv2
from ultralytics import YOLO
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save cropped bike images")
    parser.add_argument("--stride", type=int, default=10, help="Process every Nth frame to avoid duplicates")
    parser.add_argument("--gp", type=str, required=True, help="Grand Prix identifier")
    parser.add_argument("--yolo_weights", type=str, default="models/YOLO/motogp_yolov8m/weights/best.pt", help="YOLO detector weights")
    parser.add_argument("--conf", type=float, default=0.6, help="Detection confidence threshold")
    return parser.parse_args()


def harvest(video_path, outdir, stride, gp, yolo_weights, conf):
    print(f"Loading model from {yolo_weights} to harvest bikes from {video_path}")
    model = YOLO(yolo_weights)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    saved_count = 0

    progress = tqdm(total=total_frames, desc="Processing video frames")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_id % stride == 0:
            results = model.predict(frame, conf=conf, verbose=False)

            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                h, w, _ = frame.shape
                x1 = max(0, x1 - 5)
                y1 = max(0, y1 - 5)
                x2 = min(w, x2 + 5)
                y2 = min(h, y2 + 5)

                bike_crop = frame[y1:y2, x1:x2]

                if bike_crop.size > 0:
                    crop_filename = f"{outdir}/{gp}_bike_{frame_id}_{i}.jpg"
                    cv2.imwrite(crop_filename, bike_crop)
                    saved_count += 1

        progress.update(1)
        frame_id += 1

    progress.close()
    cap.release()
    print(f"Harvested {saved_count} bike crops from the video. Saved to {outdir}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    harvest(args.video, args.outdir, args.stride, args.gp, args.yolo_weights, args.conf)


if __name__ == "__main__":
    main()
