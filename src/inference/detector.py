"""Run YOLO-only bike detection on a race video and save annotated output."""
import cv2
from ultralytics import YOLO
import os
import sys
import logging
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--yolo_weights", type=str, default="models/YOLO/motogp_yolov8m/weights/best.pt", help="YOLO model weights")
    parser.add_argument("--output", type=str, default="results/yolo_detection/race_annotated.mp4", help="Output video path")
    parser.add_argument("--log", type=str, default="logs/yolo_detection/detection_log.txt", help="Log file path")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def setup_logger(log_path):
    ensure_parent_dir(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def detect(video_path, yolo_weights, output_path):
    logging.info(f"Initializing MotoGP model for {video_path}...")
    try:
        model = YOLO(yolo_weights)
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        logging.error(f"Failed to setup: {e}")
        return

    if not cap.isOpened():
        logging.error(f"Error: Could not open video {video_path}.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ensure_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logging.info(f"Saving video to {output_path} ({width}x{height} @ {fps}fps)...")
    logging.info("Starting detection loop...")

    with tqdm(total=total_frames, desc="Detecting", unit="frame") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(frame, verbose=False)

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("Processing complete.")


def main():
    args = parse_args()
    setup_logger(args.log)
    detect(args.video, args.yolo_weights, args.output)


if __name__ == "__main__":
    main()
