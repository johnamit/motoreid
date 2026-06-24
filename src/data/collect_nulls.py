"""Interactively collect null-sample frames from a race video."""
import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--outdir", type=str, default="data/null_samples", help="Output directory for null samples")
    return parser.parse_args()

def collect(video_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    print(f"Video: {video_path}")
    print(f"Press 's' to save a sample")
    print(f"Press 'q' to quit")

    saved_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        cv2.imshow("Null Sample Collector", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            filename = f"{outdir}/null_{saved_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            saved_count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    collect(args.video, args.outdir)
