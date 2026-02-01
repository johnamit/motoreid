import cv2
from ultralytics import YOLO
import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save cropped bike images.")
    parser.add_argument("--stride", type=int, default=10, help="Save a crop every N frames (to avoid duplicates)")
    parser.add_argument("--GP", type=str, required=True, help="Grand Prix identifier")
    return parser.parse_args()


def harvest(video_path, outdir, stride, GP):
    print(f"Loading V2 model to harvest bikes from {video_path}")
    model = YOLO("runs/detect/runs/detect/motogp_v2/weights/best.pt")

    cap = cv2.VideoCapture(video_path) # Open the video file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total number of frames in the video

    frame_id = 0
    saved_count = 0

    progress = tqdm(total=total_frames, desc="Processing video frames") # Progress bar
    
    while cap.isOpened():
        success, frame = cap.read() # Read a frame
        if not success:
            break # Break the loop if no frame is read

        # process every nth (stride) frame to avoid duplicates
        if frame_id % stride == 0:
            results = model.predict(frame, conf=0.60, verbose=False) # using 0.6 confidence threshold reflecting the detector.py threshold

            for i, box in enumerate(results[0].boxes): # Iterate over detected boxes
                # get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Dead Zone Filter (Ignore leaderboard)
                if (x1 + x2) / 2 < (frame.shape[1] * 0.20):
                    continue

                # Crop the bike
                # We add a tiny bit of padding (optional, but helps DINO with tight crops)
                h, w, _ = frame.shape
                x1 = max(0, x1 - 5)
                y1 = max(0, y1 - 5)
                x2 = min(w, x2 + 5)
                y2 = min(h, y2 + 5)

                bike_crop = frame[y1:y2, x1:x2]

                # Save the cropped bike image
                if bike_crop.size > 0:
                    crop_filename = f"{outdir}/{GP}_bike_{frame_id}_{i}.jpg"
                    cv2.imwrite(crop_filename, bike_crop)
                    saved_count += 1

        progress.update(1) # Update progress bar
        frame_id += 1
    
    progress.close()
    cap.release() # Release the video capture object
    print(f"Harvested {saved_count} bike crops from the video. Saved to {outdir}")


def main():
    args = parse_args() # Parse command-line arguments
    os.makedirs(args.outdir, exist_ok=True) # Create output directory if it doesn't exist
    harvest(args.video, args.outdir, args.stride, args.GP) # Call the harvest function

if __name__ == "__main__":
    main()