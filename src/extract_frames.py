import cv2
import os
import argparse
import logging
from pathlib import Path

# supported video file extensions
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%H:%M:%S'
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video file or directory containing videos.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--interval", "-n", type=int, default=150, help="Extract every nth frame.")

    return parser.parse_args()


def extract_frames(video_path, output_dir, frame_interval):
    """Extract frames from a video at specified intervals"""
    video_name = Path(video_path).stem # get video name without extension
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return 0

    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Processing video: {video_path} | Total frames: {total_frames}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # check if its time to save the frame
        if frame_count % frame_interval == 0:
            # naming format: videoname_frameXXXX.jpg
            filename = f"{video_name}_frame{frame_count:04d}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    # release video capture object
    cap.release()
    logging.info(f"Extracted {saved_count} frames from {video_path}")
    return saved_count


def main():
    # setup logging, paths and parse arguments
    setup_logging()
    args = parse_args()
    input_path = Path(args.input) # input video file or directory
    output_path = Path(args.output) # output directory

    # validation
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return
    
    videos_to_process = [] # list to hold video files to process

    if input_path.is_dir(): # if the input path is a directory
        logging.info(f"Scanning directory for video files: {input_path}")
        for file in input_path.iterdir():
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                videos_to_process.append(file)
    elif input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        videos_to_process.append(input_path)
    else:
        logging.error("No valid video files found.")
        return

    # extraction loop
    total_saved = 0
    logging.info(f"Found {len(videos_to_process)} video(s) to process. Starting extraction...")

    for video in videos_to_process:
        count = extract_frames(str(video), str(output_path), args.interval)
        total_saved += count

    logging.info(f"Frame extraction completed. Total frames extracted: {total_saved}")

if __name__ == "__main__":
    main()


