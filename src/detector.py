import cv2
from ultralytics import YOLO
import sys, os
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, default="output/annotated_race.mp4", help="Path to save annotated output video.")
    parser.add_argument("--log", type=str, default="logs/detection_log.txt", help="Path to save log file.")
    return parser.parse_args()

def setup_logger(log_path):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def detect(video_path, output_path):
    """Detect bikes in the video and save annotated output."""
    logging.info(f"Initializing Custom MotoGP model for {video_path}...")
    try:
        model = YOLO("runs/detect/runs/detect/motogp_v2/weights/best.pt")
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        logging.error(f"Failed to setup: {e}")
        return

    if not cap.isOpened():
        logging.error(f"Error: Could not open video {video_path}.")
        return

    # Get video properties to match the input
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec (mp4v is standard for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Saving video to {output_path} ({width}x{height} @ {fps}fps)...")
    logging.info(f"Saving video to {output_path} ({width}x{height} @ {fps}fps)...")

    frame_id = 0
    logging.info("Starting detection loop...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logging.info("End of video stream.")
            break

        # Run prediction
        # (We use a lower conf here so we can filter manually later)
        results = model.predict(frame, conf=0.55, verbose=False)

        valid_detections = 0

        # Drawing logic (manual only)
        # We draw directly on 'frame' so the saved video has the boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            
            # 1. Dead Zone Filter (Left 20%)
            img_width = frame.shape[1]
            box_center_x = (x1 + x2) / 2
            if box_center_x < (img_width * 0.20): 
                continue 

            # 2. Confidence Filter (High Bar)
            if confidence < 0.60:
                continue
            
            # If we get here, it's a valid bike!
            valid_detections += 1
            
            # Draw Box (Green)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw Label (Optional: "Bike 0.95")
            label = f"Bike {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the file
        out.write(frame) 

        # Display (Optional - you can comment this out if you just want to process fast)
        # cv2.imshow("MotoGP Detection", frame)
        
        if frame_id % 30 == 0:
            logging.info(f"Frame {frame_id}: {valid_detections} valid bikes.")

        frame_id += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # Cleanup
    cap.release() # release the video capture object
    out.release() # release the video writer
    cv2.destroyAllWindows()
    logging.info("Processing complete.")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    setup_logger(args.log)
    detect(args.video, args.output)


if __name__ == "__main__":
    main()