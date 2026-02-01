import cv2
import os
import argparse

# Create the folder for your null images
OUTPUT_FOLDER = "data/null_samples"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    return parser.parse_args()

def collect(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"👀 Watching {video_path}...")
    print(f"👉 Press 's' to save a Null Sample.")
    print(f"👉 Press 'q' to quit.")

    saved_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Show the video (Raw, no boxes)
        cv2.imshow("Null Sample Collector", frame)
        
        key = cv2.waitKey(10) & 0xFF # Wait 10ms (plays slightly faster)

        if key == ord('s'):
            # Save the frame
            filename = f"{OUTPUT_FOLDER}/null_{saved_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
            saved_count += 1
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    collect(args.video)

# spain done
# italy done
