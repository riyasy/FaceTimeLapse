import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurable constants
MAX_WORKERS = 4  # Number of threads to use

def save_first_frame(video_path, output_dir):
    """Saves the first frame of a video as a JPG with the same filename."""
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return video_path, False

    # Read first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        return video_path, False

    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Save frame as JPG
    output_path = os.path.join(output_dir, f"{video_name}.jpg")
    cv2.imwrite(output_path, frame)
    
    return video_path, True

def process_videos(input_dir="01_input_videos_vfr", output_dir="99_debug_frames"):
    """Processes all videos in the input directory to save their first frames."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    video_tasks = []
    for video in sorted(os.listdir(input_dir)):
        if not video.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        video_path = os.path.join(input_dir, video)
        video_tasks.append(video_path)

    processing_results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {
            executor.submit(save_first_frame, video_path, output_dir): video_path
            for video_path in video_tasks
        }

        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                video_path, success = future.result()
                processing_results[video_path] = success
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                processing_results[video_path] = False

    # Print summary
    print("\n=== Processing Summary ===")
    for video_path, success in processing_results.items():
        status = "✓" if success else "✗"
        print(f"{status} {os.path.basename(video_path)} - First frame saved: {success}")

# Run processing
process_videos("01_input_videos_vfr", "99_debug_frames")