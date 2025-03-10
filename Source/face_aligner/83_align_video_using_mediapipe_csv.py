import cv2
import numpy as np
import os
import math
import subprocess
import shutil
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import clear_directory  # Assuming this is your utility function

# Configurable output resolution constants
OUTPUT_WIDTH = 4096
OUTPUT_HEIGHT = 2160
MAX_WORKERS = 4  # Start with 4 for M1 Mac, adjustable (e.g., 2-8)

# Padding color for foreground (RGB)
PADDING_COLOR = (57, 255, 20)

# Debug flag to control image writing
DEBUG = False  # Set to True to enable debug image writing


# Load CSV data into a dictionary
def load_csv_data(csv_path="80_media_pipe_data/media_pipe_output_merged.csv"):
    """Loads eye center data from CSV into a dict: filename -> list of (face_num, left_eye, right_eye).
    Handles 'N/A' as None for eye coordinates."""
    csv_data = {}
    try:
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row["input_file_name"]
                face_num = int(row["face_number"])

                # Handle iris_left_value
                if row["iris_left_value"].strip('"') == "N/A":
                    left_eye = None
                else:
                    left_eye = tuple(
                        map(int, row["iris_left_value"].strip('"').split(","))
                    )

                # Handle iris_right_value
                if row["iris_right_value"].strip('"') == "N/A":
                    right_eye = None
                else:
                    right_eye = tuple(
                        map(int, row["iris_right_value"].strip('"').split(","))
                    )

                if filename not in csv_data:
                    csv_data[filename] = []
                csv_data[filename].append((face_num, left_eye, right_eye))
    except FileNotFoundError:
        print(f"Error: CSV file {csv_path} not found.")
        return {}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}
    return csv_data


# === Transformation Matrix Function ===
def compute_transformation_matrix(frame, left_eye, right_eye, target_eye_ratio=0.08):
    """Computes transformation matrices using provided eye center points."""
    h, w = frame.shape[:2]

    # Check if eye data is valid
    if left_eye is None or right_eye is None:
        return None, None

    # Calculate face center and eye distance
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    if eye_distance == 0:  # Avoid division by zero
        return None, None

    # Compute rotation angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Compute scale factor
    target_eye_distance = OUTPUT_WIDTH * target_eye_ratio
    scale_factor = target_eye_distance / eye_distance

    # Create transformation matrices
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor)
    M2 = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor * 2)

    # Translate to center of output frame
    tx = OUTPUT_WIDTH // 2 - eyes_center[0]
    ty = OUTPUT_HEIGHT // 2 - eyes_center[1]
    M[0, 2] += tx
    M[1, 2] += ty
    M2[0, 2] += tx
    M2[1, 2] += ty

    return M, M2


# === Video Processing Functions ===
def initialize_video_reader(video_path):
    """Initializes video capture and reads first frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {video_path}")
        return None, None, None
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        cap.release()
        return None, None, None
    return cap, fps, first_frame


def process_frames(cap, out, M, M2, debug_dir, video_name, face_suffix):
    """Processes all frames with transformation, using M2 for background and M for foreground."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_frame_transformed = None
    frames_written = 0
    frames_read = 0

    # Precompute mask from the first frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Failed to read first frame for mask computation")
        return 0, 0
    foreground = cv2.warpAffine(
        frame,
        M,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PADDING_COLOR,
    )
    mask = (foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255
    mask = cv2.merge([mask, mask, mask])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start processing all frames

    while True:
        ret, frame = cap.read()
        if not ret:
            if frames_read == 0:
                print(f"Error: Failed to read any frames from video")
            break
        frames_read += 1

        background = cv2.warpAffine(frame, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        background = cv2.GaussianBlur(background, (41, 41), 0)
        foreground = cv2.warpAffine(
            frame,
            M,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=PADDING_COLOR,
        )
        aligned_frame = np.where(mask == 0, foreground, background)

        out.write(aligned_frame)
        frames_written += 1
        if first_frame_transformed is None:
            first_frame_transformed = aligned_frame

    if DEBUG and first_frame_transformed is not None:
        cv2.imwrite(
            os.path.join(
                debug_dir, f"{video_name}{face_suffix}_first_frame_transformed.jpg"
            ),
            first_frame_transformed,
        )

    return frames_read, frames_written


def add_audio_to_video(temp_video_path, video_path, final_output_path):
    """Adds audio from original video to processed video using FFmpeg."""
    try:
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            temp_video_path,
            "-i",
            video_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            final_output_path,
            "-y",
        ]
        subprocess.run(
            ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding audio to {final_output_path}: {e}")
        return False


def just_save_first_frame(
    video_path,
    output_path,
    debug_dir,
    csv_data,
    face_num,
    left_eye,
    right_eye,
    face_suffix="",
):
    """Process first video frame and save as image using CSV eye data with foreground/background composition."""
    cap, fps, first_frame = initialize_video_reader(video_path)
    if cap is None:
        return False

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_image_path = os.path.join(
        os.path.dirname(output_path), f"{video_name}{face_suffix}.jpg"
    )

    M, M2 = compute_transformation_matrix(first_frame, left_eye, right_eye)
    if M is None or M2 is None:
        print(
            f"Error computing transformation for {video_path} face {face_num} (eye data missing or invalid), skipping."
        )
        cap.release()
        return False

    # Process only the first frame with foreground/background composition
    try:
        # Precompute mask from first frame
        foreground = cv2.warpAffine(
            first_frame,
            M,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=PADDING_COLOR,
        )
        mask = (foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255
        mask = cv2.merge([mask, mask, mask])

        # Apply transformations to first frame
        background = cv2.warpAffine(first_frame, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        background = cv2.GaussianBlur(background, (41, 41), 0)
        foreground = cv2.warpAffine(
            first_frame,
            M,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=PADDING_COLOR,
        )
        processed_frame = np.where(mask == 0, foreground, background)

        # Save the processed frame as an image
        success = cv2.imwrite(output_image_path, processed_frame)

        if success:
            print(f"Saved processed first frame: {output_image_path}")

            # Optional debug output
            if debug_dir and processed_frame is not None:
                debug_path = os.path.join(
                    debug_dir, f"{video_name}{face_suffix}_first_frame_transformed.jpg"
                )
                cv2.imwrite(debug_path, processed_frame)

            cap.release()
            return True
        else:
            print(f"Error: Failed to save image at {output_image_path}")
            cap.release()
            return False

    except Exception as e:
        print(f"Error processing frame for {output_image_path}: {str(e)}")
        cap.release()
        return False


def process_video(
    video_path,
    output_path,
    debug_dir,
    csv_data,
    face_num,
    left_eye,
    right_eye,
    face_suffix="",
):
    # just_save_first_frame(video_path, output_path, debug_dir, csv_data, face_num, left_eye, right_eye, face_suffix)
    # return True

    """Main video processing function using CSV eye data."""
    cap, fps, first_frame = initialize_video_reader(video_path)
    if cap is None:
        return False

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_video_path = os.path.join(
        os.path.dirname(output_path), f"temp_{video_name}{face_suffix}.mp4"
    )
    final_output_path = os.path.join(
        os.path.dirname(output_path), f"{video_name}{face_suffix}.mp4"
    )

    M, M2 = compute_transformation_matrix(first_frame, left_eye, right_eye)
    if M is None or M2 is None:
        print(
            f"Error computing transformation for {video_path} face {face_num} (eye data missing or invalid), skipping."
        )
        cap.release()
        return False

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not out.isOpened():
        print(f"Error: Could not open output video writer for {temp_video_path}")
        cap.release()
        return False

    frames_read, frames_written = process_frames(
        cap, out, M, M2, debug_dir, video_name, face_suffix
    )
    success = frames_read > 0 and frames_read == frames_written

    cap.release()
    out.release()

    if success:
        audio_success = add_audio_to_video(
            temp_video_path, video_path, final_output_path
        )
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if audio_success:
            print(
                f"Processed: {final_output_path} ({frames_written}/{frames_read} frames)"
            )
            return True
        else:
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            return False
    else:
        print(
            f"Error: Video processing failed for {final_output_path} "
            f"(read: {frames_read}, written: {frames_written}), removing incomplete file"
        )
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return False


def copy_to_failed_videos(video_path, failed_dir):
    """Copies the input video to the failed_videos directory."""
    try:
        failed_path = os.path.join(failed_dir, os.path.basename(video_path))
        shutil.copy2(video_path, failed_path)
        print(f"Copied failed video to: {failed_path}")
    except Exception as e:
        print(f"Error copying {video_path} to failed_videos: {e}")


def process_single_video(video_path, output_dir, debug_dir, failed_dir, csv_data):
    """Processes a single video using CSV data and returns result."""
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        copy_to_failed_videos(video_path, failed_dir)
        return video_path, {"detected": 0, "processed": 0}
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error reading {video_path}")
        copy_to_failed_videos(video_path, failed_dir)
        return video_path, {"detected": 0, "processed": 0}

    if DEBUG:
        cv2.imwrite(
            os.path.join(debug_dir, f"{video_name}_first_frame_original.jpg"),
            first_frame,
        )

    # Check if filename exists in CSV
    if video_name not in csv_data:
        print(f"No CSV entry for {video_path}, skipping.")
        copy_to_failed_videos(video_path, failed_dir)
        return video_path, {"detected": 0, "processed": 0}

    # Process each face from CSV data
    faces_to_process = csv_data[video_name]
    processed_count = 0

    for face_num, left_eye, right_eye in faces_to_process:
        face_suffix = f"_face{face_num:02d}"
        success = process_video(
            video_path,
            output_path,
            debug_dir,
            csv_data,
            face_num,
            left_eye,
            right_eye,
            face_suffix,
        )
        if success:
            processed_count += 1

    result = {"detected": len(faces_to_process), "processed": processed_count}
    if processed_count == 0:
        copy_to_failed_videos(video_path, failed_dir)

    return video_path, result


# Parallelized process_videos
def process_videos(
    video_dir,
    output_dir,
    debug_dir,
    failed_dir="failed_videos",
    csv_path="80_media_pipe_data/media_pipe_output_merged.csv",
):
    """Processes all videos in the folder with multithreading using CSV data."""
    clear_directory(output_dir)
    clear_directory(debug_dir)
    clear_directory(failed_dir)

    # Load CSV data
    csv_data = load_csv_data(csv_path)
    if not csv_data:
        print("No valid CSV data loaded. Exiting.")
        return

    processing_results = {}
    video_tasks = []

    for video in sorted(os.listdir(video_dir)):
        if not video.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        video_path = os.path.join(video_dir, video)
        video_tasks.append(video_path)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {
            executor.submit(
                process_single_video,
                video_path,
                output_dir,
                debug_dir,
                failed_dir,
                csv_data,
            ): video_path
            for video_path in video_tasks
        }
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                video_path, result = future.result()
                processing_results[video_path] = result
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                processing_results[video_path] = {"detected": 0, "processed": 0}
                copy_to_failed_videos(video_path, failed_dir)

    print("\n=== Processing Summary ===")
    for video_path, stats in processing_results.items():
        status = "✓" if stats["processed"] > 0 else "✗"
        print(
            f"{status} {os.path.basename(video_path)} - Faces detected: {stats['detected']}, Processed: {stats['processed']}"
        )


# Run processing
process_videos(
    "M:\\Photos\\Project\\Processed_Hadi\\01_FaceVideos_Trimmed",
    "08_output_videos",
    "99_debug_frames",
    "92_failed_videos",
)
