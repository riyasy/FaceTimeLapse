import shutil
import cv2
import numpy as np
import os
import math
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import clear_directory  # Assuming this is your utility function

# Configurable constants
OUTPUT_WIDTH = 4096
OUTPUT_HEIGHT = 2160
OUTPUT_FPS = 24  # Frames per second for output video
FRAME_DUPLICATION_COUNT = (
    24  # Number of times to duplicate each image (1 second at 24 FPS)
)
MAX_WORKERS = min(8, max(2, os.cpu_count() // 2))  # Adaptive workers based on CPU cores

# Padding color for foreground (RGB)
PADDING_COLOR = (57, 255, 20)

# Debug flag to control image writing
DEBUG = False  # Set to True to enable debug image writing


# Load CSV data into a dictionary
def load_csv_data(csv_path="80_media_pipe_data/media_pipe_output_merged.csv"):
    """Loads eye center data from CSV into a dict: filename -> list of (face_num, left_eye, right_eye)."""
    csv_data = {}
    try:
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row["input_file_name"]
                face_num = int(row["face_number"])

                if row["iris_left_value"].strip('"') == "N/A":
                    left_eye = None
                else:
                    left_eye = tuple(
                        map(int, row["iris_left_value"].strip('"').split(","))
                    )

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


# Compute transformation matrix
def compute_transformation_matrix(image, left_eye, right_eye, target_eye_ratio=0.08):
    """Computes transformation matrices using provided eye center points."""
    h, w = image.shape[:2]

    if left_eye is None or right_eye is None:
        return None, None

    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    if eye_distance == 0:
        return None, None

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    target_eye_distance = OUTPUT_WIDTH * target_eye_ratio
    scale_factor = target_eye_distance / eye_distance

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor)
    M2 = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor * 2)

    tx = OUTPUT_WIDTH // 2 - eyes_center[0]
    ty = OUTPUT_HEIGHT // 2 - eyes_center[1]
    M[0, 2] += tx
    M[1, 2] += ty
    M2[0, 2] += tx
    M2[1, 2] += ty

    return M, M2


# Process a single image and write to video
def process_image_to_video(
    image_path,
    output_path,
    debug_dir,
    csv_data,
    face_num,
    left_eye,
    right_eye,
    face_suffix="",
):
    """Processes an image and creates a video by duplicating the processed frame."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    final_output_path = os.path.join(
        os.path.dirname(output_path), f"{image_name}{face_suffix}.mp4"
    )

    M, M2 = compute_transformation_matrix(image, left_eye, right_eye)
    if M is None or M2 is None:
        print(
            f"Error computing transformation for {image_path} face {face_num}, skipping."
        )
        return False

    # Process the image with foreground/background composition
    foreground = cv2.warpAffine(
        image,
        M,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PADDING_COLOR,
    )
    mask = (foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255
    mask = cv2.merge([mask, mask, mask])

    background = cv2.warpAffine(image, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    background = cv2.GaussianBlur(background, (41, 41), 0)
    processed_frame = np.where(mask == 0, foreground, background)

    # Write to video
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(
        final_output_path, fourcc, OUTPUT_FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )
    if not out.isOpened():
        print(f"Error: Could not open output video writer for {final_output_path}")
        return False

    # Duplicate the processed frame
    for _ in range(FRAME_DUPLICATION_COUNT):
        out.write(processed_frame)

    out.release()

    # Debug output
    if DEBUG and debug_dir:
        debug_path = os.path.join(
            debug_dir, f"{image_name}{face_suffix}_processed_frame.jpg"
        )
        cv2.imwrite(debug_path, processed_frame)

    print(f"Processed: {final_output_path}")
    return True


# Process a single image file
def process_single_image(image_path, output_dir, debug_dir, failed_dir, csv_data):
    """Processes a single image using CSV data and returns result."""
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        copy_to_failed_videos(image_path, failed_dir)
        return image_path, {"detected": 0, "processed": 0}

    if DEBUG:
        cv2.imwrite(
            os.path.join(debug_dir, f"{image_name}_original.jpg"),
            image,
        )

    if image_name not in csv_data:
        print(f"No CSV entry for {image_path}, skipping.")
        copy_to_failed_videos(image_path, failed_dir)
        return image_path, {"detected": 0, "processed": 0}

    faces_to_process = csv_data[image_name]
    processed_count = 0

    for face_num, left_eye, right_eye in faces_to_process:
        face_suffix = f"_face{face_num:02d}"
        success = process_image_to_video(
            image_path,
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
        copy_to_failed_videos(image_path, failed_dir)

    return image_path, result


# Copy failed files (renamed to avoid confusion with original script)
def copy_to_failed_videos(image_path, failed_dir):
    """Copies the input image to the failed_videos directory."""
    try:
        failed_path = os.path.join(failed_dir, os.path.basename(image_path))
        shutil.copy2(image_path, failed_path)
        print(f"Copied failed image to: {failed_path}")
    except Exception as e:
        print(f"Error copying {image_path} to failed_videos: {e}")


# Parallelized process_images
def process_images(
    image_dir,
    output_dir,
    debug_dir,
    failed_dir="failed_videos",
    csv_path="80_media_pipe_data/media_pipe_output_merged.csv",
):
    """Processes all images in the folder with multithreading using CSV data."""
    clear_directory(output_dir)
    clear_directory(debug_dir)
    clear_directory(failed_dir)

    csv_data = load_csv_data(csv_path)
    if not csv_data:
        print("No valid CSV data loaded. Exiting.")
        return

    processing_results = {}
    image_tasks = []

    for image in sorted(os.listdir(image_dir)):
        if not image.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image)
        image_tasks.append(image_path)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_image = {
            executor.submit(
                process_single_image,
                image_path,
                output_dir,
                debug_dir,
                failed_dir,
                csv_data,
            ): image_path
            for image_path in image_tasks
        }
        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                image_path, result = future.result()
                processing_results[image_path] = result
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                processing_results[image_path] = {"detected": 0, "processed": 0}
                copy_to_failed_videos(image_path, failed_dir)

    print("\n=== Processing Summary ===")
    for image_path, stats in processing_results.items():
        status = "✓" if stats["processed"] > 0 else "✗"
        print(
            f"{status} {os.path.basename(image_path)} - Faces detected: {stats['detected']}, Processed: {stats['processed']}"
        )


# Run processing
process_images(
    "M:\\Photos\\Project\\Processed_Hadi\\01_FaceVideos_Trimmed",
    "08_output_videos",
    "99_debug_frames",
    "92_failed_videos",
)
