import shutil
import cv2
import numpy as np
import os
import math
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import clear_directory, get_eye_to_screen_ratio, get_filtered_videos

# --- CONFIGURATION ---
INPUT_DIR    = "00_input_images"
OUTPUT_DIR   = "03_output_videos"
DEBUG_DIR    = "99_debug_frames"
FAILED_DIR   = "99_failed_images"
CSV_PATH     = "02_annotated_csv/annotated_eye_coordinates.csv"

OUTPUT_WIDTH           = 4096
OUTPUT_HEIGHT          = 2160
OUTPUT_FPS             = 24   # Frames per second for the output video
FRAME_DUPLICATION_COUNT = 24  # Each image becomes this many identical frames (1 sec at 24 FPS)
MAX_WORKERS            = min(8, max(2, os.cpu_count() // 2))  # Adaptive thread count
PADDING_COLOR          = (57, 255, 20)  # Neon green used as chroma-key fill for empty border areas
DEBUG                  = False  # Set True to write intermediate debug images
# ---------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_csv_data(csv_path):
    """
    Reads the eye-centre CSV and returns a dict:
      { filename_stem: [(face_num, left_eye, right_eye), ...] }
    Coordinates are stored as (x, y) int tuples; 'N/A' entries become None.
    """
    csv_data = {}
    try:
        with open(csv_path, newline="") as csvfile:
            for row in csv.DictReader(csvfile):
                # Strip extension so 'clip.jpg' and 'clip.mp4' both map to 'clip'
                filename = os.path.splitext(row["input_file_name"])[0]
                face_num = int(row["face_number"])

                # Parse 'x,y' strings into tuples; treat 'N/A' as missing
                def parse_eye(val):
                    v = val.strip('"')
                    return tuple(map(int, v.split(","))) if v != "N/A" else None

                left_eye  = parse_eye(row["iris_left_value"])
                right_eye = parse_eye(row["iris_right_value"])

                csv_data.setdefault(filename, []).append((face_num, left_eye, right_eye))
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return csv_data


def compute_transformation_matrix(image, left_eye, right_eye, target_eye_ratio):
    """
    Builds two affine transformation matrices from eye positions:
      M  — foreground transform: rotates + scales so eyes are horizontal & centred.
      M2 — background transform: same rotation but 2× zoom for a blurred backdrop.
    Returns (None, None) if eye data is missing or invalid.
    """
    if left_eye is None or right_eye is None:
        return None, None

    eyes_center  = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    if eye_distance == 0:
        return None, None  # Degenerate case — both eyes at the same pixel

    # Rotation angle to level the eyes horizontally
    angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Scale so the inter-eye distance matches a fixed fraction of the output width
    scale_factor = (OUTPUT_WIDTH * target_eye_ratio) / eye_distance

    # Build rotation+scale matrices centred on the eyes midpoint
    M  = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor)
    M2 = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor * 2)

    # Translate so the eyes midpoint lands at the centre of the output frame
    tx = OUTPUT_WIDTH  // 2 - eyes_center[0]
    ty = OUTPUT_HEIGHT // 2 - eyes_center[1]
    M[0, 2]  += tx;  M[1, 2]  += ty
    M2[0, 2] += tx;  M2[1, 2] += ty

    return M, M2


def compose_frame(image, M, M2):
    """
    Applies foreground (M) and background (M2) transforms and composites them.
    Empty border pixels from the foreground warp are filled with the
    blurred, zoomed-out background instead of the PADDING_COLOR.
    """
    # Foreground: tightly cropped, aligned face; empty areas filled with neon green
    foreground = cv2.warpAffine(
        image, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        borderMode=cv2.BORDER_CONSTANT, borderValue=PADDING_COLOR,
    )
    # Mask marks every pixel that is still the neon-green padding colour
    mask = cv2.merge([(foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255] * 3)

    # Background: wider zoom + Gaussian blur gives a soft, defocused fill
    background = cv2.GaussianBlur(image, (41, 41), 0)
    background = cv2.warpAffine(background, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # Where mask == 0 keep foreground; where mask > 0 use background
    return np.where(mask == 0, foreground, background)


def process_image_to_video(image_path, output_dir, debug_dir, face_num, left_eye, right_eye, face_suffix):
    """
    Aligns a single image using the provided eye coordinates and writes a short
    video clip by repeating the aligned frame FRAME_DUPLICATION_COUNT times.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False

    image_name      = os.path.splitext(os.path.basename(image_path))[0]
    final_out_path  = os.path.join(output_dir, f"{image_name}{face_suffix}.mp4")

    M, M2 = compute_transformation_matrix(image, left_eye, right_eye, get_eye_to_screen_ratio(image_name))
    if M is None:
        print(f"Error: Could not compute transform for {image_path} face {face_num}, skipping.")
        return False

    processed_frame = compose_frame(image, M, M2)

    # Write the same aligned frame repeatedly to produce a still-image clip
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out    = cv2.VideoWriter(final_out_path, fourcc, OUTPUT_FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {final_out_path}")
        return False

    for _ in range(FRAME_DUPLICATION_COUNT):
        out.write(processed_frame)
    out.release()

    if DEBUG:
        cv2.imwrite(os.path.join(debug_dir, f"{image_name}{face_suffix}_processed.jpg"), processed_frame)

    print(f"Processed: {final_out_path}")
    return True


def copy_to_failed(src_path, failed_dir):
    """Copies a failed source file to the failed directory for later inspection."""
    try:
        shutil.copy2(src_path, os.path.join(failed_dir, os.path.basename(src_path)))
        print(f"Copied to failed: {os.path.basename(src_path)}")
    except Exception as e:
        print(f"Error copying {src_path} to failed dir: {e}")


def process_single_image(image_path, output_dir, debug_dir, failed_dir, csv_data):
    """
    Processes all faces listed in the CSV for a given image.
    Returns (image_path, {detected, processed}) for the summary report.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        copy_to_failed(image_path, failed_dir)
        return image_path, {"detected": 0, "processed": 0}

    if DEBUG:
        cv2.imwrite(os.path.join(debug_dir, f"{image_name}_original.jpg"), image)

    if image_name not in csv_data:
        print(f"No CSV entry for {image_path}, skipping.")
        copy_to_failed(image_path, failed_dir)
        return image_path, {"detected": 0, "processed": 0}

    faces = csv_data[image_name]
    processed_count = 0

    for face_num, left_eye, right_eye in faces:
        face_suffix = f"_face{face_num:02d}"
        if process_image_to_video(image_path, output_dir, debug_dir, face_num, left_eye, right_eye, face_suffix):
            processed_count += 1

    if processed_count == 0:
        copy_to_failed(image_path, failed_dir)

    return image_path, {"detected": len(faces), "processed": processed_count}


def main():
    # clear_directory(OUTPUT_DIR)  # Uncomment to wipe the output folder before each run
    clear_directory(DEBUG_DIR)
    clear_directory(FAILED_DIR)

    csv_data = load_csv_data(CSV_PATH)
    if not csv_data:
        print("No valid CSV data loaded. Exiting.")
        return

    # Collect all valid image files, filtered by the utility helper
    image_tasks = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(get_filtered_videos(INPUT_DIR))
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # Process images in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_single_image, path, OUTPUT_DIR, DEBUG_DIR, FAILED_DIR, csv_data): path
            for path in image_tasks
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                path, result = future.result()
                results[path] = result
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results[path] = {"detected": 0, "processed": 0}
                copy_to_failed(path, FAILED_DIR)

    print("\n=== Processing Summary ===")
    for path, stats in results.items():
        status = "✓" if stats["processed"] > 0 else "✗"
        print(f"{status} {os.path.basename(path)} — Faces detected: {stats['detected']}, Processed: {stats['processed']}")


if __name__ == "__main__":
    main()
