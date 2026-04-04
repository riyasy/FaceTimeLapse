import cv2
import numpy as np
import os
import math
import subprocess
import shutil
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import clear_directory, get_eye_to_screen_ratio, get_filtered_videos

# --- CONFIGURATION ---
INPUT_DIR    = "00_input_videos"
OUTPUT_DIR   = "03_output_videos"
DEBUG_DIR    = "99_debug_frames"
FAILED_DIR   = "99_failed_videos"
CSV_PATH     = "02_annotated_csv/annotated_eye_coordinates.csv"

OUTPUT_WIDTH  = 4096
OUTPUT_HEIGHT = 2160
MAX_WORKERS   = 4        # Parallel threads; tune for your machine (2–8 typical)
PADDING_COLOR = (57, 255, 20)  # Neon green used as chroma-key fill for empty border areas
DEBUG         = False    # Set True to write first-frame debug images
# ---------------------

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")


# --- HDR HANDLING FUNCTIONS ---
def is_hdr(video_path):
    """Uses ffprobe to detect if a video uses an HDR color transfer function (HLG or PQ)."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=color_transfer",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        color_transfer = result.stdout.strip().lower()
        # smpte2084 = PQ (HDR10/Dolby Vision), arib-std-b67 = HLG (iPhone HDR)
        return color_transfer in ["smpte2084", "arib-std-b67"]
    except Exception as e:
        print(f"Warning: Failed to check HDR status for {video_path}: {e}")
        return False

def convert_hdr_to_sdr(video_path, output_dir):
    """Converts HDR video to SDR using FFmpeg tone mapping to fix washed-out colors in OpenCV."""
    video_name = os.path.basename(video_path)
    temp_sdr_path = os.path.join(output_dir, f"temp_sdr_{video_name}")
    
    # FFmpeg command for tone-mapping HDR to SDR using zscale (high quality)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        temp_sdr_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return temp_sdr_path
    except subprocess.CalledProcessError:
        # Fallback if the user's FFmpeg is not compiled with zscale (zimg)
        print(f"Standard tonemapping failed, trying fallback for {video_name}...")
        fallback_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "colorspace=all=bt709:illuminate=d65:format=yuv420p",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            temp_sdr_path
        ]
        try:
            subprocess.run(fallback_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return temp_sdr_path
        except subprocess.CalledProcessError as e:
            print(f"Error: FFmpeg failed to convert HDR to SDR for {video_path}: {e}")
            return None
# ------------------------------


def load_csv_data(csv_path):
    """
    Reads the eye-centre CSV and returns a dict:
      { filename_stem: [(face_num, left_eye, right_eye), ...] }
    Coordinates are stored as (x, y) int tuples; 'N/A' entries become None.

    The extension is stripped from input_file_name so that a CSV annotated
    from JPEG first-frames can still match MP4/MOV files in the input folder.
    """
    csv_data = {}
    try:
        with open(csv_path, newline="") as csvfile:
            for row in csv.DictReader(csvfile):
                # Strip extension so 'clip.jpg' and 'clip.mp4' both map to 'clip'
                filename = os.path.splitext(row["input_file_name"])[0]
                face_num = int(row["face_number"])

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


def compute_transformation_matrix(frame, left_eye, right_eye, target_eye_ratio):
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


def initialize_video_reader(video_path):
    """Opens a video file and reads the first frame. Returns (cap, fps, first_frame) or (None, None, None) on error."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        cap.release()
        return None, None, None

    return cap, fps, first_frame


def process_frames(cap, out, M, M2, debug_dir, video_name, face_suffix):
    """
    Iterates every frame of the video, applies the affine transform, and writes to output.

    For each frame:
      - Foreground (M): aligned, tightly cropped face; padding areas filled with PADDING_COLOR.
      - Background (M2): 2× zoomed, Gaussian-blurred frame used to fill padding areas.
    If the foreground completely fills the output (no padding), the background step is skipped.
    """
    # Precompute the padding mask from the very first frame (constant throughout the video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read first frame for mask computation")
        return 0, 0

    foreground = cv2.warpAffine(
        frame, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        borderMode=cv2.BORDER_CONSTANT, borderValue=PADDING_COLOR,
    )
    mask = cv2.merge([(foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255] * 3)
    foreground_fills_frame = not np.any(mask)  # True → foreground covers every pixel; skip background

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to process all frames from the start
    frames_read = frames_written = 0
    first_transformed = None

    while True:
        ret, frame = cap.read()
        if not ret:
            if frames_read == 0:
                print("Error: Failed to read any frames from video")
            break
        frames_read += 1

        foreground = cv2.warpAffine(
            frame, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            borderMode=cv2.BORDER_CONSTANT, borderValue=PADDING_COLOR,
        )

        if foreground_fills_frame:
            aligned_frame = foreground
        else:
            # Blur first, then warp — slightly cheaper than warp-then-blur
            background = cv2.GaussianBlur(frame, (41, 41), 0)
            background = cv2.warpAffine(background, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            aligned_frame = np.where(mask == 0, foreground, background)

        out.write(aligned_frame)
        frames_written += 1

        if first_transformed is None:
            first_transformed = aligned_frame

    if DEBUG and first_transformed is not None:
        cv2.imwrite(
            os.path.join(debug_dir, f"{video_name}{face_suffix}_first_frame_transformed.jpg"),
            first_transformed,
        )

    return frames_read, frames_written


def add_audio_to_video(temp_video_path, source_video_path, final_output_path):
    """
    Muxes audio from the original source video into the processed video using FFmpeg.
    Returns True on success, False on FFmpeg error.
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", temp_video_path,
                "-i", source_video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                final_output_path,
                "-y",
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding audio to {final_output_path}: {e}")
        return False


def process_video(processing_path, original_path, output_dir, debug_dir, face_num, left_eye, right_eye, face_suffix):
    """
    Processes a full video:
      1. Reads first frame to compute the transformation matrix (from the OpenCV-safe SDR path).
      2. Writes all aligned frames to a temporary file.
      3. Muxes the original audio back in via FFmpeg (from the original HDR path).
    Returns True if all frames were successfully written and audio muxed.
    """
    cap, fps, first_frame = initialize_video_reader(processing_path)
    if cap is None:
        return False

    video_name     = os.path.splitext(os.path.basename(original_path))[0]
    temp_path      = os.path.join(output_dir, f"temp_{video_name}{face_suffix}.mp4")
    final_out_path = os.path.join(output_dir, f"{video_name}{face_suffix}.mp4")

    M, M2 = compute_transformation_matrix(first_frame, left_eye, right_eye, get_eye_to_screen_ratio(video_name))
    if M is None:
        print(f"Error: Could not compute transform for {original_path} face {face_num}, skipping.")
        cap.release()
        return False

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out    = cv2.VideoWriter(temp_path, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {temp_path}")
        cap.release()
        return False

    frames_read, frames_written = process_frames(cap, out, M, M2, debug_dir, video_name, face_suffix)
    cap.release()
    out.release()

    success = frames_read > 0 and frames_read == frames_written
    if success:
        audio_ok = add_audio_to_video(temp_path, original_path, final_out_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if audio_ok:
            print(f"Processed: {final_out_path} ({frames_written}/{frames_read} frames)")
            return True
        else:
            if os.path.exists(final_out_path):
                os.remove(final_out_path)
            return False
    else:
        print(
            f"Error: Incomplete video for {final_out_path} "
            f"(read: {frames_read}, written: {frames_written})"
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def copy_to_failed(src_path, failed_dir):
    """Copies a failed source file to the failed directory for later inspection."""
    try:
        shutil.copy2(src_path, os.path.join(failed_dir, os.path.basename(src_path)))
        print(f"Copied to failed: {os.path.basename(src_path)}")
    except Exception as e:
        print(f"Error copying {src_path} to failed dir: {e}")


def process_single_video(video_path, output_dir, debug_dir, failed_dir, csv_data):
    """
    Processes all faces listed in the CSV for a given video.
    Returns (video_path, {detected, processed}) for the summary report.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if video_name not in csv_data:
        print(f"No CSV entry for {video_path}, skipping.")
        copy_to_failed(video_path, failed_dir)
        return video_path, {"detected": 0, "processed": 0}

    # --- HDR DETECTION & FIX LOGIC ---
    processing_path = video_path
    temp_sdr_file = None
    
    if is_hdr(video_path):
        print(f"HDR detected for {video_name}. Tone-mapping to SDR...")
        temp_sdr_file = convert_hdr_to_sdr(video_path, output_dir)
        if temp_sdr_file:
            processing_path = temp_sdr_file # OpenCV will read this correctly instead
        else:
            print(f"Warning: Tone-mapping failed for {video_name}, falling back to original file.")
    # ---------------------------------

    # Read first frame to validate the file is readable
    cap = cv2.VideoCapture(processing_path)
    if not cap.isOpened():
        print(f"Error: Could not open {processing_path}")
        copy_to_failed(video_path, failed_dir)
        if temp_sdr_file and os.path.exists(temp_sdr_file): os.remove(temp_sdr_file)
        return video_path, {"detected": 0, "processed": 0}
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error reading first frame: {processing_path}")
        copy_to_failed(video_path, failed_dir)
        if temp_sdr_file and os.path.exists(temp_sdr_file): os.remove(temp_sdr_file)
        return video_path, {"detected": 0, "processed": 0}

    if DEBUG:
        cv2.imwrite(os.path.join(debug_dir, f"{video_name}_first_frame_original.jpg"), first_frame)

    faces = csv_data[video_name]
    processed_count = 0

    for face_num, left_eye, right_eye in faces:
        face_suffix = f"_face{face_num:02d}"
        if process_video(processing_path, video_path, output_dir, debug_dir, face_num, left_eye, right_eye, face_suffix):
            processed_count += 1

    if processed_count == 0:
        copy_to_failed(video_path, failed_dir)

    # Clean up the temporary SDR file when done with all faces
    if temp_sdr_file and os.path.exists(temp_sdr_file):
        try:
            os.remove(temp_sdr_file)
        except OSError as e:
            print(f"Warning: Could not remove temporary SDR file {temp_sdr_file}: {e}")

    return video_path, {"detected": len(faces), "processed": processed_count}


def main():
    clear_directory(OUTPUT_DIR)
    clear_directory(DEBUG_DIR)
    clear_directory(FAILED_DIR)

    csv_data = load_csv_data(CSV_PATH)
    if not csv_data:
        print("No valid CSV data loaded. Exiting.")
        return

    # Collect video files, filtered by the utility helper (e.g. skips already-processed files)
    video_tasks = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(get_filtered_videos(INPUT_DIR))
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    # Process videos in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_single_video, path, OUTPUT_DIR, DEBUG_DIR, FAILED_DIR, csv_data): path
            for path in video_tasks
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