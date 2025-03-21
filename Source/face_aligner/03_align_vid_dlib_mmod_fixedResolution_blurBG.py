import cv2
import dlib
import numpy as np
import os
import math
import subprocess
import shutil

from util import clear_directory

# Configurable output resolution constants
OUTPUT_WIDTH = 4096
OUTPUT_HEIGHT = 2160

# Padding color for foreground (RGB)
PADDING_COLOR = (57, 255, 20)

# Load dlib's MMOD face detector & landmark predictor
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# === Face Detection Functions ===
def detect_face_and_eyes(image, face_rect=None):
    """Detects face and eyes using dlib MMOD, returns face center, eye distance, and eye positions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if face_rect is None:
        faces = detector(gray, 0)
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda x: x.rect.width() * x.rect.height(), reverse=True)
        face = faces[0].rect
    else:
        face = face_rect

    landmarks = predictor(gray, face)
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    face_center = ((face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2)
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    return face_center, eye_distance, left_eye, right_eye

def compute_transformation_matrix(frame, face_rect, target_eye_ratio=0.15):
    """Computes transformation matrices to align face with proper scaling and rotation."""
    h, w = frame.shape[:2]
    face_data = detect_face_and_eyes(frame, face_rect)

    if face_data is None:
        return None, None

    face_center, eye_distance, left_eye, right_eye = face_data

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    target_eye_distance = OUTPUT_WIDTH * target_eye_ratio
    scale_factor = target_eye_distance / eye_distance

    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor)
    M2 = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor * 2)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            if frames_read == 0:
                print(f"Error: Failed to read any frames from video")
            break
            
        frames_read += 1
        
        # Apply M2 transformation with default (black) padding and blur for background
        background = cv2.warpAffine(frame, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        background = cv2.GaussianBlur(background, (41, 41), 0)
        
        # Apply M transformation with specific padding color for foreground
        foreground = cv2.warpAffine(frame, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=PADDING_COLOR)
        
        # Create mask based on exact padding color
        mask = (foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255
        mask = cv2.merge([mask, mask, mask])  # Convert to 3-channel
        
        # Combine background and foreground (where mask is 0, use foreground)
        aligned_frame = np.where(mask == 0, foreground, background)
        
        out.write(aligned_frame)
        frames_written += 1

        if first_frame_transformed is None:
            first_frame_transformed = aligned_frame

    if first_frame_transformed is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{video_name}{face_suffix}_first_frame_transformed.jpg"), first_frame_transformed)
    
    return frames_read, frames_written

def add_audio_to_video(temp_video_path, video_path, final_output_path):
    """Adds audio from original video to processed video using FFmpeg."""
    try:
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_video_path,
            '-i', video_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            final_output_path,
            '-y'
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding audio to {final_output_path}: {e}")
        return False

def process_video(video_path, output_path, debug_dir, face_rect=None, face_suffix=""):
    """Main video processing function coordinating all steps."""
    cap, fps, first_frame = initialize_video_reader(video_path)
    if cap is None:
        return False

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_video_path = os.path.join(os.path.dirname(output_path), f"temp_{video_name}{face_suffix}.mp4")
    final_output_path = os.path.join(os.path.dirname(output_path), f"{video_name}{face_suffix}.mp4")

    M, M2 = compute_transformation_matrix(first_frame, face_rect)
    if M is None or M2 is None:
        print(f"No face detected in first frame of {video_path} for {face_suffix}, skipping.")
        cap.release()
        return False

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not out.isOpened():
        print(f"Error: Could not open output video writer for {temp_video_path}")
        cap.release()
        return False

    frames_read, frames_written = process_frames(cap, out, M, M2, debug_dir, video_name, face_suffix)
    success = frames_read > 0 and frames_read == frames_written

    cap.release()
    out.release()

    if success:
        audio_success = add_audio_to_video(temp_video_path, video_path, final_output_path)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        if audio_success:
            print(f"Processed: {final_output_path} ({frames_written}/{frames_read} frames)")
            return True
        else:
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            return False
    else:
        print(f"Error: Video processing failed for {final_output_path} "
              f"(read: {frames_read}, written: {frames_written}), removing incomplete file")
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

def process_videos(video_dir, output_dir, debug_dir, failed_dir="failed_videos"):
    """Processes all videos in the folder and prints summary."""
    clear_directory(output_dir)
    clear_directory(debug_dir)
    clear_directory(failed_dir)
    
    processing_results = {}

    for video in sorted(os.listdir(video_dir)):
        if not video.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        video_path = os.path.join(video_dir, video)
        output_path = os.path.join(output_dir, video)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            processing_results[video_path] = {'detected': 0, 'processed': 0}
            copy_to_failed_videos(video_path, failed_dir)
            continue
            
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error reading {video_path}")
            processing_results[video_path] = {'detected': 0, 'processed': 0}
            copy_to_failed_videos(video_path, failed_dir)
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"{video_name}_first_frame_original.jpg"), first_frame)

        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if len(faces) == 0:
            print(f"No faces detected in {video_path}, skipping.")
            processing_results[video_path] = {'detected': 0, 'processed': 0}
            copy_to_failed_videos(video_path, failed_dir)
            continue

        faces = sorted(faces, key=lambda x: x.rect.width() * x.rect.height(), reverse=True)
        faces_to_process = [f.rect for f in faces[:min(3, len(faces))]]
        processed_count = 0

        for i, face in enumerate(faces_to_process, 1):
            face_suffix = f"_face{i:02d}"
            success = process_video(video_path, output_path, debug_dir, face, face_suffix)
            if success:
                processed_count += 1

        processing_results[video_path] = {'detected': len(faces), 'processed': processed_count}
        
        if processed_count == 0:
            copy_to_failed_videos(video_path, failed_dir)

    print("\n=== Processing Summary ===")
    for video_path, stats in processing_results.items():
        status = "✓" if stats['processed'] > 0 else "✗"
        print(f"{status} {os.path.basename(video_path)} - Faces detected: {stats['detected']}, Processed: {stats['processed']}")

# Run processing
process_videos("02_input_videos_cfr", "08_output_videos", "99_debug_frames", "92_failed_videos")