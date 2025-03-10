import cv2
import dlib
import numpy as np
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import clear_directory  # Assuming this is your utility function

# Configurable output resolution constants
OUTPUT_WIDTH = 4096
OUTPUT_HEIGHT = 2160
MAX_WORKERS = 4  # Adjustable for your system

# Padding color for foreground (RGB)
PADDING_COLOR = (57, 255, 20)
# Fluorescent green for eye markers (RGB)
EYE_COLOR = (0, 255, 0)

# Load dlib's MMOD face detector & landmark predictor
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_face_and_eyes(image, face_rect=None):
    """Detects face and computes eye midpoints for alignment."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face_rect)

    # Compute midpoint between inner (39, 42) and outer (36, 45) eye corners
    left_eye = (
        (landmarks.part(36).x + landmarks.part(39).x) // 2,
        (landmarks.part(36).y + landmarks.part(39).y) // 2,
    )
    right_eye = (
        (landmarks.part(42).x + landmarks.part(45).x) // 2,
        (landmarks.part(42).y + landmarks.part(45).y) // 2,
    )

    face_center = (
        (face_rect.left() + face_rect.right()) // 2,
        (face_rect.top() + face_rect.bottom()) // 2,
    )
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    return face_center, eye_distance, left_eye, right_eye

def compute_transformation_matrix(frame, face_rect, target_eye_ratio=0.105):
    """Computes transformation matrices to align face with proper scaling and rotation."""
    h, w = frame.shape[:2]
    face_data = detect_face_and_eyes(frame, face_rect)
    if face_data is None:
        return None, None, None, None
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
    return M, M2, left_eye, right_eye

def process_first_frame(video_path, output_dir, debug_dir, face_rect=None, face_suffix=""):
    """Process the first frame with full transformation and eye marking."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        cap.release()
        return False
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}{face_suffix}_frame.jpg")
    
    # Compute transformations and get eye positions
    M, M2, orig_left_eye, orig_right_eye = compute_transformation_matrix(frame, face_rect)
    if M is None or M2 is None:
        print(f"No face detected in first frame of {video_path}")
        cap.release()
        return False
    
    # Transform foreground
    foreground = cv2.warpAffine(
        frame,
        M,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PADDING_COLOR,
    )
    
    # Create mask
    mask = (foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255
    mask = cv2.merge([mask, mask, mask])
    
    # Transform and blur background
    background = cv2.warpAffine(frame, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    background = cv2.GaussianBlur(background, (41, 41), 0)
    
    # Combine foreground and background
    transformed_frame = np.where(mask == 0, foreground, background)
    
    # Transform eye coordinates
    left_eye = cv2.transform(np.array([[orig_left_eye]], dtype=np.float32), M)[0][0]
    right_eye = cv2.transform(np.array([[orig_right_eye]], dtype=np.float32), M)[0][0]
    
    # Mark eyes
    cv2.circle(transformed_frame, (int(left_eye[0]), int(left_eye[1])), 20, EYE_COLOR, -1)
    cv2.circle(transformed_frame, (int(right_eye[0]), int(right_eye[1])), 20, EYE_COLOR, -1)
    
    # Save output
    cv2.imwrite(output_path, transformed_frame)
    if os.path.exists(output_path):
        print(f"Saved: {output_path}")
    
    cap.release()
    return True

def process_single_video(video_path, output_dir, debug_dir, failed_dir):
    """Process the first frame of a single video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return video_path, {"detected": 0, "processed": 0}
    
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error reading {video_path}")
        return video_path, {"detected": 0, "processed": 0}
    
    # Detect faces
    scale_factor = 0.5
    scaled_frame = cv2.resize(
        first_frame,
        (int(first_frame.shape[1] * scale_factor), int(first_frame.shape[0] * scale_factor)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    if len(faces) == 0:
        print(f"No faces detected in {video_path}")
        return video_path, {"detected": 0, "processed": 0}
    
    scale_up_factor = 1 / scale_factor
    faces = sorted(faces, key=lambda x: x.rect.width() * x.rect.height(), reverse=True)
    faces_to_process = [
        dlib.rectangle(
            int(f.rect.left() * scale_up_factor),
            int(f.rect.top() * scale_up_factor),
            int(f.rect.right() * scale_up_factor),
            int(f.rect.bottom() * scale_up_factor),
        )
        for f in faces[:min(3, len(faces))]
    ]
    
    processed_count = 0
    for i, face in enumerate(faces_to_process, 1):
        face_suffix = f"_face{i:02d}"
        success = process_first_frame(video_path, output_dir, debug_dir, face, face_suffix)
        if success:
            processed_count += 1
    
    return video_path, {"detected": len(faces), "processed": processed_count}

def process_videos(video_dir, output_dir, debug_dir, failed_dir="failed_videos"):
    """Process first frames of all videos in directory with multithreading."""
    clear_directory(output_dir)
    clear_directory(debug_dir)
    clear_directory(failed_dir)
    
    processing_results = {}
    video_tasks = []
    for video in sorted(os.listdir(video_dir)):
        if not video.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        video_path = os.path.join(video_dir, video)
        video_tasks.append(video_path)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {
            executor.submit(process_single_video, video_path, output_dir, debug_dir, failed_dir): 
            video_path for video_path in video_tasks
        }
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                video_path, result = future.result()
                processing_results[video_path] = result
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                processing_results[video_path] = {"detected": 0, "processed": 0}
    
    print("\n=== Processing Summary ===")
    for video_path, stats in processing_results.items():
        status = "✓" if stats["processed"] > 0 else "✗"
        print(
            f"{status} {os.path.basename(video_path)} - Faces detected: {stats['detected']}, Processed: {stats['processed']}"
        )

# Run processing
process_videos(
    "01_input_videos_vfr", "08_output_videos", "99_debug_frames", "92_failed_videos"
)