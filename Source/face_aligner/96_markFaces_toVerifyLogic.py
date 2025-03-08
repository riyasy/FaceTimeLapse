import cv2
import dlib
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import clear_directory  # Assuming this is your utility function

# Configurable constants
MAX_WORKERS = 4  # Start with 4 for M1 Mac, adjustable (e.g., 2-8)

# Load dlib's MMOD face detector & landmark predictor
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def initialize_video_reader(video_path):
    """Initializes video capture and reads first frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {video_path}")
        return None, None

    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        return None, None

    return first_frame, os.path.splitext(os.path.basename(video_path))[0]

def annotate_frame(frame, debug_dir, video_name):
    """Annotates the first frame with face rectangles (red) and eye centers (green X)."""
    # Scale down frame for faster detection
    scale_factor = 0.5
    scaled_frame = cv2.resize(
        frame,
        (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)

    # Detect all faces
    faces = detector(gray, 0)
    if len(faces) == 0:
        print(f"No faces detected in {video_name}")
        return False

    # Scale up coordinates to original frame size
    scale_up_factor = 1 / scale_factor
    annotated_frame = frame.copy()

    for face in faces:
        # Scale face rectangle back to original size
        rect = dlib.rectangle(
            int(face.rect.left() * scale_up_factor),
            int(face.rect.top() * scale_up_factor),
            int(face.rect.right() * scale_up_factor),
            int(face.rect.bottom() * scale_up_factor),
        )

        # Draw red rectangle around face
        cv2.rectangle(
            annotated_frame,
            (rect.left(), rect.top()),
            (rect.right(), rect.bottom()),
            (0, 0, 255),  # Red in BGR
            2,  # Thickness
        )

        # Get landmarks for eyes
        landmarks = predictor(frame, rect)  # Use original frame for accuracy

        # Calculate left eye center (average of landmarks 36-41)
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        left_eye_center = (
            int(sum(p[0] for p in left_eye_points) / 6),
            int(sum(p[1] for p in left_eye_points) / 6)
        )

        # Calculate right eye center (average of landmarks 42-47)
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        right_eye_center = (
            int(sum(p[0] for p in right_eye_points) / 6),
            int(sum(p[1] for p in right_eye_points) / 6)
        )

        # Draw green X for left eye center
        cv2.drawMarker(
            annotated_frame,
            left_eye_center,
            (0, 255, 0),  # Green in BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )

        # Draw green X for right eye center
        cv2.drawMarker(
            annotated_frame,
            right_eye_center,
            (0, 255, 0),  # Green in BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )

    # Save annotated frame
    output_path = os.path.join(debug_dir, f"{video_name}_faces_annotated.jpg")
    cv2.imwrite(output_path, annotated_frame)
    return True

def annotate_frame2(frame, debug_dir, video_name):
    """Annotates the first frame with face rectangles (red) and eye positions (green X)."""
    # Scale down frame for faster detection
    scale_factor = 0.5
    scaled_frame = cv2.resize(
        frame,
        (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)

    # Detect all faces
    faces = detector(gray, 0)
    if len(faces) == 0:
        print(f"No faces detected in {video_name}")
        return False

    # Scale up coordinates to original frame size
    scale_up_factor = 1 / scale_factor
    annotated_frame = frame.copy()

    for face in faces:
        # Scale face rectangle back to original size
        rect = dlib.rectangle(
            int(face.rect.left() * scale_up_factor),
            int(face.rect.top() * scale_up_factor),
            int(face.rect.right() * scale_up_factor),
            int(face.rect.bottom() * scale_up_factor),
        )

        # Draw red rectangle around face
        cv2.rectangle(
            annotated_frame,
            (rect.left(), rect.top()),
            (rect.right(), rect.bottom()),
            (0, 0, 255),  # Red in BGR
            2,  # Thickness
        )

        # Get landmarks for eyes
        landmarks = predictor(frame, rect)  # Use original frame for accuracy
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Draw green X for left eye
        cv2.drawMarker(
            annotated_frame,
            left_eye,
            (0, 255, 0),  # Green in BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )

        # Draw green X for right eye
        cv2.drawMarker(
            annotated_frame,
            right_eye,
            (0, 255, 0),  # Green in BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )

    # Save annotated frame
    output_path = os.path.join(debug_dir, f"{video_name}_faces_annotated.jpg")
    cv2.imwrite(output_path, annotated_frame)
    return True

def process_single_video(video_path, input_dir, debug_dir):
    """Processes a single video to annotate its first frame."""
    first_frame, video_name = initialize_video_reader(video_path)
    if first_frame is None:
        return video_path, False

    success = annotate_frame(first_frame, debug_dir, video_name)
    return video_path, success

def process_videos(input_dir="02_input_videos_cfr", debug_dir="99_debug_frames"):
    """Processes all videos in the input directory with multithreading to annotate faces."""
    clear_directory(debug_dir)

    video_tasks = []
    for video in sorted(os.listdir(input_dir)):
        if not video.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        video_path = os.path.join(input_dir, video)
        video_tasks.append(video_path)

    processing_results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {
            executor.submit(process_single_video, video_path, input_dir, debug_dir): video_path
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
        print(f"{status} {os.path.basename(video_path)} - Annotated: {success}")

# Run processing
process_videos("/Users/riyasyoosuf/Desktop/Input/Phase1_CleanedUp/Videos+LivePhotos_4K_CFR", "99_debug_frames")