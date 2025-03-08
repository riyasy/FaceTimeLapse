import cv2
import numpy as np
import os

# Load OpenCV's pre-trained face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_face_and_eyes(image):
    """Detects face and eyes, returns face center and eye distance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return None

    # Use the largest detected face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) < 2:
        return None  # Both eyes not detected, skip this frame

    # Sort eyes by x position (left to right)
    eyes = sorted(eyes, key=lambda e: e[0])
    left_eye = (x + eyes[0][0] + eyes[0][2] // 2, y + eyes[0][1] + eyes[0][3] // 2)
    right_eye = (x + eyes[1][0] + eyes[1][2] // 2, y + eyes[1][1] + eyes[1][3] // 2)

    face_center = (x + w // 2, y + h // 2)
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    return face_center, eye_distance, left_eye, right_eye

def compute_transformation_matrix(frame, target_eye_ratio=0.05):
    """Computes transformation matrix to center face and scale based on eye distance."""
    h, w = frame.shape[:2]
    face_data = detect_face_and_eyes(frame)

    if face_data is None:
        return None  # No face or both eyes not detected

    face_center, eye_distance, _, _ = face_data

    # Target eye distance (20% of frame width)
    target_eye_distance = w * target_eye_ratio
    scale_factor = target_eye_distance / eye_distance

    # Compute the translation needed to center the face
    center_x, center_y = w / 2, h / 2
    tx = center_x - face_center[0]  # Move face to center
    ty = center_y - face_center[1]  

    # Apply scaling correctly
    M = np.array([
        [scale_factor, 0, scale_factor * tx + (1 - scale_factor) * center_x],  
        [0, scale_factor, scale_factor * ty + (1 - scale_factor) * center_y]   
    ], dtype=np.float32)

    return M


def process_video(video_path, output_path, debug_dir):
    """Applies transformation to all frames of a video and saves first frames for debugging."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read first frame and calculate transformation matrix
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        return

    video_name = os.path.basename(video_path)

    # Save the first frame BEFORE transformation
    cv2.imwrite(os.path.join(debug_dir, f"{video_name}_first_frame_original.jpg"), first_frame)
    print(f"Saved original first frame: {video_name}_first_frame_original.jpg")

    M = compute_transformation_matrix(first_frame)
    if M is None:
        print(f"No face detected in first frame of {video_path}, skipping.")
        return

    # Apply transformation to all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    first_frame_transformed = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        aligned_frame = cv2.warpAffine(frame, M, (width, height))
        out.write(aligned_frame)

        if first_frame_transformed is None:
            first_frame_transformed = aligned_frame

    # Save the first frame AFTER transformation
    if first_frame_transformed is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{video_name}_first_frame_transformed.jpg"), first_frame_transformed)
        print(f"Saved transformed first frame: {video_name}_first_frame_transformed.jpg")

    cap.release()
    out.release()
    print(f"Processed: {output_path}")

def process_videos(video_dir, output_dir, debug_dir):
    """Processes all videos in the folder."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    for video in sorted(os.listdir(video_dir)):
        if not video.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        video_path = os.path.join(video_dir, video)
        output_path = os.path.join(output_dir, video)
        process_video(video_path, output_path, debug_dir)

# Run processing
process_videos("02_input_videos_cfr", "08_output_videos", "99_debug_frames")