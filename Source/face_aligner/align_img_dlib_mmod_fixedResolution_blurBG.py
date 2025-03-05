import cv2
import dlib
import numpy as np
import os
import math
import shutil

# Configurable output resolution constants
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
FPS = 30  # Fixed FPS for output videos
VIDEO_DURATION_FRAMES = 30  # 1 second at 30 FPS

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
        faces = detector(gray, 1)
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

# === Image Processing Functions ===
def initialize_image_reader(image_path):
    """Reads the input image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    return image

def process_image_to_video(image, out, M, M2, debug_dir, image_name, face_suffix):
    """Processes the image and writes it as a 30-frame video."""
    # Apply M2 transformation with default (black) padding and blur for background
    background = cv2.warpAffine(image, M2, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    background = cv2.GaussianBlur(background, (41, 41), 0)
    
    # Apply M transformation with specific padding color for foreground
    foreground = cv2.warpAffine(image, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=PADDING_COLOR)
    
    # Create mask based on exact padding color
    mask = (foreground == PADDING_COLOR).all(axis=2).astype(np.uint8) * 255
    mask = cv2.merge([mask, mask, mask])
    
    # Combine background and foreground
    transformed_frame = np.where(mask == 0, foreground, background)
    
    # Write the transformed frame 30 times for a 1-second video
    for _ in range(VIDEO_DURATION_FRAMES):
        out.write(transformed_frame)

    # Save the transformed frame to debug folder
    cv2.imwrite(os.path.join(debug_dir, f"{image_name}{face_suffix}_first_frame_transformed.jpg"), transformed_frame)
    
    return True  # Success if we reach here

def process_image(image_path, output_path, debug_dir, face_rect=None, face_suffix=""):
    """Main image processing function coordinating all steps."""
    image = initialize_image_reader(image_path)
    if image is None:
        return False

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_video_path = os.path.join(os.path.dirname(output_path), f"{image_name}{face_suffix}.mp4")

    M, M2 = compute_transformation_matrix(image, face_rect)
    if M is None or M2 is None:
        print(f"No face detected in image {image_path} for {face_suffix}, skipping.")
        return False

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not out.isOpened():
        print(f"Error: Could not open output video writer for {output_video_path}")
        return False

    success = process_image_to_video(image, out, M, M2, debug_dir, image_name, face_suffix)
    
    out.release()
    
    if success:
        print(f"Processed: {output_video_path} ({VIDEO_DURATION_FRAMES} frames)")
        return True
    else:
        print(f"Error: Image processing failed for {output_video_path}, removing incomplete file")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        return False

# === Main Processing Function ===
def clear_directory(directory):
    """Removes all files and subdirectories in the specified directory."""
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error clearing {item_path}: {e}")
        print(f"Cleared directory: {directory}")
    else:
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def copy_to_failed_images(image_path, failed_dir):
    """Copies the input image to the failed_images directory."""
    try:
        failed_path = os.path.join(failed_dir, os.path.basename(image_path))
        shutil.copy2(image_path, failed_path)
        print(f"Copied failed image to: {failed_path}")
    except Exception as e:
        print(f"Error copying {image_path} to failed_images: {e}")

def process_images(input_dir, output_dir, debug_dir, failed_dir="failed_images"):
    """Processes all images in the folder and prints summary."""
    clear_directory(output_dir)
    clear_directory(debug_dir)
    clear_directory(failed_dir)
    
    processing_results = {}

    for image in sorted(os.listdir(input_dir)):
        if not image.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        image_path = os.path.join(input_dir, image)
        output_path = os.path.join(output_dir, image)  # Base path, modified later

        image_data = cv2.imread(image_path)
        if image_data is None:
            print(f"Error: Could not open {image_path}")
            processing_results[image_path] = {'detected': 0, 'processed': 0}
            copy_to_failed_images(image_path, failed_dir)
            continue

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"{image_name}_first_frame_original.jpg"), image_data)

        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if len(faces) == 0:
            print(f"No faces detected in {image_path}, skipping.")
            processing_results[image_path] = {'detected': 0, 'processed': 0}
            copy_to_failed_images(image_path, failed_dir)
            continue

        faces = sorted(faces, key=lambda x: x.rect.width() * x.rect.height(), reverse=True)
        faces_to_process = [f.rect for f in faces[:min(3, len(faces))]]
        processed_count = 0

        for i, face in enumerate(faces_to_process, 1):
            face_suffix = f"_face{i:02d}"
            success = process_image(image_path, output_path, debug_dir, face, face_suffix)
            if success:
                processed_count += 1

        processing_results[image_path] = {'detected': len(faces), 'processed': processed_count}
        
        if processed_count == 0:
            copy_to_failed_images(image_path, failed_dir)

    print("\n=== Processing Summary ===")
    for image_path, stats in processing_results.items():
        status = "✓" if stats['processed'] > 0 else "✗"
        print(f"{status} {os.path.basename(image_path)} - Faces detected: {stats['detected']}, Processed: {stats['processed']}")

# Run processing
process_images("input_images", "output_videos", "debug_frames", "failed_images")