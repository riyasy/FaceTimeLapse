import cv2
import numpy as np
import os
import csv
from insightface.app import FaceAnalysis

# --- CONFIGURATION ---
INPUT_FOLDER = "01_extracted_frames"
CSV_FILE     = "02_annotated_csv/annotated_by_insightface.csv"
# ---------------------

# --- INSIGHTFACE SETUP ---
# buffalo_s is highly optimized for CPUs.
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])

# det_size=(640, 640) is InsightFace's default internal resolution scale. 
# It is fast enough for CPUs. If you want extreme speed and your faces are 
# always very large in the frame, you can lower this to (320, 320).
app.prepare(ctx_id=0, det_size=(640, 640))
# -------------------------

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')


def detect_eyes_in_frame(frame):
    """
    Detects up to 2 faces in a BGR frame using InsightFace.
    Returns eye centers for each, sorted by face area descending (largest first).
    """
    faces = app.get(frame)

    if len(faces) == 0:
        return []

    # Sort faces by bounding box area (width * height), largest first
    faces_sorted = sorted(
        faces, 
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), 
        reverse=True
    )[:2] # Keep only top 2 largest faces

    results = []
    for face in faces_sorted:
        # InsightFace provides 5 keypoints (kps) natively:
        # kps[0] = Viewer's Left Eye center
        # kps[1] = Viewer's Right Eye center
        # kps[2] = Nose
        # kps[3] = Viewer's Left Mouth Corner
        # kps[4] = Viewer's Right Mouth Corner
        
        # Extract eye centers and convert float coordinates to integers
        left_eye  = (int(face.kps[0][0]), int(face.kps[0][1]))
        right_eye = (int(face.kps[1][0]), int(face.kps[1][1]))
        
        results.append((left_eye, right_eye))

    return results


def load_frame_from_image(path):
    """Loads a BGR frame directly from an image file."""
    frame = cv2.imread(path)
    if frame is None:
        raise IOError(f"Could not read image: {path}")
    return frame


def load_first_frame_from_video(path):
    """Opens a video and returns only its first frame as a BGR image."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise IOError(f"Could not read first frame of: {path}")
    return frame


def format_coord(xy):
    """Formats an (x, y) tuple as 'x,y'."""
    return f"{xy[0]},{xy[1]}"


def main():
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    with open(CSV_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Header matches the MediaPipe output CSV for interoperability
        csv_writer.writerow(["input_file_name", "face_number", "iris_left_value", "iris_right_value"])

        for filename in sorted(os.listdir(INPUT_FOLDER)):
            lower    = filename.lower()
            filepath = os.path.join(INPUT_FOLDER, filename)

            # Determine whether we are working with an image or video
            if lower.endswith(IMAGE_EXTENSIONS):
                file_type = "image"
            elif lower.endswith(VIDEO_EXTENSIONS):
                file_type = "video"
            else:
                continue  # Skip unsupported file types

            # Load the relevant frame
            try:
                if file_type == "image":
                    frame = load_frame_from_image(filepath)
                else:
                    frame = load_first_frame_from_video(filepath)
                    print(f"[video] Extracted first frame: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                csv_writer.writerow([filename, 1, "N/A", "N/A"])
                continue

            # Detect faces and extract eye coordinates
            try:
                eye_results = detect_eyes_in_frame(frame)
            except Exception as e:
                print(f"Error detecting faces in {filename}: {e}")
                csv_writer.writerow([filename, 1, "N/A", "N/A"])
                continue

            if not eye_results:
                print(f"No faces detected in {filename}")
                csv_writer.writerow([filename, 1, "N/A", "N/A"])
                continue

            # Write one CSV row per detected face
            for face_idx, (left_eye, right_eye) in enumerate(eye_results, start=1):
                left_str  = format_coord(left_eye)
                right_str = format_coord(right_eye)
                print(f"Face {face_idx} in {filename}: Left Eye: {left_str} | Right Eye: {right_str}")
                csv_writer.writerow([filename, face_idx, left_str, right_str])

    print(f"\nDone. Results saved to: {CSV_FILE}")


if __name__ == "__main__":
    main()