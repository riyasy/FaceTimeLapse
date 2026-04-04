import cv2
import dlib
import numpy as np
import os
import csv

# --- CONFIGURATION ---
INPUT_FOLDER = "01_extracted_frames"
CSV_FILE     = "02_annotated_csv/annotated_by_dlib.csv"
# ---------------------

# dlib requires two model files in the working directory:
#   - mmod_human_face_detector.dat      (CNN face detector)
#   - shape_predictor_68_face_landmarks.dat  (68-point landmark predictor)
detector  = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')


def get_eye_center(landmarks, outer_idx, inner_idx):
    """Returns the pixel midpoint between two eye-corner landmarks."""
    x = (landmarks.part(outer_idx).x + landmarks.part(inner_idx).x) // 2
    y = (landmarks.part(outer_idx).y + landmarks.part(inner_idx).y) // 2
    return x, y


def detect_eyes_in_frame(frame):
    """
    Detects up to 3 faces in a BGR frame and returns eye midpoints for each.

    Detection is run on a 50%-scaled frame for speed, then scaled back to
    full resolution before running the 68-point landmark predictor.

    Returns a list of (left_eye, right_eye) tuples (pixel coords),
    sorted by face area descending (largest face first).
      - left_eye:  midpoint of landmarks 36 (outer) & 39 (inner)  [subject's left]
      - right_eye: midpoint of landmarks 42 (inner) & 45 (outer)  [subject's right]
    """
    # Scale down for faster MMOD detection
    scale = 0.5
    small = cv2.resize(
        frame,
        (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    detections = detector(gray_small, 0)

    if len(detections) == 0:
        return []

    # Sort largest face first, cap at 3
    scale_up = 1.0 / scale
    detections = sorted(
        detections, key=lambda d: d.rect.width() * d.rect.height(), reverse=True
    )[:3]

    # Run landmark predictor at full resolution for accuracy
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    for det in detections:
        rect = dlib.rectangle(
            int(det.rect.left()   * scale_up),
            int(det.rect.top()    * scale_up),
            int(det.rect.right()  * scale_up),
            int(det.rect.bottom() * scale_up),
        )
        landmarks = predictor(gray_full, rect)
        left_eye  = get_eye_center(landmarks, 36, 39)
        right_eye = get_eye_center(landmarks, 42, 45)
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
    """Formats an (x, y) tuple as 'x,y' — matches the MediaPipe CSV column style."""
    return f"{xy[0]},{xy[1]}"


def main():
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
