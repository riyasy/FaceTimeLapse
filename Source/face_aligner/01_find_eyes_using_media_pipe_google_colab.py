import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import cv2
import csv

# --- CONFIGURATION ---
INPUT_FOLDER          = "Input"
CSV_FILE              = "annotated_by_media_pipe.csv"
SAVE_ANNOTATED_IMAGES = False  # Set True to also save annotated JPEGs to OUTPUT_FOLDER
OUTPUT_FOLDER         = "Output"
MODEL_PATH            = "face_landmarker_v2_with_blendshapes.task"
MAX_FACES             = 3
# ---------------------

# MediaPipe Face Mesh landmark indices for irises and lips
LEFT_IRIS_CENTER   = 468
RIGHT_IRIS_CENTER  = 473
LEFT_IRIS_CONTOUR  = [469, 470, 471, 472]
RIGHT_IRIS_CONTOUR = [474, 475, 476, 477]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40]
LIPS_INNER = [78,  95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80]

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def draw_face_annotations(image, pts):
    """Draws iris circles and lip contours onto a copy of the image (only used when SAVE_ANNOTATED_IMAGES is True)."""
    annotated = np.copy(image)

    # Draw outer and inner lip contours
    cv2.polylines(annotated, [np.array([pts[i] for i in LIPS_OUTER])], isClosed=True, color=(0, 255, 0), thickness=1)
    cv2.polylines(annotated, [np.array([pts[i] for i in LIPS_INNER])], isClosed=True, color=(0, 255, 0), thickness=1)

    if len(pts) > RIGHT_IRIS_CENTER:
        # Draw left iris center and contour
        cv2.circle(annotated, tuple(pts[LEFT_IRIS_CENTER]), 2, (0, 0, 255), -1)
        cv2.polylines(annotated, [np.array([pts[i] for i in LEFT_IRIS_CONTOUR])], isClosed=True, color=(0, 0, 255), thickness=1)
        # Draw right iris center and contour
        cv2.circle(annotated, tuple(pts[RIGHT_IRIS_CENTER]), 2, (0, 0, 255), -1)
        cv2.polylines(annotated, [np.array([pts[i] for i in RIGHT_IRIS_CONTOUR])], isClosed=True, color=(0, 0, 255), thickness=1)

    return annotated


def process_landmarks(rgb_image, detection_result, csv_writer, image_file):
    """
    Extracts iris centers from all detected faces and writes one CSV row per face.
    Optionally draws face annotations if SAVE_ANNOTATED_IMAGES is enabled.

    Returns the annotated image array (or None if annotation is disabled).
    """
    height, width = rgb_image.shape[:2]
    annotated_image = None

    for idx, face_landmarks in enumerate(detection_result.face_landmarks):
        # Convert normalised (0–1) landmark coords to pixel coordinates
        pts = np.array(
            [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks],
            dtype=np.int32,
        )

        if SAVE_ANNOTATED_IMAGES:
            # Start from a fresh copy of the image for each face so drawings don't stack
            annotated_image = draw_face_annotations(rgb_image if annotated_image is None else annotated_image, pts)

        # Extract iris pixel coordinates (or 'N/A' if the landmark is missing)
        iris_left  = f"{pts[LEFT_IRIS_CENTER][0]},{pts[LEFT_IRIS_CENTER][1]}"   if len(pts) > LEFT_IRIS_CENTER  else "N/A"
        iris_right = f"{pts[RIGHT_IRIS_CENTER][0]},{pts[RIGHT_IRIS_CENTER][1]}" if len(pts) > RIGHT_IRIS_CENTER else "N/A"

        print(f"Face {idx+1} in {image_file}: Left Iris: {iris_left} | Right Iris: {iris_right}")
        csv_writer.writerow([os.path.basename(image_file), idx + 1, iris_left, iris_right])

    return annotated_image


def main():
    if SAVE_ANNOTATED_IMAGES:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Output folder ready: {OUTPUT_FOLDER}")

    # Initialise the MediaPipe Face Landmarker (Tasks API)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=MAX_FACES,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    with open(CSV_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["input_file_name", "face_number", "iris_left_value", "iris_right_value"])

        for image_file in sorted(os.listdir(INPUT_FOLDER)):
            if not image_file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(INPUT_FOLDER, image_file)

            # Load image via MediaPipe's own API (expects RGB internally)
            try:
                mp_image = mp.Image.create_from_file(image_path)
            except Exception as e:
                print(f"Error loading {image_file}: {e}")
                csv_writer.writerow([os.path.basename(image_file), 1, "N/A", "N/A"])
                continue

            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                annotated_image = process_landmarks(
                    mp_image.numpy_view(), detection_result, csv_writer, image_file
                )
            else:
                # No faces found — record a blank row so every input has an entry
                print(f"No faces detected in {image_file}")
                csv_writer.writerow([os.path.basename(image_file), 1, "N/A", "N/A"])
                annotated_image = np.copy(mp_image.numpy_view()) if SAVE_ANNOTATED_IMAGES else None

            # Optionally save the annotated image (BGR conversion required for cv2.imwrite)
            if SAVE_ANNOTATED_IMAGES and annotated_image is not None:
                output_path = os.path.join(OUTPUT_FOLDER, f"processed_{image_file}")
                cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                print(f"Saved: {output_path}")

    print(f"\nDone. Results saved to: {CSV_FILE}")


if __name__ == "__main__":
    main()