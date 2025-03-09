# Import all necessary modules
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import cv2
import csv
from google.colab.patches import cv2_imshow  # For Colab environment

# Function to draw iris and lips with coordinates
def draw_landmarks_on_image(rgb_image, detection_result, csv_writer, image_file):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    height, width = annotated_image.shape[:2]

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])
        
        # Draw iris and lips
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())

        # Calculate face rectangle from landmarks
        x_coords = [int(landmark.x * width) for landmark in face_landmarks]
        y_coords = [int(landmark.y * height) for landmark in face_landmarks]
        if x_coords and y_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            w = max_x - min_x
            h = max_y - min_y
            # Add 10% padding
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            face_rect = f"{max(0, min_x - pad_w)},{max(0, min_y - pad_h)},{w + 2*pad_w},{h + 2*pad_h}"
        else:
            face_rect = "N/A"

        # Get iris coordinates
        iris_indices = [468, 473]  # Right iris center (468), Left iris center (473)
        iris_right, iris_left = "N/A", "N/A"
        
        for iris_idx in iris_indices:
            if iris_idx < len(face_landmarks):
                landmark = face_landmarks[iris_idx]
                x_pixel = int(landmark.x * width)
                y_pixel = int(landmark.y * height)
                if iris_idx == 468:  # Right iris
                    iris_right = f"{x_pixel},{y_pixel}"
                elif iris_idx == 473:  # Left iris
                    iris_left = f"{x_pixel},{y_pixel}"
                print(f"{'Right' if iris_idx == 468 else 'Left'} Iris (Landmark {iris_idx}): ({x_pixel}, {y_pixel})")

        # Write to CSV
        csv_writer.writerow([
            os.path.basename(image_file),
            idx + 1,
            face_rect,
            iris_left,
            iris_right
        ])

    return annotated_image

# Main execution code
def main():
    # Define input and output directories
    input_folder = "Input"  # Adjust this path as needed
    output_folder = "Output3"  # Adjust this path as needed
    csv_file = "analysis.csv"  # CSV in base directory

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Create FaceLandmarker object
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=3
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # Open CSV file for writing
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["input_file_name", "face_number", "face_rect_value", "iris_left_value", "iris_right_value"])

        # Process each image in the input folder
        for image_file in os.listdir(input_folder):
            if image_file.lower().endswith(valid_extensions):
                image_path = os.path.join(input_folder, image_file)
                
                # Load the input image
                try:
                    image = mp.Image.create_from_file(image_path)
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
                    csv_writer.writerow([os.path.basename(image_file), 1, "N/A", "N/A", "N/A"])
                    continue

                # Detect face landmarks
                detection_result = detector.detect(image)

                # Process and visualize the result
                if detection_result.face_landmarks:
                    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result, csv_writer, image_file)
                else:
                    annotated_image = np.copy(image.numpy_view())
                    csv_writer.writerow([os.path.basename(image_file), 1, "N/A", "N/A", "N/A"])
                    print(f"No faces detected in {image_file}")

                # Save the processed image
                output_path = os.path.join(output_folder, f"processed_{image_file}")
                cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                print(f"Saved processed image to: {output_path}")

if __name__ == "__main__":
    main()