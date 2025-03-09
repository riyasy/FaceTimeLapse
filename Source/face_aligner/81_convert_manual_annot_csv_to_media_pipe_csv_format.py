import csv
import json
import os

# Input and output CSV paths
INPUT_CSV = "80_media_pipe_data/manual_annotation.csv"  # out put of via tool
OUTPUT_CSV = "80_media_pipe_data/manual_annotation_converted.csv"

def convert_csv(input_path, output_path):
    """Converts the new CSV format to the old format with proper quoting."""
    # Dictionary to store eye data per filename
    eye_data = {}

    # Read the new CSV
    with open(input_path, newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            filename = os.path.splitext(row['filename'])[0]  # Remove extension
            region_id = int(row['region_id'])
            shape_attrs = json.loads(row['region_shape_attributes'])
            cx = shape_attrs['cx']
            cy = shape_attrs['cy']

            if filename not in eye_data:
                eye_data[filename] = {'left': None, 'right': None}
            
            if region_id == 0:  # Left eye
                eye_data[filename]['left'] = f"{cx},{cy}"
            elif region_id == 1:  # Right eye
                eye_data[filename]['right'] = f"{cx},{cy}"

    # Write to the old CSV format
    with open(output_path, 'w', newline='') as outfile:
        fieldnames = ['input_file_name', 'face_number', 'face_rect_value', 'iris_left_value', 'iris_right_value']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename, eyes in eye_data.items():
            # Handle missing eye data
            if eyes['left'] is None or eyes['right'] is None:
                print(f"Warning: Incomplete eye data for {filename}, setting to N/A")
                left_eye = eyes['left'] if eyes['left'] else "N/A"
                right_eye = eyes['right'] if eyes['right'] else "N/A"
            else:
                left_eye = eyes['left']
                right_eye = eyes['right']

            writer.writerow({
                'input_file_name': filename,
                'face_number': 1,  # Assuming one face per file
                'face_rect_value': "N/A",  # Not provided in new CSV
                'iris_left_value': left_eye,  # No extra quotes
                'iris_right_value': right_eye  # No extra quotes
            })

    print(f"Conversion complete. Output written to {output_path}")




# Run the conversion
convert_csv(INPUT_CSV, OUTPUT_CSV)