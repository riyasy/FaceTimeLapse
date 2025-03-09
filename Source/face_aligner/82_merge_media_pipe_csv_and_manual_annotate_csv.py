import csv

# Input and output CSV paths
MEDIA_PIPE_CSV = "80_media_pipe_data/media_pipe_output.csv"  # MediaPipe output CSV
MANUAL_CSV = "80_media_pipe_data/manual_annotation_converted.csv"  # Converted manual annotations CSV
OUTPUT_CSV = "80_media_pipe_data/media_pipe_output_merged.csv"  # Merged output CSV

def merge_csvs(media_pipe_path, manual_path, output_path):
    """Merges two CSVs by filtering out overlapping entries and appending manual data."""
    # Step 1: Load manual annotations to identify filenames to exclude
    manual_filenames = set()
    manual_data = []
    
    with open(manual_path, newline='') as manual_file:
        manual_reader = csv.DictReader(manual_file)
        for row in manual_reader:
            manual_filenames.add(row['input_file_name'])
            manual_data.append(row)
    
    # Step 2: Load and filter MediaPipe data, excluding overlapping filenames
    media_pipe_data = []
    
    with open(media_pipe_path, newline='') as media_pipe_file:
        media_pipe_reader = csv.DictReader(media_pipe_file)
        for row in media_pipe_reader:
            if row['input_file_name'] not in manual_filenames:
                media_pipe_data.append(row)
    
    # Step 3: Combine the filtered MediaPipe data with manual data
    merged_data = media_pipe_data + manual_data
    
    # Step 4: Write the merged data to the output CSV
    with open(output_path, 'w', newline='') as output_file:
        fieldnames = ['input_file_name', 'face_number', 'face_rect_value', 'iris_left_value', 'iris_right_value']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_data:
            writer.writerow(row)
    
    print(f"Merge complete. Output written to {output_path}")
    print(f"Entries from MediaPipe: {len(media_pipe_data)}")
    print(f"Entries from Manual: {len(manual_data)}")
    print(f"Total entries in merged file: {len(merged_data)}")

# Run the merge
merge_csvs(MEDIA_PIPE_CSV, MANUAL_CSV, OUTPUT_CSV)