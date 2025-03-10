import pandas as pd

# Read the text file and parse the filename and face_id
def parse_text_file(text_file_path):
    matches = set()
    with open(text_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split the string to get filename and face_id
                # Find the last occurrence of '_face' to handle '_Edit' cases
                last_face_idx = line.rfind('_face')
                if last_face_idx != -1:
                    filename = line[:last_face_idx]
                    face_id = line[last_face_idx + 5:]  # Get digits after '_face'
                    matches.add((filename, int(face_id)))
    return matches

# Main processing function
def filter_csv_inplace(csv_file_path, text_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get original row count
    original_rows = len(df)
    
    # Parse the text file
    matches = parse_text_file(text_file_path)
    
    # Create a combined key for matching
    df['match_key'] = df.apply(lambda row: (row['input_file_name'], row['face_number']), axis=1)
    
    # Filter out rows where the combination exists in the text file
    filtered_df = df[~df['match_key'].isin(matches)]
    
    # Drop the temporary match_key column
    filtered_df = filtered_df.drop('match_key', axis=1)
    
    # Overwrite the original CSV file
    filtered_df.to_csv(csv_file_path, index=False)
    
    print(f"Modified CSV file: {csv_file_path}")
    print(f"Original rows: {original_rows}, Remaining rows: {len(filtered_df)}")
    print(f"Rows removed: {original_rows - len(filtered_df)}")

# Example usage
if __name__ == "__main__":
    csv_file = "80_media_pipe_data/media_pipe_output_merged.csv"    # Replace with your CSV file path
    text_file = "80_media_pipe_data/toBeRemovedFaces.txt"   # Replace with your text file path
    
    filter_csv_inplace(csv_file, text_file)