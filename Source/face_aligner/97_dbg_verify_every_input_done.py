import os

def check_matching_filenames(input_dir, output_dir):
    """
    Checks if any output filename starts with an input filename's stem (ignoring extension).
    Prints input filenames with no matches in output directory.
    
    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        return
    if not os.path.isdir(output_dir):
        print(f"Error: '{output_dir}' is not a valid directory.")
        return

    # Get all filenames from input directory
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    # Get all filenames from output directory
    output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

    # Check each input file against output files, ignoring extensions
    missing_matches = []
    for input_file in input_files:
        # Get the stem (filename without extension)
        input_stem = os.path.splitext(input_file)[0]
        # Check if any output file starts with the input stem
        has_match = any(output_file.startswith(input_stem) for output_file in output_files)
        if not has_match:
            missing_matches.append(input_file)

    # Print results
    if missing_matches:
        print(f"\nInput files with no matching prefix in '{output_dir}':")
        for file in missing_matches:
            print(f"  - {file}")
    else:
        print(f"All input files from '{input_dir}' have matches in '{output_dir}'.")


# Example usage
if __name__ == "__main__":
    check_matching_filenames("/Users/riyasyoosuf/Desktop/Input/Phase2_FaceVideos_CleanedUp/FaceVideos_CFR", "/Users/riyasyoosuf/Desktop/Input/Phase2_FaceVideos_CleanedUp/FaceVideos_CFR_Aligned")