import os

# --- CONFIGURATION ---
INPUT_DIR  = "00_input_videos"
OUTPUT_DIR = "03_output_videos"
# ---------------------


def check_matching_filenames(input_dir, output_dir):
    """
    Verifies that every file in input_dir has at least one corresponding file
    in output_dir whose name starts with the same stem (ignoring extension).

    This is useful after processing to confirm no input was accidentally skipped.
    Multi-face outputs (e.g. 'clip_face01.mp4', 'clip_face02.mp4') are all
    caught by the prefix match, so one input can map to many outputs correctly.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        return
    if not os.path.isdir(output_dir):
        print(f"Error: '{output_dir}' is not a valid directory.")
        return

    input_files  = [f for f in os.listdir(input_dir)  if os.path.isfile(os.path.join(input_dir,  f))]
    output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

    # For each input, check whether any output filename begins with the input's stem
    missing = [
        f for f in input_files
        if not any(out.startswith(os.path.splitext(f)[0]) for out in output_files)
    ]

    if missing:
        print(f"\nInput files with no matching output in '{output_dir}':")
        for f in missing:
            print(f"  ✗ {f}")
    else:
        print(f"✓ All {len(input_files)} input files have matching outputs in '{output_dir}'.")


def main():
    check_matching_filenames(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()