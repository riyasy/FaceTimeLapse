import os
import csv

# --- CONFIGURATION ---
INPUT_DIR = "00_input_videos"
CSV_PATH  = "02_annotated_csv/annotated_eye_coordinates.csv"
# ---------------------

VALID_EXTENSIONS = (".mp4", ".mov", ".avi", ".jpg", ".jpeg", ".png", ".bmp")


def verify_csv_against_input(input_dir, csv_path):
    """
    Checks the annotated CSV against the input folder to ensure every input file
    has an annotation, and every annotation corresponds to a real input file.

    File extensions are ignored during the comparison to support the workflow
    where annotations are generated from JPEG first-frames but the final
    rendering is done from the original MP4/MOV videos.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    # 1. Collect all valid input file stems
    input_stems = set()
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(VALID_EXTENSIONS):
            stem = os.path.splitext(filename)[0]
            input_stems.add(stem)

    # 2. Collect all file stems from the CSV
    csv_stem_counts = {}
    csv_row_count = 0
    try:
        with open(csv_path, newline="") as csvfile:
            for row in csv.DictReader(csvfile):
                # Ensure the CSV actually has this column
                if "input_file_name" in row:
                    stem = os.path.splitext(row["input_file_name"])[0]
                    csv_stem_counts[stem] = csv_stem_counts.get(stem, 0) + 1
                    csv_row_count += 1
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    csv_stems = set(csv_stem_counts.keys())

    # 3. Compare Both Sides
    missing_in_csv   = sorted(input_stems - csv_stems)
    missing_in_input = sorted(csv_stems - input_stems)

    # 4. Print Report
    print("\n=== VERIFICATION REPORT ===")
    print(f"Total valid input files:  {len(input_stems)}")
    print(f"Total CSV rows/faces:     {csv_row_count} (spanning {len(csv_stems)} unique files)")
    print("-" * 30)

    # 4b. Find multi-face files
    multi_face_files = {stem: count for stem, count in csv_stem_counts.items() if count > 1}
    if multi_face_files:
        print(f"\n[i] MULTI-FACE ANNOTATIONS ({len(multi_face_files)} files):")
        print("    These files have multiple faces annotated in the CSV.")
        for stem, count in sorted(multi_face_files.items()):
            print(f"      - {stem}: {count} faces")
        print("-" * 30)
    
    if not missing_in_csv and not missing_in_input:
        print("✓ Perfect Match! All input files are annotated, and no orphan annotations exist.")
        return

    if missing_in_csv:
        print(f"\n[✗] MISSING IN CSV ({len(missing_in_csv)} files):")
        print("    These files exist in the input folder but have no annotations in the CSV.")
        for stem in missing_in_csv:
            print(f"      - {stem}")

    if missing_in_input:
        print(f"\n[✗] MISSING IN INPUT FOLDER ({len(missing_in_input)} files):")
        print("    These files are annotated in the CSV but do not exist in the input folder.")
        for stem in missing_in_input:
            print(f"      - {stem}")


def main():
    verify_csv_against_input(INPUT_DIR, CSV_PATH)


if __name__ == "__main__":
    main()
