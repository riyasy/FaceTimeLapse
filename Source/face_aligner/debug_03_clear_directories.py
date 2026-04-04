from util import clear_directory

# --- CONFIGURATION ---
# List every directory that should be wiped clean before a fresh processing run.
# Add or remove entries here as the pipeline evolves.
DIRECTORIES_TO_CLEAR = [
    "00_input_images",
    "00_input_videos",
    "01_extracted_frames",
    "03_output_videos",
    "04_output_clips",
    "05_final_video",
    "99_failed_images",
    "99_failed_videos",
    "99_debug_frames",
]
# ---------------------


def main():
    """Clears (or creates if missing) every directory in DIRECTORIES_TO_CLEAR."""
    print("Starting directory cleanup…")
    for directory in DIRECTORIES_TO_CLEAR:
        clear_directory(directory)
    print("Done — all directories processed.")


if __name__ == "__main__":
    main()
