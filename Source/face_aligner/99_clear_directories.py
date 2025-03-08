import os
import shutil

from util import clear_directory


# List of directories to clear
directories = [
    "00_input_images",
    "01_input_videos_vfr", 
    "02_input_videos_cfr",
    "07_output_images",
    #"08_output_videos",
    "09_output_clips",
    "10_final_video",
    "91_failed_images",
    "92_failed_videos",
    "99_debug_frames",
]


# Clear all directories
def clear_all_directories():
    print("Starting directory cleanup process...")
    for directory in directories:
        clear_directory(directory)
    print("All directories have been processed.")


# Run the cleanup
if __name__ == "__main__":
    clear_all_directories()
