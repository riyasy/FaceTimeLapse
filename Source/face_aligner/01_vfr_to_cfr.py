import os
import subprocess
import shutil
from util import clear_directory


def convert_to_constant_fps(input_dir, output_dir, failed_dir, target_fps=24):
    """Converts all videos in input_dir to constant FPS and saves them in output_dir."""
    # Clear and prepare directories
    clear_directory(output_dir)
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)

    for video in sorted(os.listdir(input_dir)):
        if not video.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        input_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, video)
        failed_path = os.path.join(failed_dir, video)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-filter:v",
            f"fps={target_fps}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            output_path,
        ]

        try:
            subprocess.run(
                ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(f"Successfully converted: {video}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video}: {e}")
            os.rename(input_path, failed_path)


# Run the conversion
convert_to_constant_fps(
    "01_input_videos_vfr", "02_input_videos_cfr", "92_failed_videos"
)
