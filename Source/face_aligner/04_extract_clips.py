import os
import ffmpeg
from datetime import datetime
import shutil

from util import clear_directory

# Configurable constants
CLIP_DURATION = 0.50  # Duration of each extracted clip in seconds


def extract_date_from_filename(filename):
    """Extracts and formats the date from filenames like '20200604_165011'."""
    try:
        date_str = filename.split("_")[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj.strftime("%d %B %Y")
    except Exception:
        return None


def extract_clips(input_dir, output_clips_dir, failed_dir):
    """Extracts the first `CLIP_DURATION` seconds from each video and overlays date text."""

    clear_directory(output_clips_dir)
    clear_directory(failed_dir)

    video_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
        ]
    )

    for i, filename in enumerate(video_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_clips_dir, f"clip_{i:04d}.mp4")

        try:
            date_text = extract_date_from_filename(filename)
            if date_text:
                draw_text = (
                    f"drawtext=text='{date_text}':fontfile='/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf':"
                    f"fontcolor=white:fontsize=200:x=(w-text_w)/2:y=h-200:"
                    f"shadowcolor=black:shadowx=5:shadowy=5"
                )
                ffmpeg.input(input_path, ss=0, t=CLIP_DURATION).output(
                    output_path,
                    vf=draw_text,
                    vcodec="libx264",
                    acodec="copy",
                    preset="slow",
                    crf=18,
                ).run(overwrite_output=True, quiet=True)
            else:
                ffmpeg.input(input_path, ss=0, t=CLIP_DURATION).output(
                    output_path, c="copy"
                ).run(overwrite_output=True, quiet=True)
        except ffmpeg.Error as e:
            print(f"Error processing {filename}: {e}")
            shutil.copy2(input_path, os.path.join(failed_dir, filename))


# Entry point
if __name__ == "__main__":
    extract_clips("08_output_videos", "09_output_clips", "92_failed_videos")
