import os
import shutil
import ffmpeg
from datetime import datetime

from util import clear_directory

# --- CONFIGURATION ---
INPUT_DIR         = "03_output_videos"
OUTPUT_CLIPS_DIR  = "04_output_clips"
FAILED_DIR        = "99_failed_videos"

CLIP_DURATION = 0.25  # Seconds to extract from the start of each video

# FFmpeg drawtext settings for the date overlay
FONT_FILE  = "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"
FONT_SIZE  = 200
FONT_COLOR = "white"
SHADOW_COLOR = "black"
SHADOW_OFFSET = 5     # Pixels of drop shadow on x and y
# ---------------------

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")


def extract_date_from_filename(filename):
    """
    Parses a date string from filenames with the pattern '20200604_165011…'.
    Returns a human-readable 'Month YYYY' string, or None if parsing fails.
    """
    try:
        date_str = filename.split("_")[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj.strftime("%B %Y")  # e.g. "June 2020"
    except Exception:
        return None  # Filename doesn't follow the expected date pattern


def build_drawtext_filter(date_text):
    """
    Builds an FFmpeg drawtext filter string that renders the date centred
    near the bottom of the frame with a drop shadow.
    """
    return (
        f"drawtext=text='{date_text}'"
        f":fontfile='{FONT_FILE}'"
        f":fontcolor={FONT_COLOR}"
        f":fontsize={FONT_SIZE}"
        f":x=(w-text_w)/2"       # Horizontally centred
        f":y=h-{FONT_SIZE}"      # Near the bottom edge
        f":shadowcolor={SHADOW_COLOR}"
        f":shadowx={SHADOW_OFFSET}"
        f":shadowy={SHADOW_OFFSET}"
    )


def main():
    clear_directory(OUTPUT_CLIPS_DIR)
    clear_directory(FAILED_DIR)

    # Collect and sort all video files in the input directory
    video_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )

    for i, filename in enumerate(video_files, start=1):
        input_path  = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_CLIPS_DIR, f"clip_{i:04d}.mp4")

        date_text = extract_date_from_filename(filename)

        try:
            if date_text:
                # Overlay the date text while extracting the clip
                (
                    ffmpeg
                    .input(input_path, ss=0, t=CLIP_DURATION)
                    .output(
                        output_path,
                        vf=build_drawtext_filter(date_text),
                        vcodec="libx264",
                        acodec="copy",
                        preset="ultrafast",  # Fast encode; quality controlled by crf
                        crf=18,
                    )
                    .run(overwrite_output=True, quiet=True)
                )
            else:
                # No date found — copy the clip without any text overlay
                (
                    ffmpeg
                    .input(input_path, ss=0, t=CLIP_DURATION)
                    .output(output_path, c="copy")
                    .run(overwrite_output=True, quiet=True)
                )
            print(f"✓ {filename} → {os.path.basename(output_path)}" + (f" [{date_text}]" if date_text else ""))
        except ffmpeg.Error as e:
            print(f"✗ Error processing {filename}: {e}")
            shutil.copy2(input_path, os.path.join(FAILED_DIR, filename))


if __name__ == "__main__":
    main()
