import os
import ffmpeg

# --- CONFIGURATION ---
INPUT_CLIPS_DIR = "04_output_clips"
OUTPUT_FILE     = "05_final_video/final_video.mp4"

# Audio normalisation settings applied during the clip-normalisation pass
AUDIO_SAMPLE_RATE = 44100   # Hz
AUDIO_BITRATE     = "192k"
AUDIO_CHANNELS    = 2       # Stereo
# ---------------------


def normalize_clips(input_clips_dir, normalized_dir):
    """
    Re-encodes every clip's audio to a consistent format (AAC, 44.1 kHz, stereo)
    while keeping the video stream unchanged. This prevents audio discontinuities
    in the final concatenated video.
    Returns a list of paths to the normalised clips.
    """
    os.makedirs(normalized_dir, exist_ok=True)

    clip_files = sorted(f for f in os.listdir(input_clips_dir) if f.lower().endswith(".mp4"))
    normalized_clips = []

    for clip in clip_files:
        input_path  = os.path.join(input_clips_dir, clip)
        output_path = os.path.join(normalized_dir, clip)

        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vcodec="copy",               # Pass video through unmodified
                acodec="aac",
                ar=AUDIO_SAMPLE_RATE,
                ab=AUDIO_BITRATE,
                ac=AUDIO_CHANNELS,
                strict="experimental",
            )
            .run(overwrite_output=True, quiet=True)
        )
        normalized_clips.append(output_path)
        print(f"Normalised: {clip}")

    return normalized_clips


def concatenate_videos(input_clips_dir, output_file):
    """
    Normalises all clips in input_clips_dir, then concatenates them into a
    single output video using FFmpeg's concat demuxer.

    Steps:
      1. Normalise audio of each clip into a temporary sub-folder.
      2. Write a text file listing all normalised clips (required by ffmpeg concat).
      3. Concatenate and re-encode to a single output file.
      4. Clean up the temporary file list.
    """
    normalized_dir   = os.path.join(input_clips_dir, "normalized")
    normalized_clips = normalize_clips(input_clips_dir, normalized_dir)

    if not normalized_clips:
        print("No normalised clips found. Exiting.")
        return

    # Write the ffmpeg concat file — each line must be: file '/absolute/path/to/clip.mp4'
    concat_list_path = os.path.join(normalized_dir, "file_list.txt")
    with open(concat_list_path, "w") as f:
        for clip in normalized_clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    try:
        (
            ffmpeg
            .input(concat_list_path, format="concat", safe=0)
            .output(
                output_file,
                vcodec="libx264",
                acodec="aac",
                ar=AUDIO_SAMPLE_RATE,
                ab=AUDIO_BITRATE,
                preset="fast",       # Balance between encode speed and file size
                fflags="+genpts",    # Regenerate PTS to avoid timestamp gaps at clip boundaries
            )
            .run(overwrite_output=True)
        )
        print(f"\nFinal video saved: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error concatenating videos: {e}")
        print(e.stderr.decode())
    finally:
        # Remove the temporary file list regardless of success or failure
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)


def main():
    concatenate_videos(INPUT_CLIPS_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()
