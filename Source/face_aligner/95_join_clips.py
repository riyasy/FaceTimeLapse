import os
import ffmpeg

def normalize_clips(input_clips_dir="09_output_clips", normalized_dir="09_output_clips/normalized"):
    """Normalizes all video clips' audio format for smooth concatenation."""
    os.makedirs(normalized_dir, exist_ok=True)  # Ensure output directory exists
    
    clip_files = sorted([f for f in os.listdir(input_clips_dir) if f.lower().endswith(".mp4")])
    normalized_clips = []
    
    for clip in clip_files:
        input_path = os.path.join(input_clips_dir, clip)
        output_path = os.path.join(normalized_dir, clip)  # Fix: Correct output path
        
        # Normalize audio while keeping video as is
        ffmpeg.input(input_path).output(
            output_path,
            vcodec="copy",  # Keep video unchanged
            acodec="aac",   # Re-encode audio
            ar=44100,       # Set sample rate
            ab="192k",      # Set audio bitrate
            ac=2,           # Ensure stereo audio
            strict="experimental"
        ).run(overwrite_output=True)
        
        normalized_clips.append(output_path)  # Fix: Append correct path
    
    return normalized_clips, normalized_dir


def concatenate_videos(input_clips_dir="09_output_clips", output_file="final_video.mp4"):
    """Concatenates all normalized video clips into a final video."""
    # Step 1: Normalize clips
    normalized_clips, normalized_dir = normalize_clips(input_clips_dir)

    if not normalized_clips:
        print("No normalized clips found.")
        return

    # Step 2: Create a text file with correct paths for ffmpeg concat
    concat_list_path = os.path.join(normalized_dir, "file_list.txt")
    with open(concat_list_path, "w") as f:
        for clip in normalized_clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")  # Fix: Ensure absolute paths

    try:
        # Step 3: Concatenate normalized clips
        ffmpeg.input(concat_list_path, format="concat", safe=0).output(
            output_file,
            vcodec="libx264",  # Re-encode video for compatibility
            acodec="aac",      # Ensure audio consistency
            ar=44100,          # Consistent sample rate
            ab="192k",         # High-quality audio
            preset="fast",     # Speed vs. quality tradeoff
            fflags="+genpts"   # Regenerate timestamps for smooth transitions
        ).run(overwrite_output=True)
        print(f"Final video saved as {output_file}")
    except ffmpeg.Error as e:
        print(f"Error concatenating videos: {e}")
        print("FFmpeg stderr output:")
        print(e.stderr.decode())  # Show detailed error if it fails
    finally:
        os.remove(concat_list_path)  # Clean up temp file list


# Example usage
if __name__ == "__main__":
    concatenate_videos("09_output_clips", "09_output_clips/final_video.mp4")
