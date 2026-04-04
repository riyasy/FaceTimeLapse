import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
INPUT_DIR  = "00_input_videos"
OUTPUT_DIR = "01_extracted_frames"
MAX_WORKERS = 4  # Parallel threads; tune based on available CPU cores
# ---------------------

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")


def is_hdr(video_path):
    """Uses ffprobe to detect if a video uses an HDR color transfer function (HLG or PQ)."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=color_transfer",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        color_transfer = result.stdout.strip().lower()
        return color_transfer in ["smpte2084", "arib-std-b67"]
    except Exception as e:
        print(f"Warning: Failed to check HDR status for {video_path}: {e}")
        return False


def save_first_frame(video_path, output_dir):
    """
    Extracts the first frame using FFmpeg.
    FFmpeg automatically handles orientation metadata (fixing inverted frames)
    and we apply tone-mapping if the source is HDR (fixing washed-out colors).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}.jpg")

    hdr_detected = is_hdr(video_path)

    # Base ffmpeg command: extract 1 frame (-vframes 1) at high quality (-q:v 2)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vframes", "1",
        "-q:v", "2"
    ]

    # If HDR, inject the tone-mapping filter
    if hdr_detected:
        tone_map_filter = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
        cmd.extend(["-vf", tone_map_filter])

    cmd.append(output_path)

    try:
        # Attempt primary extraction
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return video_path, True
    except subprocess.CalledProcessError:
        # If standard HDR tone-mapping fails (e.g., missing zscale library in FFmpeg), try a fallback
        if hdr_detected:
            try:
                fallback_cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vframes", "1", "-q:v", "2",
                    "-vf", "colorspace=all=bt709:illuminate=d65:format=yuv420p",
                    output_path
                ]
                subprocess.run(fallback_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return video_path, True
            except subprocess.CalledProcessError:
                pass
        
        # Absolute fallback: Just extract the frame ignoring HDR adjustments
        try:
            basic_cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vframes", "1", "-q:v", "2", output_path
            ]
            subprocess.run(basic_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return video_path, True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame from {video_path}")
            return video_path, False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all video files from the input directory
    video_tasks = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    # Process videos in parallel and collect results
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(save_first_frame, path, OUTPUT_DIR): path
            for path in video_tasks
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                path, success = future.result()
                results[path] = success
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results[path] = False

    # Print summary
    print("\n=== Processing Summary ===")
    for path, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {os.path.basename(path)} - First frame saved: {success}")


if __name__ == "__main__":
    main()