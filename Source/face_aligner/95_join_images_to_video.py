import os
import ffmpeg

def images_to_video(input_images_dir="images", output_file="output_video.mp4", frames_per_image=60, frame_rate=30):
    """
    Converts a series of JPG/JPEG images into a video with configurable frames per image.
    
    Args:
        input_images_dir (str): Directory containing the input images
        output_file (str): Path for the output video file
        frames_per_image (int): Number of frames each image should appear in the video
        frame_rate (int): Frame rate of the output video (frames per second)
    """
    # Create temporary directory for processed images
    temp_dir = os.path.join(input_images_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Get list of image files
    image_files = sorted([f for f in os.listdir(input_images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg'))])
    
    if not image_files:
        print("No JPG/JPEG images found in the specified directory.")
        return

    # Calculate duration in seconds for concat file (frames / frame_rate)
    duration_per_image = frames_per_image / frame_rate

    # Prepare images for video
    processed_images = []
    for img in image_files:
        input_path = os.path.join(input_images_dir, img)
        output_path = os.path.join(temp_dir, f"processed_{img}")
        
        try:
            # Copy image with same format
            ffmpeg.input(input_path).output(
                output_path,
                vcodec='copy'
            ).run(overwrite_output=True, quiet=True)
            processed_images.append(output_path)
        except ffmpeg.Error as e:
            print(f"Error processing image {img}: {e}")
            continue

    # Create a text file for ffmpeg concat
    concat_list_path = os.path.join(temp_dir, "image_list.txt")
    with open(concat_list_path, "w") as f:
        for img_path in processed_images:
            f.write(f"file '{os.path.abspath(img_path)}'\n")
            f.write(f"duration {duration_per_image}\n")

    try:
        # Convert images to video with precise frame control
        stream = ffmpeg.input(concat_list_path, format='concat', safe=0)
        stream = stream.output(
            output_file,
            vcodec='libx264',    # H.264 video codec
            pix_fmt='yuv420p',   # Pixel format for broad compatibility
            r=frame_rate,        # Set exact frame rate
            preset='fast',       # Faster encoding
            vf='fps=fps={}'.format(frame_rate),  # Ensure consistent frame rate
            fflags='+genpts+igndts'  # Regenerate timestamps and ignore DTS
        )
        stream.run(overwrite_output=True, quiet=True)
        print(f"Video successfully created: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error creating video: {e}")
        print("FFmpeg stderr output:")
        print(e.stderr.decode())
    finally:
        # Clean up temporary files
        os.remove(concat_list_path)
        for img_path in processed_images:
            try:
                os.remove(img_path)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

# Example usage
if __name__ == "__main__":
    # Convert images to video, each image displayed for 30 frames at 30 fps (1 second)
    images_to_video(
        input_images_dir="99_debug_frames",
        output_file="combined_video.mp4",
        frames_per_image=6,    # 30 frames per image
        frame_rate=30          # 30 fps
    )