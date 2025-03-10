## Using DLIB

### 03_align_vid_dlib_mmod_fixedResolution_blurBG.py
Takes a folder with videos as input. 
Reads each frames using opencv. 
Takes the first frame and finds eye position (outer eye corners) using dlib detection and predictor.
Does the affine transform to place eye in the center of video and a particular ratio of the screen width.
Applies the same transform for the rest of images and creates a temp video.
Uses ffmpeg to add audio from original audtio back to temp video and create final transformed video.

### 03_align_vid_dlib_mmod_fixedResolution_blurBG_optimized.py
Scales the first frame by half before giving to dlib for speeding up and also helps to ignore small insignificant faces which helps in speeding up things.

### 03_align_vid_dlib_mmod_fixedResolution_blurBG_optimized_threaded_eyeOuter.py
Adds multi threading capability 

### 03_align_vid_dlib_mmod_fixedResolution_blurBG_optimized_threaded_eyeCenter.py
Uses eye center point instead of outer point

### 02_align_img_dlib_mmod_fixedResolution_blurBG.py
Takes a folder of images as input and transforms each image using eye outer corner and then converts to a static video of specified duration. (Please note it uses eye outer.. So results may not match with eye center aligned previous script)

### 04_extract_clips.py
Extracts specified duration from already created videos.

## Using Media Pipe.

## 80_colab_landmark_using_media_pipe.py
Put all images and first frames of videos in an input folder. To get first frames of all videos (use 96_dbg_extractFirstFrames.py)
Aligns using media pipe. Creates an output folder with the eyes and lips marked.
Also outputs a csv with all detected faces rect and iris centers annotated.
Removes extensions from file names in the csv.

## VIA tool (download from online)
Verify the output images and find files for which the iris center is not correct. For them use the (VIA) "VGG Image Annotator" tool to annotate the left eye center (point 1) and right eye center (point 2).

## 81_convert_manual_annot_csv_to_media_pipe_csv_format.py
Export the csv from the VIA and convert to a format similar to one produced by media-pipe using script 

## 82_merge_media_pipe_csv_and_manual_annotate_csv.py
Merge the manual and automatic csv using this script.

Now we have a comprehensive csv with all filenames (photo and video) and position of left iris and right iris inside it.

## 83_align_image_using_mediapipe_csv.py
## 83_align_video_using_mediapipe_csv.py
Create face aligned videos from both input videos and images using these two scripts.

## 84_cleanup_media_pipe_csv.py
Verify all the outputs and find names of files which we dont need. Collect all such filenames without extension and put it in a text file. We can remove these entries from the csv so that, in case of further regeneration, we dont need to do manual verification and cleanup again.

## Clip merging
FFMPEG muxing corrupts audio. Need to find reason why.
Use imovie or other editors to combine all the video clips to a single file. 

