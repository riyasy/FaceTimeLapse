# Face Align Time-Lapse Pipeline

A modular Python pipeline that detects faces, aligns them around the eyes across hundreds of video/image clips, and concatenates them into a continuous time-lapse with blurred backgrounds.

## The Workflow

### 1. Preparation & Detection
- **`00_extract_first_frames.py`**
  Extracts the first frame of every video in the input folder. Its easier and faster to work on the image rather than the video.

- **`01_find_eyes_using_dlib.py`** or **`01_find_eyes_using_media_pipe_google_colab.py`**
  Automatically detects faces in the inputs (output of last step) using either DLIB or MediaPipe. Generates a master CSV mapping every face to the exact left and right eye pixel coordinates.
  (The web tool `02_review_annotation.html` can be used to review and correct these annotations in the next step.)

### 2. Manual Review & Correction
- **`02_review_annotation.html`**
  A local web-based tool. Open this in your browser and load the generated CSV along with the input folder containing images used in last step(using the File System Access API). 
  It allows you to rapidly cycle through every detected face, verify the eye coordinates and manually correct any mistakes by adjusting eye positions. Export the newly refined CSV to use for actual alignment.

### 3. Alignment & Transformation
- **`03_align_video_using_annotation_csv.py`** and **`03_align_image_using_annotation_csv.py`**
  The heavy lifters. They read the eye-coordinate CSV and process the original high-res videos/images:
  1. Detects rotation angle and scale to perfectly level the eyes at a consistent width (scaling dynamically based on the age/date of the clip).
  2. Creates a zoomed-in, tightly cropped **foreground** over a 2× zoomed, Gaussian blurred **background** (eliminating black borders for portrait/oddly cropped videos).
  3. Muxes original audio back into the final aligned clip.

### 4. Polish & Concatenation
- **`04_draw_date_over_video.py`**
  Extracts short slices (e.g., 0.25 seconds) from the beginning of each processed clip. Parses the date out of the filename (e.g. `20201225_...` to "December 2020") and overlays it aesthetically via FFmpeg drawtext.

- **`05_join_clips.py`**
  Normalises the audio sample rate and codecs across all short clips to prevent audio discontinuities or playback errors, then concatenates them losslessly into a single continuous final video. Sometimes audio issues are noticed when concatenating, in which case you can use a tool like imovie to join the clips.

---

## Utilities & Debugging

- **`util.py`**
  Shared utility functions.

- **`debug_01_verify_annotated_csv_with_input.py`**
  Checks the annotated CSV against the input folder to ensure every input file has an annotation, and every annotation corresponds to a real input file. Report if there are two or more faces per file in annotation CSV.

- **`debug_02_verify_every_input_done.py`**
  Sanity check: matches input filenames against output filenames to flag any files that the pipeline failed on or skipped.

- **`debug_03_clear_directories.py`**
  Utility to wipe all intermediate `output`, `failed`, and `debug` folders clean before running a fresh sequential batch.

## Required Setup
- Dependencies are found in `requirements.txt`
- DLIB processing requires placing `mmod_human_face_detector.dat` and `shape_predictor_68_face_landmarks.dat` into the root directory.
- MediaPipe pipeline requires `face_landmarker_v2_with_blendshapes.task`.
