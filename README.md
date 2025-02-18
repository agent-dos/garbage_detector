# Garbage Detector

## Overview
This project implements a **Garbage Detector** that processes images and videos to identify trash, specifically **paper and plastic wrappers**. The detection is based on color segmentation and contour filtering techniques using **OpenCV** and **NumPy**.

## Features
- Detects garbage (paper and plastic) in images and videos
- Optimized preprocessing and morphology operations
- Batch processing of images from a folder
- Video processing with real-time progress tracking
- Optional display of processing results

## Requirements
Make sure you have the following dependencies installed:

```sh
pip install opencv-python numpy
```

## Files Description

### `garbage_detector.py`
Contains the **GarbageDetector** class, which handles:
- Image preprocessing (grayscale conversion, noise reduction, adaptive thresholding)
- Robust color masking for paper and plastic detection
- Contour filtering to refine detected garbage regions
- Annotation of detected trash with bounding boxes

### `gd_cli.py`
Provides a command-line interface (CLI) to process images and videos. It includes:
- **process_video**: Processes video files, applies detection, and saves the output video
- **process_folder**: Processes all images in a folder and saves annotated images
- CLI argument parsing for mode selection (video or image folder processing)

## How to Run

### 1. Process Images from a Folder
Run the following command to process all images in a folder:
```sh
python gd_cli.py --mode folder --input path/to/images --output path/to/output_folder
```

### 2. Process a Video
To process a video and save the annotated output:
```sh
python gd_cli.py --mode video --input path/to/input_video.mp4 --output path/to/output_video.mp4
```

### 3. Display Processing Output in Real-time
Use the `--display` flag to visualize the processing while it runs:
```sh
python gd_cli.py --mode folder --input path/to/images --output path/to/output_folder --display
```
```sh
python gd_cli.py --mode video --input path/to/input_video.mp4 --output path/to/output_video.mp4 --display
```

## Example Usage
### Processing a folder of images:
```sh
python gd_cli.py --mode folder --input ./test_images --output ./processed_images
```

### Processing a video:
```sh
python gd_cli.py --mode video --input ./test_video.mp4 --output ./processed_video.mp4
```

## Notes
- The program supports **JPG, PNG, BMP, and TIFF** image formats.
- Press `q` to stop real-time display during processing.
- If an invalid FPS is detected in a video, the program defaults to 30 FPS.

## Logging
The CLI provides real-time logging, including:
- Number of frames processed
- Processing speed in FPS
- Average number of detected trash items per frame

## Author
- **Your Name**
- **Your Contact (Optional)**

Happy Coding! 🚀