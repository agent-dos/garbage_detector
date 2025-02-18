import cv2
import numpy as np
import os
import time
import logging
import argparse

from garbage_detector import GarbageDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_video(detector, video_source, output_path, display=False):
    """
    Process a video file and write an annotated output video.

    Args:
        detector (GarbageDetector): An instance of GarbageDetector.
        video_source (str): Path to the input video file.
        output_path (str): Path to save the annotated output video.
        display (bool): If True, display each processed frame.
    """
    logger.info(f"Starting video processing: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Error opening video source: {video_source}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps < 1:
        fps = 30
        logger.warning(f"Invalid FPS detected, using default: {fps}")

    logger.info(f"Video properties: {detector.resize_dims[0]}x{detector.resize_dims[1]}, {fps} FPS, {total_frames} frames")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, detector.resize_dims)
    if not writer.isOpened():
        logger.error(f"Failed to create output video: {output_path}")
        cap.release()
        return 0

    processed_frames = 0
    start_time = time.time()
    last_log_time = start_time
    trash_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, detections, _ = detector.detect(frame)
        trash_counts.append(len(detections['trash']))

        writer.write(result_frame)
        processed_frames += 1
        current_time = time.time()

        # Log progress every 5 seconds
        if current_time - last_log_time >= 5:
            elapsed = current_time - start_time
            progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
            processing_fps = processed_frames / elapsed if elapsed > 0 else 0
            avg_trash = sum(trash_counts) / len(trash_counts)
            logger.info(f"Progress: {processed_frames}/{total_frames} frames ({progress:.1f}%) | "
                        f"Speed: {processing_fps:.1f} FPS | "
                        f"Avg Paper: {avg_trash:.1f} items/frame")
            last_log_time = current_time

        if display:
            cv2.imshow('Processing', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elapsed_time = time.time() - start_time
    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    logger.info(f"Video processing complete: {output_path}")
    logger.info(f"Processed {processed_frames} frames in {elapsed_time:.1f} seconds")
    logger.info(f"Average processing speed: {processed_frames / elapsed_time:.1f} FPS")
    logger.info(f"Avg Trash detected: {sum(trash_counts) / len(trash_counts):.1f} items/frame")

    return processed_frames


def process_folder(detector, input_folder, output_folder, display=False):
    """
    Process all images in an input folder and save annotated images to an output folder.

    Args:
        detector (GarbageDetector): An instance of GarbageDetector.
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        display (bool): If True, display each processed image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output folder: {output_folder}")

    allowed_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(allowed_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                logger.warning(f"Could not read image: {input_path}")
                continue

            result_image, detections, _ = detector.detect(image)
            cv2.imwrite(output_path, result_image)
            logger.info(f"Processed and saved: {output_path}")

            if display:
                cv2.imshow('Processed Image', result_image)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

    if display:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Garbage Detection: Process images or video for detection of paper and plastic wrappers."
    )
    parser.add_argument("--mode", choices=["video", "folder"], required=True,
                        help="Processing mode: 'video' for video processing, 'folder' for batch image processing.")
    parser.add_argument("--input", required=True, help="Path to the input video file or image folder.")
    parser.add_argument("--output", required=True, help="Path to the output video file or folder.")
    parser.add_argument("--display", action="store_true", help="Display the processing output in real-time.")
    args = parser.parse_args()

    detector = GarbageDetector()

    if args.mode == "video":
        process_video(detector, args.input, args.output, args.display)
    elif args.mode == "folder":
        process_folder(detector, args.input, args.output, args.display)


if __name__ == "__main__":
    main()
