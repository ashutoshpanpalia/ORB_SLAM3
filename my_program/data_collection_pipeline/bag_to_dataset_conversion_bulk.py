#!/usr/bin/env python3
"""
Convert RealSense D435i bag files to ORB-SLAM3 compatible datasets
Creates RGB images, depth images, and timestamps files

DEFAULT MODE:
- Reads all .bag files from: my_program/data_processing/data/raw_bag_files/
- Creates output folders per bag file inside: my_program/data_processing/data/
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
from pathlib import Path

DEFAULT_BAG_DIR = "my_program/data_collection_pipeline/data/raw_bag_files/"
DEFAULT_OUTPUT_ROOT = "my_program/data_collection_pipeline/data/processed_dataset"


def create_directories(output_dir):
    """Create output directory structure"""
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    return rgb_dir, depth_dir


def extract_bag_to_dataset(bag_path, output_dir):
    """Extract RGB and depth frames from bag file to dataset format"""

    if not os.path.exists(bag_path):
        print(f"[ERROR] Bag file not found: {bag_path}")
        return False

    print(f"\n[INFO] Processing bag: {bag_path}")
    print(f"[INFO] Output folder: {output_dir}")

    rgb_dir, depth_dir = create_directories(output_dir)

    rgb_timestamps_path = os.path.join(output_dir, 'rgb.txt')
    depth_timestamps_path = os.path.join(output_dir, 'depth.txt')

    try:
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device_from_file(bag_path, repeat_playback=False)

        profile = pipeline.start(config)

        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        duration = playback.get_duration().total_seconds()
        print(f"[INFO] Bag duration: {duration:.2f} seconds")

        align_to_color = rs.align(rs.stream.color)

        rgb_ts_file = open(rgb_timestamps_path, 'w')
        depth_ts_file = open(depth_timestamps_path, 'w')

        frame_count = 0
        print("[INFO] Extracting frames...")

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = align_to_color.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                timestamp = frames.get_timestamp() / 1000.0

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                frame_filename = f"{frame_count:06d}.png"

                rgb_path = os.path.join(rgb_dir, frame_filename)
                depth_path = os.path.join(depth_dir, frame_filename)

                cv2.imwrite(rgb_path, color_image_rgb)
                cv2.imwrite(depth_path, depth_image)

                rgb_ts_file.write(f"{timestamp:.6f}\n")
                depth_ts_file.write(f"{timestamp:.6f}\n")

                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"[PROGRESS] Extracted {frame_count} frames")

            except RuntimeError as e:
                if "Frame didn't arrive" in str(e):
                    print("[INFO] End of bag reached")
                    break
                else:
                    print(f"[WARNING] Runtime error: {e}")
                    break

        rgb_ts_file.close()
        depth_ts_file.close()
        pipeline.stop()

        print("========================================")
        print("Extraction Complete!")
        print(f"Total frames: {frame_count}")
        print(f"RGB folder: {rgb_dir}")
        print(f"Depth folder: {depth_dir}")
        print("========================================")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to process bag file: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_all_bags():
    """Process all bag files in default folder"""

    bag_dir = Path(DEFAULT_BAG_DIR)

    if not bag_dir.exists():
        print(f"[ERROR] Default bag folder not found: {bag_dir}")
        return False

    bag_files = list(bag_dir.glob("*.bag"))

    if not bag_files:
        print(f"[WARNING] No .bag files found in: {bag_dir}")
        return False

    print(f"[INFO] Found {len(bag_files)} bag files")

    for bag_file in bag_files:
        bag_name = bag_file.stem
        output_dir = Path(DEFAULT_OUTPUT_ROOT) / bag_name

        extract_bag_to_dataset(str(bag_file), str(output_dir))

    return True


def main():
    # If user provides args, use single-file mode
    if len(sys.argv) == 3:
        bag_path = sys.argv[1]
        output_dir = sys.argv[2]
        return 0 if extract_bag_to_dataset(bag_path, output_dir) else 1

    # Otherwise run default batch mode
    print("[INFO] Running in DEFAULT BATCH MODE")
    print(f"[INFO] Input folder: {DEFAULT_BAG_DIR}")
    print(f"[INFO] Output root: {DEFAULT_OUTPUT_ROOT}")

    success = process_all_bags()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
