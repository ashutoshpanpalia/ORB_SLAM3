#!/usr/bin/env python3
"""
Convert RealSense D435i bag file to ORB-SLAM3 compatible dataset
Creates RGB images, depth images, and timestamps file
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys


# ===== DEFAULT PATHS =====
DEFAULT_BAG_FILE = "my_program/data_collection_pipeline/data/raw_bag_files/2026-01-27_14-54-39.bag"
DEFAULT_OUTPUT_DIR = "my_program/data_collection_pipeline/data/processed_dataset"       


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

    print(f"[INFO] Using bag file: {bag_path}")
    print(f"[INFO] Output directory: {output_dir}")

    rgb_dir, depth_dir = create_directories(output_dir)

    rgb_timestamps_path = os.path.join(output_dir, "rgb.txt")
    depth_timestamps_path = os.path.join(output_dir, "depth.txt")

    try:
        pipeline = rs.pipeline()
        config = rs.config()

        print(f"[INFO] Loading bag file...")
        config.enable_device_from_file(bag_path, repeat_playback=False)

        profile = pipeline.start(config)

        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        duration = playback.get_duration().total_seconds()
        print(f"[INFO] Bag duration: {duration:.2f} seconds")

        align_to_color = rs.align(rs.stream.color)

        rgb_ts_file = open(rgb_timestamps_path, "w")
        depth_ts_file = open(depth_timestamps_path, "w")

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

                filename = f"{frame_count:06d}.png"

                rgb_path = os.path.join(rgb_dir, filename)
                depth_path = os.path.join(depth_dir, filename)

                cv2.imwrite(rgb_path, color_image_rgb)
                cv2.imwrite(depth_path, depth_image)

                rgb_ts_file.write(f"{timestamp:.6f}\n")
                depth_ts_file.write(f"{timestamp:.6f}\n")

                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"[PROGRESS] Extracted {frame_count} frames")

            except RuntimeError as e:
                if "Frame didn't arrive" in str(e):
                    print("[INFO] End of bag file reached")
                    break
                else:
                    print(f"[WARNING] Runtime error: {e}")
                    break

        rgb_ts_file.close()
        depth_ts_file.close()
        pipeline.stop()

        print("\n========================================")
        print("Extraction Complete!")
        print(f"Total frames: {frame_count}")
        print(f"RGB images: {rgb_dir}")
        print(f"Depth images: {depth_dir}")
        print(f"Timestamps: {rgb_timestamps_path}")
        print("========================================")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to process bag file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Allow default paths if no args provided
    bag_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BAG_FILE
    base_output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    # Append bag filename (without extension) to output dir
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    output_dir = os.path.join(base_output_dir, bag_name)

    print("[INFO] Running with:")
    print(f"       Bag file: {bag_path}")
    print(f"       Output dir: {output_dir}")

    success = extract_bag_to_dataset(bag_path, output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
