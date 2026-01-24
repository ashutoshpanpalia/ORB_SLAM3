#!/usr/bin/env python3
"""
Record RealSense D435i to bag file and optionally convert to ORB-SLAM3 dataset
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
from datetime import datetime
from pathlib import Path


def record_bag_file(output_dir="my_program/data_processing", duration=None):
    """
    Record from RealSense camera to bag file
    
    Args:
        output_dir: Directory to save bag file
        duration: Optional recording duration in seconds (None = manual stop with 'q')
    
    Returns:
        Path to recorded bag file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bag_filename = os.path.join(output_dir, f"{timestamp}.bag")
    
    print("========================================")
    print("RealSense Recording")
    print("========================================")
    print(f"Recording to: {bag_filename}")
    if duration:
        print(f"Duration: {duration} seconds")
    else:
        print("Press 'q' to stop recording")
    print("========================================\n")
    
    # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Enable recording to bag file
    config.enable_record_to_file(bag_filename)
    
    # Start streaming
    pipeline.start(config)
    
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            
            # Add recording indicator
            cv2.putText(color_image, "REC", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame count
            cv2.putText(color_image, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show live preview
            cv2.imshow("Recording... Press Q to Stop", color_image)
            
            frame_count += 1
            
            # Check duration-based stop
            if duration:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration:
                    print(f"\n[INFO] Reached {duration} second duration, stopping...")
                    break
            
            # Stop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User stopped recording...")
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print("\n========================================")
    print("Recording Complete")
    print("========================================")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count/elapsed_time:.2f}")
    print(f"File saved: {bag_filename}")
    print("========================================\n")
    
    return bag_filename


def convert_bag_to_dataset(bag_path, output_dir=None):
    """
    Convert bag file to ORB-SLAM3 dataset format
    
    Args:
        bag_path: Path to bag file
        output_dir: Output directory (default: same name as bag without extension)
    """
    
    if output_dir is None:
        # Create output directory with same name as bag file
        output_dir = os.path.splitext(bag_path)[0] + "_dataset"
    
    print("[INFO] Converting bag to dataset...")
    print(f"[INFO] Input: {bag_path}")
    print(f"[INFO] Output: {output_dir}")
    
    # Create output directories
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    try:
        # Configure pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_path, repeat_playback=False)
        
        # Start pipeline
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        
        # Create alignment object
        align_to_color = rs.align(rs.stream.color)
        
        # Open timestamp files
        rgb_ts_file = open(os.path.join(output_dir, 'rgb.txt'), 'w')
        depth_ts_file = open(os.path.join(output_dir, 'depth.txt'), 'w')
        
        frame_count = 0
        
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
                
                # Convert BGR to RGB
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Save with zero-padded filenames
                frame_filename = f"{frame_count:06d}.png"
                cv2.imwrite(os.path.join(rgb_dir, frame_filename), color_image_rgb)
                cv2.imwrite(os.path.join(depth_dir, frame_filename), depth_image)
                
                rgb_ts_file.write(f"{timestamp:.6f}\n")
                depth_ts_file.write(f"{timestamp:.6f}\n")
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"[PROGRESS] Converted {frame_count} frames")
                    
            except RuntimeError:
                break
        
        rgb_ts_file.close()
        depth_ts_file.close()
        pipeline.stop()
        
        print(f"\n[SUCCESS] Converted {frame_count} frames to {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Record RealSense bag and convert to dataset')
    parser.add_argument('--duration', type=int, help='Recording duration in seconds (optional)')
    parser.add_argument('--no-convert', action='store_true', help='Skip dataset conversion')
    parser.add_argument('--output-dir', default='my_program/data_processing', 
                       help='Output directory for bag files')
    
    args = parser.parse_args()
    
    # Record bag file
    bag_path = record_bag_file(output_dir=args.output_dir, duration=args.duration)
    
    # Convert to dataset
    if not args.no_convert:
        print("\n" + "="*40)
        dataset_path = convert_bag_to_dataset(bag_path)
        
        if dataset_path:
            print("\n" + "="*40)
            print("Ready for ORB-SLAM3!")
            print("="*40)
            print("Run:")
            print(f"./Examples/RGB-D/rgbd_dataset \\")
            print(f"    ./Vocabulary/ORBvoc.txt \\")
            print(f"    ./Examples/RGB-D/RealSense_D435i.yaml \\")
            print(f"    {dataset_path}")
            print("="*40)
    else:
        print("\n[INFO] Skipped dataset conversion")
        print(f"[INFO] To convert later, run:")
        print(f"python bag_to_dataset.py {bag_path} <output_dir>")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())