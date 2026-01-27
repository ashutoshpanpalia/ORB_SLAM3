import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

# Create pipeline and config
pipeline = rs.pipeline()
config = rs.config()

# Generate timestamp filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
bag_filename = f"my_program/data_collection_pipeline/data/raw_bag_files/{timestamp}.bag"

print(f"Recording to: {bag_filename}")

# Enable streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Enable recording to bag file
config.enable_record_to_file(bag_filename)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())

        # Show live preview
        cv2.imshow("Recording... Press Q to Stop", color_image)

        # Stop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping recording...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Recording saved.")
