import pyrealsense2 as rs
import numpy as np
import cv2

# Create pipeline
pipeline = rs.pipeline()
config = rs.config()

# Path to your .bag file
bag_file = "recording.bag"

# Enable playback from bag file
config.enable_device_from_file(bag_file, repeat_playback=False)

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get playback device
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False)  # Process frame-by-frame instead of real-time

try:
    while True:
        # Wait for next frame set
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Display image
        cv2.imshow("RealSense Bag Playback", color_image)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

except RuntimeError:
    print("Playback finished.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
