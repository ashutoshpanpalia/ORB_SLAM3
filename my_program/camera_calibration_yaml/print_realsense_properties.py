import pyrealsense2 as rs
import numpy as np

# -----------------------------
# Initialize pipeline and config
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

# Start pipeline
profile = pipeline.start(config)

try:
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    print("\n=== DEVICE INFO ===")
    print("Name:", device.get_info(rs.camera_info.name))
    print("Serial:", device.get_info(rs.camera_info.serial_number))
    print("Firmware:", device.get_info(rs.camera_info.firmware_version))

    # -------------------------
    # COLOR INTRINSICS
    # -------------------------
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()

    print("\n=== COLOR INTRINSICS ===")
    print("Resolution:", color_intr.width, "x", color_intr.height)
    print("fx:", color_intr.fx)
    print("fy:", color_intr.fy)
    print("cx:", color_intr.ppx)
    print("cy:", color_intr.ppy)
    print("Distortion model:", color_intr.model)
    print("Distortion coeffs:", list(color_intr.coeffs))
    print("FPS:", color_profile.fps())

    # -------------------------
    # DEPTH INTRINSICS
    # -------------------------
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_profile.get_intrinsics()

    print("\n=== DEPTH INTRINSICS ===")
    print("Resolution:", depth_intr.width, "x", depth_intr.height)
    print("fx:", depth_intr.fx)
    print("fy:", depth_intr.fy)
    print("cx:", depth_intr.ppx)
    print("cy:", depth_intr.ppy)
    print("Distortion model:", depth_intr.model)
    print("Distortion coeffs:", list(depth_intr.coeffs))
    print("FPS:", depth_profile.fps())

    # -------------------------
    # STEREO BASELINE (from IR extrinsics)
    # -------------------------
    ir1 = profile.get_stream(rs.stream.infrared, 1)
    ir2 = profile.get_stream(rs.stream.infrared, 2)

    extrinsics = ir1.get_extrinsics_to(ir2)
    baseline = abs(extrinsics.translation[0])
    print("\n=== STEREO PARAMS ===")
    print("Baseline (meters):", baseline)

    # -------------------------
    # DEPTH MAP FACTOR
    # -------------------------
    depth_scale = depth_sensor.get_depth_scale()
    depth_map_factor = 1.0 / depth_scale
    print("\n=== DEPTH PARAMS ===")
    print("Depth scale (meters per unit):", depth_scale)
    print("Depth map factor (for YAML):", depth_map_factor)

finally:
    pipeline.stop()
