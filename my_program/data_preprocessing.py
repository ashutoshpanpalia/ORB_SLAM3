import os
import cv2
import numpy as np
from pathlib import Path
import math

# ============================================================
# ======================= CONFIG =============================
# ============================================================

RAW_ROOT = Path("my_program/data_collection_pipeline/data/processed_dataset")
OUT_ROOT = Path("my_program/data_collection_pipeline/data/processed_dataset/npz_episodes")

USE_DEPTH = True

# Image preprocessing
IMG_SIZE = (128, 128)          # (W, H)
DEPTH_CLIP_MAX = 2.0           # meters
DEPTH_SCALE = 0.001            # mm → meters

MIN_EPISODE_LENGTH = 30

# Action normalization (per-DoF, SE(3))
MAX_ACTION = np.array([
    0.01,                      # dx (m)
    0.01,                      # dy (m)
    0.005,                     # dz (m)
    np.deg2rad(2.0),           # droll
    np.deg2rad(2.0),           # dpitch
    np.deg2rad(5.0),           # dyaw
])

# Camera → EE transform (replace when calibrated)
T_CAM2EE = np.eye(4)

# ============================================================
# ======================= HELPERS =============================
# ============================================================

def sorted_images(folder):
    return sorted([p for p in folder.iterdir() if p.suffix in [".png", ".jpg"]])

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ])

def rot_to_rpy(R):
    roll = math.atan2(R[2,1], R[2,2])
    pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = math.atan2(R[1,0], R[0,0])
    return np.array([roll, pitch, yaw])

# ============================================================
# ======================= LOADERS =============================
# ============================================================

def load_rgb(ep):
    imgs = []
    for p in sorted_images(ep / "rgb"):
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs

def load_depth(ep):
    imgs = []
    for p in sorted_images(ep / "depth"):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) * DEPTH_SCALE
        imgs.append(img)
    return imgs

def load_trajectory(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("t") or not line.strip():
                continue
            _, x, y, z, qx, qy, qz, qw = map(float, line.split())
            R = quat_to_rot(qx, qy, qz, qw)
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = [x, y, z]
            poses.append(T)
    return poses

# ============================================================
# ======================= PREPROCESS ==========================
# ============================================================

def preprocess_rgb(rgb):
    return np.stack([
        cv2.resize(img, IMG_SIZE, cv2.INTER_AREA).astype(np.float32) / 255.0
        for img in rgb
    ])

def preprocess_depth(depth):
    return np.stack([
        cv2.resize(
            np.clip(d, 0.0, DEPTH_CLIP_MAX) / DEPTH_CLIP_MAX,
            IMG_SIZE,
            cv2.INTER_NEAREST
        )[..., None].astype(np.float32)
        for d in depth
    ])

def compute_actions_se3(poses):
    actions = []
    for t in range(len(poses)-1):
        T1, T2 = poses[t], poses[t+1]
        dpos = T2[:3,3] - T1[:3,3]

        R_rel = T1[:3,:3].T @ T2[:3,:3]
        droll, dpitch, dyaw = rot_to_rpy(R_rel)

        actions.append([
            dpos[0], dpos[1], dpos[2],
            wrap_angle(droll),
            wrap_angle(dpitch),
            wrap_angle(dyaw),
        ])
    return np.array(actions, dtype=np.float32)

def normalize_actions(a):
    return np.clip(a / MAX_ACTION, -1.0, 1.0)

# ============================================================
# ======================= MAIN ================================
# ============================================================

def process_episode(ep):
    print(f"Processing {ep.name}")

    rgb = load_rgb(ep)
    depth = load_depth(ep) if USE_DEPTH else None
    traj = load_trajectory(ep / "trajectory_CameraTrajectory.txt")

    T = min(len(rgb), len(traj))
    if T < MIN_EPISODE_LENGTH:
        return None

    rgb, traj = rgb[:T], traj[:T]
    if depth is not None:
        depth = depth[:T]

    # Camera → EE
    traj_ee = [T_CAM2EE @ T for T in traj]

    actions_6d = compute_actions_se3(traj_ee)
    actions_6d = normalize_actions(actions_6d)

    rgb = preprocess_rgb(rgb[:-1])
    if depth is not None:
        depth = preprocess_depth(depth[:-1])

    pose_ee = np.array([T[:3,3] for T in traj_ee[:-1]], dtype=np.float32)

    return {
        "rgb": rgb,
        "depth": depth,
        "actions_6d": actions_6d,
        "actions_scara": actions_6d[:, [0,1,5]],
        "pose_ee": pose_ee,
        "episode_id": ep.name,
    }

def main():
    OUT_ROOT.mkdir(exist_ok=True)

    for ep in sorted(p for p in RAW_ROOT.iterdir() if (p / "trajectory_CameraTrajectory.txt").exists()):
        data = process_episode(ep)
        if data is None:
            continue

        np.savez_compressed(
            OUT_ROOT / f"{data['episode_id']}.npz",
            **data
        )
        print("  Saved")

if __name__ == "__main__":
    main()
