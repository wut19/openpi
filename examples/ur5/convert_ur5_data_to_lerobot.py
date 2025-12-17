"""
Script to convert UR5 hdf5 data to the LeRobot dataset v2.0 format.

Example usage: 
    uv run examples/ur5/convert_ur5_data_to_lerobot.py \
        --raw-dir /path/to/raw/data \
        --repo-id <org>/<dataset-name> \
        --task "Pick and place task"
"""

import dataclasses
import shutil
from pathlib import Path
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Configuration for LeRobot dataset creation."""
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str = "ur5",
    mode: Literal["video", "image"] = "video",
    *,
    num_joints: int = 6,
    use_gripper: bool = True,
    use_tactile: bool = False,
    has_velocity: bool = True,
    has_effort: bool = True,
    fps: int = 10,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Create an empty LeRobot dataset with UR5 configuration."""
    
    # Define motor names for UR5
    motors = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    
    if use_gripper:
        motors.append("gripper")
    
    # Define cameras
    cameras = ["cam_world", "cam_wrist"]
    
    if use_tactile:
        cameras.extend([f"tactile_{i}" for i in range(5)])
    
    # Build features dictionary
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }
    
    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }
    
    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }
    
    # Add camera features
    for cam in cameras:
        # Tactile sensors may have different resolution
        if cam.startswith("tactile"):
            shape = (3, 224, 224)
        else:
            shape = (3, 480, 640)
        
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": shape,
            "names": ["channels", "height", "width"],
        }
    
    # Remove existing dataset if present
    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)
    
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras_from_hdf5(hdf5_files: list[Path]) -> list[str]:
    """Get list of camera names from HDF5 file."""
    with h5py.File(hdf5_files[0], "r") as ep:
        if "/observations/images" in ep:
            return list(ep["/observations/images"].keys())
    return []


def has_velocity(hdf5_files: list[Path]) -> bool:
    """Check if velocity data is present in HDF5 files."""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    """Check if effort data is present in HDF5 files."""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def has_gripper(hdf5_files: list[Path]) -> bool:
    """Check if gripper data is present in HDF5 files."""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/gripper" in ep


def has_tactile(hdf5_files: list[Path]) -> bool:
    """Check if tactile data is present in HDF5 files."""
    with h5py.File(hdf5_files[0], "r") as ep:
        if "/observations/images" in ep:
            return any("tactile" in key for key in ep["/observations/images"].keys())
    return False


def load_raw_images_per_camera(
    ep: h5py.File, 
    cameras: list[str]
) -> dict[str, np.ndarray]:
    """Load images from HDF5 file for all cameras."""
    imgs_per_cam = {}
    for camera in cameras:
        if f"/observations/images/{camera}" not in ep:
            continue
            
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        
        if uncompressed:
            # Load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            
            # Load one compressed image after the other and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    cameras: list[str],
) -> tuple[
    dict[str, np.ndarray],  # images
    torch.Tensor,           # state (qpos + gripper)
    torch.Tensor,           # action
    torch.Tensor | None,    # velocity
    torch.Tensor | None,    # effort
]:
    """Load all data from a single episode HDF5 file."""
    with h5py.File(ep_path, "r") as ep:
        # Load joint positions
        qpos = ep["/observations/qpos"][:]
        
        # Load gripper if present
        if "/observations/gripper" in ep:
            gripper = ep["/observations/gripper"][:]
            state = np.concatenate([qpos, gripper], axis=-1)
        else:
            state = qpos
        
        state = torch.from_numpy(state)
        
        # Load action
        action = torch.from_numpy(ep["/action"][:])
        
        # Load velocity
        velocity = None
        if "/observations/qvel" in ep:
            qvel = ep["/observations/qvel"][:]
            if "/observations/gripper" in ep:
                # Add zero velocity for gripper dimension
                gripper_vel = np.zeros((qvel.shape[0], 1))
                qvel = np.concatenate([qvel, gripper_vel], axis=-1)
            velocity = torch.from_numpy(qvel)
        
        # Load effort
        effort = None
        if "/observations/effort" in ep:
            eff = ep["/observations/effort"][:]
            if "/observations/gripper" in ep:
                # Add zero effort for gripper dimension
                gripper_eff = np.zeros((eff.shape[0], 1))
                eff = np.concatenate([eff, gripper_eff], axis=-1)
            effort = torch.from_numpy(eff)
        
        # Load images
        imgs_per_cam = load_raw_images_per_camera(ep, cameras)
    
    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    cameras: list[str],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """Populate the dataset with data from HDF5 files."""
    if episodes is None:
        episodes = range(len(hdf5_files))
    
    for ep_idx in tqdm.tqdm(episodes, desc="Converting episodes"):
        ep_path = hdf5_files[ep_idx]
        
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(
            ep_path, cameras
        )
        num_frames = state.shape[0]
        
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            
            dataset.add_frame(frame)
        
        dataset.save_episode(task=task)
    
    return dataset


def convert_ur5_data(
    raw_dir: Path,
    repo_id: str,
    task: str = "UR5 manipulation task",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    fps: int = 10,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    Convert UR5 HDF5 data to LeRobot dataset format.
    
    Args:
        raw_dir: Directory containing episode_*.hdf5 files
        repo_id: HuggingFace repo ID for the dataset (e.g., "user/ur5_dataset")
        task: Task description string
        episodes: Optional list of episode indices to convert
        push_to_hub: Whether to push the dataset to HuggingFace Hub
        fps: Frames per second of the recorded data
        mode: "video" or "image" storage mode
        dataset_config: Configuration for dataset creation
    """
    # Remove existing dataset
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)
    
    # Find HDF5 files
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise ValueError(f"Raw data directory does not exist: {raw_dir}")
    
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        # Also try without underscore
        hdf5_files = sorted(raw_dir.glob("episode*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {raw_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Detect features from first file
    cameras = get_cameras_from_hdf5(hdf5_files)
    use_gripper = has_gripper(hdf5_files)
    use_tactile = has_tactile(hdf5_files)
    
    print(f"Cameras: {cameras}")
    print(f"Has gripper: {use_gripper}")
    print(f"Has tactile: {use_tactile}")
    print(f"Has velocity: {has_velocity(hdf5_files)}")
    print(f"Has effort: {has_effort(hdf5_files)}")
    
    # Create dataset
    dataset = create_empty_dataset(
        repo_id,
        robot_type="ur5",
        mode=mode,
        num_joints=6,
        use_gripper=use_gripper,
        use_tactile=use_tactile,
        has_velocity=has_velocity(hdf5_files),
        has_effort=has_effort(hdf5_files),
        fps=fps,
        dataset_config=dataset_config,
    )
    
    # Populate dataset
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        cameras=cameras,
        task=task,
        episodes=episodes,
    )
    
    # Consolidate and optionally push to hub
    dataset.consolidate()
    
    if push_to_hub:
        dataset.push_to_hub()
    
    print(f"\nDataset saved to: {LEROBOT_HOME / repo_id}")
    return dataset


@dataclasses.dataclass
class ConvertArgs:
    """Arguments for conversion script."""
    raw_dir: Path
    repo_id: str
    task: str = "UR5 manipulation task"
    push_to_hub: bool = False
    fps: int = 10
    mode: Literal["video", "image"] = "video"


if __name__ == "__main__":
    args = tyro.cli(ConvertArgs)
    convert_ur5_data(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        task=args.task,
        push_to_hub=args.push_to_hub,
        fps=args.fps,
        mode=args.mode,
    )

