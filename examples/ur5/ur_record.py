"""
Data collection script for UR5 robot using ur_rtde and OpenCV.

This script collects data from a UR5 robot including:
- Images from cam_world and cam_wrist (with optional 5 tactile images)
- Robot joint states (6 joints + optional gripper)
- Actions (joint positions)

The collected data is saved in HDF5 format compatible with LeRobot conversion.

Requirements:
    pip install ur_rtde opencv-python h5py numpy

Example usage:
    python ur_record.py --output-dir /path/to/output --num-episodes 10 --episode-length 500
    
For teleoperation recording (freedrive mode):
    python ur_record.py --output-dir /path/to/output --use-teleop --num-episodes 10

For keyboard control recording:
    python ur_record.py --output-dir /path/to/output --use-keyboard --num-episodes 10
"""

import dataclasses
import datetime
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import h5py
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclasses.dataclass
class UR5RecordConfig:
    """Configuration for UR5 data recording."""
    
    # Output settings
    output_dir: str = "./ur5_data"
    episode_prefix: str = "episode"
    
    # Recording settings
    num_episodes: int = 10
    episode_length: int = 500  # Maximum steps per episode
    control_freq: float = 10.0  # Hz
    
    # UR5 robot settings
    robot_ip: str = "192.168.1.100"  # UR5 robot IP address
    ur5_num_joints: int = 6
    
    # Camera settings (device indices or RTSP/HTTP URLs)
    cam_world_source: int | str = 0  # Camera device index or URL
    cam_wrist_source: int | str = 1  # Camera device index or URL
    image_width: int = 640
    image_height: int = 480
    camera_fps: int = 30
    
    # Tactile sensor settings (optional)
    use_tactile: bool = False
    tactile_sources: tuple = (2, 3, 4, 5, 6)  # Camera indices for tactile sensors
    tactile_width: int = 224
    tactile_height: int = 224
    
    # Gripper settings (optional)
    use_gripper: bool = True
    gripper_port: int = 63352  # Robotiq gripper port
    
    # Control mode
    use_teleop: bool = False  # Use freedrive teleoperation mode
    use_keyboard: bool = False  # Use keyboard control
    
    # Task description
    task_description: str = "Manipulation task with UR5 robot"


# ============================================================================
# Camera Recorder using OpenCV
# ============================================================================

class CameraRecorder:
    """Records images from cameras using OpenCV."""
    
    def __init__(
        self,
        camera_configs: dict[str, int | str],
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30,
    ):
        """
        Initialize camera recorder.
        
        Args:
            camera_configs: Dict mapping camera names to device indices or URLs
                e.g., {"cam_world": 0, "cam_wrist": 1} or 
                      {"cam_world": "rtsp://192.168.1.10:554/stream"}
            image_width: Target image width
            image_height: Target image height
            fps: Target frames per second
        """
        self.camera_configs = camera_configs
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        
        self.captures: dict[str, cv2.VideoCapture] = {}
        self.latest_images: dict[str, np.ndarray] = {}
        self.image_locks: dict[str, threading.Lock] = {}
        self.running = False
        self.capture_threads: dict[str, threading.Thread] = {}
        
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Initialize all camera captures."""
        for cam_name, source in self.camera_configs.items():
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                print(f"Warning: Could not open camera {cam_name} (source: {source})")
                continue
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.captures[cam_name] = cap
            self.latest_images[cam_name] = None
            self.image_locks[cam_name] = threading.Lock()
            
            print(f"Initialized camera {cam_name} (source: {source})")
    
    def _capture_loop(self, cam_name: str):
        """Continuous capture loop for a single camera."""
        cap = self.captures[cam_name]
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Resize if needed
                if frame.shape[:2] != (self.image_height, self.image_width):
                    frame = cv2.resize(frame, (self.image_width, self.image_height))
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with self.image_locks[cam_name]:
                    self.latest_images[cam_name] = frame
            
            time.sleep(0.001)  # Small delay to prevent CPU overload
    
    def start(self):
        """Start continuous capture threads."""
        self.running = True
        for cam_name in self.captures:
            thread = threading.Thread(target=self._capture_loop, args=(cam_name,), daemon=True)
            thread.start()
            self.capture_threads[cam_name] = thread
        
        # Wait for initial frames
        time.sleep(0.5)
    
    def stop(self):
        """Stop capture threads and release cameras."""
        self.running = False
        
        for thread in self.capture_threads.values():
            thread.join(timeout=1.0)
        
        for cap in self.captures.values():
            cap.release()
    
    def get_images(self) -> dict[str, np.ndarray]:
        """Get the latest images from all cameras."""
        images = {}
        for cam_name in self.captures:
            with self.image_locks[cam_name]:
                if self.latest_images[cam_name] is not None:
                    images[cam_name] = self.latest_images[cam_name].copy()
        return images
    
    def get_single_frame(self, cam_name: str) -> np.ndarray | None:
        """Get a single frame from a specific camera (blocking)."""
        if cam_name not in self.captures:
            return None
        
        cap = self.captures[cam_name]
        ret, frame = cap.read()
        
        if ret:
            if frame.shape[:2] != (self.image_height, self.image_width):
                frame = cv2.resize(frame, (self.image_width, self.image_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None


# ============================================================================
# UR5 Robot Interface using ur_rtde
# ============================================================================

class UR5RobotInterface:
    """Interface for UR5 robot using ur_rtde library."""
    
    def __init__(
        self,
        robot_ip: str,
        num_joints: int = 6,
        use_gripper: bool = True,
        gripper_port: int = 63352,
    ):
        """
        Initialize UR5 robot interface.
        
        Args:
            robot_ip: IP address of the UR5 robot
            num_joints: Number of robot joints (always 6 for UR5)
            use_gripper: Whether to use a gripper (e.g., Robotiq)
            gripper_port: Port for gripper communication
        """
        self.robot_ip = robot_ip
        self.num_joints = num_joints
        self.use_gripper = use_gripper
        self.gripper_port = gripper_port
        
        self.rtde_r = None  # RTDEReceiveInterface
        self.rtde_c = None  # RTDEControlInterface
        self.gripper = None
        
        self._connect()
    
    def _connect(self):
        """Connect to the UR5 robot."""
        try:
            import rtde_receive
            import rtde_control
            
            print(f"Connecting to UR5 at {self.robot_ip}...")
            
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            
            print("Connected to UR5 robot")
            
            if self.use_gripper:
                self._connect_gripper()
                
        except ImportError:
            raise ImportError(
                "ur_rtde not found. Install with: pip install ur_rtde"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to UR5: {e}")
    
    def _connect_gripper(self):
        """Connect to Robotiq gripper via UR5's tool communication."""
        try:
            # Try to use Robotiq gripper through ur_rtde or socket
            import socket
            
            self.gripper_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.gripper_socket.connect((self.robot_ip, self.gripper_port))
            self.gripper_socket.settimeout(2.0)
            print("Connected to Robotiq gripper")
        except Exception as e:
            print(f"Warning: Could not connect to gripper: {e}")
            self.use_gripper = False
    
    def get_state(self) -> dict:
        """Get current robot state."""
        if self.rtde_r is None:
            return None
        
        state = {
            "qpos": np.array(self.rtde_r.getActualQ()),  # Joint positions [rad]
            "qvel": np.array(self.rtde_r.getActualQd()),  # Joint velocities [rad/s]
            "effort": np.array(self.rtde_r.getActualCurrent()),  # Joint currents [A]
        }
        
        # Add TCP pose if needed
        state["tcp_pose"] = np.array(self.rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
        state["tcp_speed"] = np.array(self.rtde_r.getActualTCPSpeed())
        state["tcp_force"] = np.array(self.rtde_r.getActualTCPForce())
        
        if self.use_gripper:
            state["gripper"] = np.array([self._get_gripper_position()])
        
        return state
    
    def _get_gripper_position(self) -> float:
        """Get gripper position (0.0 = closed, 1.0 = open)."""
        try:
            # Read gripper position from Robotiq
            # This is a simplified implementation
            # Actual implementation depends on gripper type
            if hasattr(self, 'gripper_socket'):
                self.gripper_socket.send(b"GET POS\n")
                response = self.gripper_socket.recv(1024).decode()
                # Parse response to get position
                # Normalize to 0-1 range
                return 0.5  # Placeholder
        except Exception:
            pass
        return 0.5
    
    def _set_gripper_position(self, position: float):
        """Set gripper position (0.0 = closed, 1.0 = open)."""
        try:
            if hasattr(self, 'gripper_socket'):
                # Convert 0-1 range to gripper command
                cmd = int(position * 255)
                self.gripper_socket.send(f"SET POS {cmd}\n".encode())
        except Exception as e:
            print(f"Warning: Failed to set gripper: {e}")
    
    def move_joints(self, positions: np.ndarray, speed: float = 0.5, acceleration: float = 0.5):
        """Move robot to joint positions."""
        if self.rtde_c is None:
            return
        
        self.rtde_c.moveJ(positions.tolist(), speed, acceleration, False)
    
    def move_joints_async(self, positions: np.ndarray, speed: float = 0.5, acceleration: float = 0.5):
        """Move robot to joint positions (non-blocking)."""
        if self.rtde_c is None:
            return
        
        self.rtde_c.servoJ(positions.tolist(), speed, acceleration, 0.008, 0.1, 300)
    
    def start_freedrive(self):
        """Enable freedrive mode for teleoperation."""
        if self.rtde_c is None:
            return
        
        self.rtde_c.teachMode()
        print("Freedrive mode enabled - you can now move the robot by hand")
    
    def stop_freedrive(self):
        """Disable freedrive mode."""
        if self.rtde_c is None:
            return
        
        self.rtde_c.endTeachMode()
        print("Freedrive mode disabled")
    
    def stop(self):
        """Stop robot movement."""
        if self.rtde_c is not None:
            self.rtde_c.stopJ(2.0)  # Deceleration
    
    def disconnect(self):
        """Disconnect from the robot."""
        if self.rtde_c is not None:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
        
        if self.rtde_r is not None:
            self.rtde_r.disconnect()
        
        if hasattr(self, 'gripper_socket'):
            self.gripper_socket.close()
        
        print("Disconnected from UR5")


# ============================================================================
# Mock Classes for Testing Without Hardware
# ============================================================================

class MockCameraRecorder:
    """Mock camera recorder for testing without real cameras."""
    
    def __init__(
        self,
        camera_configs: dict[str, int | str],
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30,
    ):
        self.camera_names = list(camera_configs.keys())
        self.image_width = image_width
        self.image_height = image_height
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def get_images(self) -> dict[str, np.ndarray]:
        """Return random images for testing."""
        return {
            cam_name: np.random.randint(
                0, 255, 
                (self.image_height, self.image_width, 3), 
                dtype=np.uint8
            )
            for cam_name in self.camera_names
        }


class MockUR5RobotInterface:
    """Mock robot interface for testing without real robot."""
    
    def __init__(
        self,
        robot_ip: str = "127.0.0.1",
        num_joints: int = 6,
        use_gripper: bool = True,
        gripper_port: int = 63352,
    ):
        self.num_joints = num_joints
        self.use_gripper = use_gripper
        self._current_positions = np.zeros(num_joints)
        self._gripper_pos = 0.5
        print(f"Mock UR5 interface initialized (IP: {robot_ip})")
    
    def get_state(self) -> dict:
        """Return mock robot state."""
        state = {
            "qpos": self._current_positions + np.random.randn(self.num_joints) * 0.001,
            "qvel": np.random.randn(self.num_joints) * 0.01,
            "effort": np.random.randn(self.num_joints) * 0.1,
            "tcp_pose": np.array([0.5, 0.0, 0.5, 0.0, 3.14, 0.0]),
            "tcp_speed": np.zeros(6),
            "tcp_force": np.random.randn(6) * 0.1,
        }
        if self.use_gripper:
            state["gripper"] = np.array([self._gripper_pos])
        return state
    
    def move_joints(self, positions: np.ndarray, speed: float = 0.5, acceleration: float = 0.5):
        self._current_positions = positions.copy()
    
    def move_joints_async(self, positions: np.ndarray, speed: float = 0.5, acceleration: float = 0.5):
        self._current_positions = positions.copy()
    
    def start_freedrive(self):
        print("Mock: Freedrive mode enabled")
    
    def stop_freedrive(self):
        print("Mock: Freedrive mode disabled")
    
    def stop(self):
        pass
    
    def disconnect(self):
        print("Mock: Disconnected")


# ============================================================================
# Episode Recorder
# ============================================================================

class EpisodeRecorder:
    """Records episodes and saves to HDF5 format."""
    
    def __init__(self, config: UR5RecordConfig):
        self.config = config
        self.episode_data = None
        self.step_count = 0
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_episode(self):
        """Initialize a new episode."""
        self.episode_data = {
            "observations": {
                "qpos": [],
                "qvel": [],
                "effort": [],
                "images": {},
            },
            "actions": [],
        }
        
        # Initialize image storage for each camera
        self.episode_data["observations"]["images"]["cam_world"] = []
        self.episode_data["observations"]["images"]["cam_wrist"] = []
        
        if self.config.use_tactile:
            for i in range(5):
                self.episode_data["observations"]["images"][f"tactile_{i}"] = []
        
        if self.config.use_gripper:
            self.episode_data["observations"]["gripper"] = []
        
        self.step_count = 0
    
    def add_step(
        self,
        images: dict[str, np.ndarray],
        robot_state: dict,
        action: np.ndarray,
    ):
        """Add a step to the current episode."""
        if self.episode_data is None:
            raise RuntimeError("Episode not started. Call start_episode() first.")
        
        # Store observations
        self.episode_data["observations"]["qpos"].append(robot_state["qpos"])
        self.episode_data["observations"]["qvel"].append(robot_state["qvel"])
        self.episode_data["observations"]["effort"].append(robot_state["effort"])
        
        if self.config.use_gripper and "gripper" in robot_state:
            self.episode_data["observations"]["gripper"].append(robot_state["gripper"])
        
        # Store images
        for cam_name in ["cam_world", "cam_wrist"]:
            if cam_name in images:
                self.episode_data["observations"]["images"][cam_name].append(images[cam_name])
        
        if self.config.use_tactile:
            for i in range(5):
                tactile_key = f"tactile_{i}"
                if tactile_key in images:
                    self.episode_data["observations"]["images"][tactile_key].append(images[tactile_key])
        
        # Store action
        self.episode_data["actions"].append(action)
        
        self.step_count += 1
    
    def save_episode(self, episode_idx: int, task_description: Optional[str] = None) -> str:
        """Save the current episode to HDF5 file."""
        if self.episode_data is None or self.step_count == 0:
            raise RuntimeError("No episode data to save.")
        
        # Generate filename
        filename = self.output_dir / f"{self.config.episode_prefix}_{episode_idx:04d}.hdf5"
        
        with h5py.File(filename, "w") as f:
            # Save observations
            obs_group = f.create_group("observations")
            
            # Save joint states
            obs_group.create_dataset(
                "qpos",
                data=np.array(self.episode_data["observations"]["qpos"]),
                dtype=np.float32,
            )
            obs_group.create_dataset(
                "qvel",
                data=np.array(self.episode_data["observations"]["qvel"]),
                dtype=np.float32,
            )
            obs_group.create_dataset(
                "effort",
                data=np.array(self.episode_data["observations"]["effort"]),
                dtype=np.float32,
            )
            
            if self.config.use_gripper and "gripper" in self.episode_data["observations"]:
                obs_group.create_dataset(
                    "gripper",
                    data=np.array(self.episode_data["observations"]["gripper"]),
                    dtype=np.float32,
                )
            
            # Save images
            images_group = obs_group.create_group("images")
            for cam_name, images_list in self.episode_data["observations"]["images"].items():
                if images_list:
                    images_group.create_dataset(
                        cam_name,
                        data=np.array(images_list),
                        dtype=np.uint8,
                        compression="gzip",
                        compression_opts=4,
                    )
            
            # Save actions
            f.create_dataset(
                "action",
                data=np.array(self.episode_data["actions"]),
                dtype=np.float32,
            )
            
            # Save metadata
            f.attrs["num_steps"] = self.step_count
            f.attrs["task"] = task_description or self.config.task_description
            f.attrs["timestamp"] = datetime.datetime.now().isoformat()
            f.attrs["control_freq"] = self.config.control_freq
        
        print(f"Saved episode {episode_idx} with {self.step_count} steps to {filename}")
        self.episode_data = None
        return str(filename)


# ============================================================================
# Keyboard Controller
# ============================================================================

class KeyboardController:
    """Simple keyboard controller for UR5 joint positions."""
    
    def __init__(self, num_joints: int = 6, use_gripper: bool = True):
        self.num_joints = num_joints
        self.use_gripper = use_gripper
        self.joint_increments = np.array([0.05] * num_joints)  # radians
        self.gripper_increment = 0.1
        
        # Key mappings for joints (increase/decrease)
        # Using ASCII codes for OpenCV waitKey
        self.joint_keys = {
            ord('q'): (0, 1), ord('a'): (0, -1),  # Joint 0
            ord('w'): (1, 1), ord('s'): (1, -1),  # Joint 1
            ord('e'): (2, 1), ord('d'): (2, -1),  # Joint 2
            ord('r'): (3, 1), ord('f'): (3, -1),  # Joint 3
            ord('t'): (4, 1), ord('g'): (4, -1),  # Joint 4
            ord('y'): (5, 1), ord('h'): (5, -1),  # Joint 5
        }
        self.gripper_open_key = ord('o')
        self.gripper_close_key = ord('c')
    
    def get_action_delta(self, key: int) -> tuple[np.ndarray, float]:
        """Get action delta from key press."""
        joint_delta = np.zeros(self.num_joints)
        gripper_delta = 0.0
        
        if key in self.joint_keys:
            joint_idx, direction = self.joint_keys[key]
            joint_delta[joint_idx] = direction * self.joint_increments[joint_idx]
        elif key == self.gripper_open_key:
            gripper_delta = self.gripper_increment
        elif key == self.gripper_close_key:
            gripper_delta = -self.gripper_increment
        
        return joint_delta, gripper_delta
    
    def print_controls(self):
        """Print keyboard control instructions."""
        print("\n=== Keyboard Controls ===")
        print("Joint controls (increase/decrease):")
        print("  Joint 0 (base):     Q/A")
        print("  Joint 1 (shoulder): W/S")
        print("  Joint 2 (elbow):    E/D")
        print("  Joint 3 (wrist 1):  R/F")
        print("  Joint 4 (wrist 2):  T/G")
        print("  Joint 5 (wrist 3):  Y/H")
        if self.use_gripper:
            print("Gripper: O (open) / C (close)")
        print("Episode: SPACE (start/stop recording)")
        print("Exit: ESC")
        print("========================\n")


# ============================================================================
# Data Collection Functions
# ============================================================================

def collect_episode_with_teleop(
    camera_recorder: CameraRecorder,
    robot: UR5RobotInterface,
    episode_recorder: EpisodeRecorder,
    config: UR5RecordConfig,
    episode_idx: int,
) -> bool:
    """
    Collect a single episode using freedrive teleoperation.
    
    The robot is put in freedrive mode, allowing the human operator
    to physically guide it while recording.
    
    Returns True if episode was successfully recorded.
    """
    dt = 1.0 / config.control_freq
    episode_recorder.start_episode()
    
    print(f"\n=== Starting Episode {episode_idx} (Teleoperation Mode) ===")
    print("Press SPACE to start recording, SPACE again to stop...")
    print("Press ESC to cancel episode")
    
    # Create display window
    cv2.namedWindow("UR5 Teleop Recording", cv2.WINDOW_NORMAL)
    
    recording = False
    step = 0
    
    # Enable freedrive mode
    robot.start_freedrive()
    
    try:
        while step < config.episode_length:
            loop_start = time.time()
            
            # Get current observations
            images = camera_recorder.get_images()
            robot_state = robot.get_state()
            
            if robot_state is None:
                time.sleep(dt)
                continue
            
            # Display images
            if "cam_world" in images:
                display_img = images["cam_world"].copy()
                # Convert RGB to BGR for OpenCV display
                display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                
                # Add status text
                status = "RECORDING" if recording else "READY (Press SPACE)"
                color = (0, 255, 0) if recording else (0, 165, 255)
                cv2.putText(display_img, f"Status: {status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display_img, f"Step: {step}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show joint positions
                qpos = robot_state["qpos"]
                for i, pos in enumerate(qpos):
                    cv2.putText(display_img, f"J{i}: {np.degrees(pos):.1f}°", 
                               (10, 100 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("UR5 Teleop Recording", display_img)
            
            # Handle keyboard input
            key = cv2.waitKey(int(dt * 1000)) & 0xFF
            
            if key == 27:  # ESC
                print("Recording cancelled")
                break
            elif key == 32:  # SPACE
                recording = not recording
                if recording:
                    print("Recording started!")
                else:
                    print("Recording stopped")
                    break
            
            # Record step if recording
            if recording:
                # For teleop, the action is the current joint positions
                action = robot_state["qpos"].copy()
                if config.use_gripper and "gripper" in robot_state:
                    action = np.concatenate([action, robot_state["gripper"]])
                
                episode_recorder.add_step(images, robot_state, action)
                step += 1
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    finally:
        # Disable freedrive mode
        robot.stop_freedrive()
        cv2.destroyAllWindows()
    
    # Save episode
    if step > 0:
        episode_recorder.save_episode(episode_idx, config.task_description)
        return True
    return False


def collect_episode_with_keyboard(
    camera_recorder: CameraRecorder,
    robot: UR5RobotInterface,
    episode_recorder: EpisodeRecorder,
    config: UR5RecordConfig,
    episode_idx: int,
) -> bool:
    """
    Collect a single episode using keyboard control.
    
    Uses OpenCV window to capture keyboard input and control the robot.
    Returns True if episode was successfully recorded.
    """
    keyboard = KeyboardController(
        num_joints=config.ur5_num_joints,
        use_gripper=config.use_gripper,
    )
    keyboard.print_controls()
    
    dt = 1.0 / config.control_freq
    episode_recorder.start_episode()
    
    # Get initial state
    robot_state = robot.get_state()
    current_position = np.zeros(config.ur5_num_joints)
    current_gripper = 0.5
    
    if robot_state is not None:
        current_position = robot_state["qpos"].copy()
        if config.use_gripper and "gripper" in robot_state:
            current_gripper = robot_state["gripper"][0]
    
    print(f"\n=== Starting Episode {episode_idx} (Keyboard Control) ===")
    print("Press SPACE to start recording...")
    
    # Create display window
    cv2.namedWindow("UR5 Keyboard Recording", cv2.WINDOW_NORMAL)
    
    recording = False
    step = 0
    
    while step < config.episode_length:
        loop_start = time.time()
        
        # Get current observations
        images = camera_recorder.get_images()
        robot_state = robot.get_state()
        
        # Display images
        if "cam_world" in images:
            display_img = images["cam_world"].copy()
            display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            
            # Add status text
            status = "RECORDING" if recording else "PAUSED"
            color = (0, 255, 0) if recording else (0, 0, 255)
            cv2.putText(display_img, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display_img, f"Step: {step}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show target positions
            for i, pos in enumerate(current_position):
                cv2.putText(display_img, f"Target J{i}: {np.degrees(pos):.1f}°", 
                           (10, 100 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("UR5 Keyboard Recording", display_img)
        
        # Handle keyboard input
        key = cv2.waitKey(int(dt * 1000)) & 0xFF
        
        if key == 27:  # ESC
            print("Recording cancelled")
            cv2.destroyAllWindows()
            return False
        elif key == 32:  # SPACE
            recording = not recording
            if recording:
                print("Recording started!")
            else:
                print("Recording paused")
        elif key != 255:
            # Process movement key
            joint_delta, gripper_delta = keyboard.get_action_delta(key)
            current_position += joint_delta
            current_gripper = np.clip(current_gripper + gripper_delta, 0, 1)
            
            # Send command to robot
            robot.move_joints_async(current_position)
        
        # Record step if recording
        if recording and robot_state is not None:
            action = current_position.copy()
            if config.use_gripper:
                action = np.concatenate([action, [current_gripper]])
            
            episode_recorder.add_step(images, robot_state, action)
            step += 1
        
        # Maintain control frequency
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
    
    cv2.destroyAllWindows()
    
    # Save episode
    if step > 0:
        episode_recorder.save_episode(episode_idx, config.task_description)
        return True
    return False


def collect_episode_autonomous(
    camera_recorder: CameraRecorder,
    robot: UR5RobotInterface,
    episode_recorder: EpisodeRecorder,
    config: UR5RecordConfig,
    episode_idx: int,
    action_fn: Callable[[dict, dict], np.ndarray] | None = None,
) -> bool:
    """
    Collect a single episode with autonomous or scripted actions.
    
    Args:
        action_fn: Optional function that takes (images, robot_state) and returns action.
                   If None, records current robot state as action.
    
    Returns True if episode was successfully recorded.
    """
    dt = 1.0 / config.control_freq
    episode_recorder.start_episode()
    
    print(f"\n=== Starting Episode {episode_idx} (Autonomous Mode) ===")
    
    # Create display window
    cv2.namedWindow("UR5 Autonomous Recording", cv2.WINDOW_NORMAL)
    
    step = 0
    
    while step < config.episode_length:
        loop_start = time.time()
        
        # Get current observations
        images = camera_recorder.get_images()
        robot_state = robot.get_state()
        
        if robot_state is None:
            time.sleep(dt)
            continue
        
        # Display images
        if "cam_world" in images:
            display_img = images["cam_world"].copy()
            display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            cv2.putText(display_img, f"Step: {step}/{config.episode_length}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("UR5 Autonomous Recording", display_img)
        
        # Check for cancel
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Recording cancelled")
            cv2.destroyAllWindows()
            return False
        
        # Get action
        if action_fn is not None:
            action = action_fn(images, robot_state)
        else:
            # Default: record current position as action
            action = robot_state["qpos"].copy()
            if config.use_gripper and "gripper" in robot_state:
                action = np.concatenate([action, robot_state["gripper"]])
        
        # Record step
        episode_recorder.add_step(images, robot_state, action)
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}/{config.episode_length}")
        
        # Maintain control frequency
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
    
    cv2.destroyAllWindows()
    
    # Save episode
    episode_recorder.save_episode(episode_idx, config.task_description)
    return True


# ============================================================================
# Main Data Collection Function
# ============================================================================

def collect_ur5_data(config: UR5RecordConfig, use_mock: bool = False):
    """
    Main function to collect UR5 robot data.
    
    Args:
        config: Recording configuration
        use_mock: If True, use mock recorders for testing without hardware
    """
    print("=" * 60)
    print("UR5 Data Collection (ur_rtde + OpenCV)")
    print("=" * 60)
    print(f"Output directory: {config.output_dir}")
    print(f"Number of episodes: {config.num_episodes}")
    print(f"Episode length: {config.episode_length}")
    print(f"Control frequency: {config.control_freq} Hz")
    print(f"Robot IP: {config.robot_ip}")
    print(f"Use tactile: {config.use_tactile}")
    print(f"Use gripper: {config.use_gripper}")
    print("=" * 60)
    
    # Set up camera configs
    camera_configs = {
        "cam_world": config.cam_world_source,
        "cam_wrist": config.cam_wrist_source,
    }
    
    if config.use_tactile:
        for i, source in enumerate(config.tactile_sources[:5]):
            camera_configs[f"tactile_{i}"] = source
    
    # Initialize components
    if use_mock:
        print("\nUsing MOCK components (no hardware)")
        camera_recorder = MockCameraRecorder(
            camera_configs=camera_configs,
            image_width=config.image_width,
            image_height=config.image_height,
            fps=config.camera_fps,
        )
        robot = MockUR5RobotInterface(
            robot_ip=config.robot_ip,
            num_joints=config.ur5_num_joints,
            use_gripper=config.use_gripper,
        )
    else:
        print("\nInitializing hardware components...")
        camera_recorder = CameraRecorder(
            camera_configs=camera_configs,
            image_width=config.image_width,
            image_height=config.image_height,
            fps=config.camera_fps,
        )
        robot = UR5RobotInterface(
            robot_ip=config.robot_ip,
            num_joints=config.ur5_num_joints,
            use_gripper=config.use_gripper,
            gripper_port=config.gripper_port,
        )
    
    episode_recorder = EpisodeRecorder(config)
    
    # Start camera capture
    camera_recorder.start()
    
    # Collect episodes
    successful_episodes = 0
    try:
        for ep_idx in range(config.num_episodes):
            print(f"\n--- Episode {ep_idx + 1}/{config.num_episodes} ---")
            
            try:
                if config.use_teleop:
                    success = collect_episode_with_teleop(
                        camera_recorder, robot, episode_recorder,
                        config, ep_idx
                    )
                elif config.use_keyboard:
                    success = collect_episode_with_keyboard(
                        camera_recorder, robot, episode_recorder,
                        config, ep_idx
                    )
                else:
                    success = collect_episode_autonomous(
                        camera_recorder, robot, episode_recorder,
                        config, ep_idx
                    )
                
                if success:
                    successful_episodes += 1
                    
            except KeyboardInterrupt:
                print("\nData collection interrupted by user")
                break
            except Exception as e:
                print(f"Error during episode {ep_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    finally:
        # Cleanup
        camera_recorder.stop()
        robot.disconnect()
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"Data collection complete!")
    print(f"Successfully recorded {successful_episodes}/{config.num_episodes} episodes")
    print(f"Data saved to: {config.output_dir}")
    print("=" * 60)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line interface for UR5 data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect demonstration data from UR5 robot using ur_rtde and OpenCV"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", type=str, default="./ur5_data",
        help="Output directory for HDF5 files"
    )
    parser.add_argument(
        "--episode-prefix", type=str, default="episode",
        help="Prefix for episode filenames"
    )
    
    # Recording settings
    parser.add_argument(
        "--num-episodes", type=int, default=10,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--episode-length", type=int, default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--control-freq", type=float, default=10.0,
        help="Control frequency in Hz"
    )
    
    # Robot settings
    parser.add_argument(
        "--robot-ip", type=str, default="192.168.1.100",
        help="IP address of UR5 robot"
    )
    
    # Camera settings
    parser.add_argument(
        "--cam-world", type=str, default="0",
        help="World camera source (device index or URL)"
    )
    parser.add_argument(
        "--cam-wrist", type=str, default="1",
        help="Wrist camera source (device index or URL)"
    )
    parser.add_argument(
        "--image-width", type=int, default=640,
        help="Image width"
    )
    parser.add_argument(
        "--image-height", type=int, default=480,
        help="Image height"
    )
    
    # Tactile settings
    parser.add_argument(
        "--use-tactile", action="store_true",
        help="Enable tactile sensor recording"
    )
    
    # Gripper settings
    parser.add_argument(
        "--no-gripper", action="store_true",
        help="Disable gripper recording"
    )
    
    # Control mode
    parser.add_argument(
        "--use-teleop", action="store_true",
        help="Use freedrive teleoperation mode"
    )
    parser.add_argument(
        "--use-keyboard", action="store_true",
        help="Use keyboard control mode"
    )
    
    # Task settings
    parser.add_argument(
        "--task", type=str, default="Manipulation task with UR5 robot",
        help="Task description"
    )
    
    # Testing
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock components for testing without hardware"
    )
    
    args = parser.parse_args()
    
    # Parse camera sources (convert to int if numeric)
    def parse_camera_source(source: str):
        try:
            return int(source)
        except ValueError:
            return source
    
    # Create config
    config = UR5RecordConfig(
        output_dir=args.output_dir,
        episode_prefix=args.episode_prefix,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        control_freq=args.control_freq,
        robot_ip=args.robot_ip,
        cam_world_source=parse_camera_source(args.cam_world),
        cam_wrist_source=parse_camera_source(args.cam_wrist),
        image_width=args.image_width,
        image_height=args.image_height,
        use_tactile=args.use_tactile,
        use_gripper=not args.no_gripper,
        use_teleop=args.use_teleop,
        use_keyboard=args.use_keyboard,
        task_description=args.task,
    )
    
    # Run data collection
    collect_ur5_data(config, use_mock=args.mock)


if __name__ == "__main__":
    main()
