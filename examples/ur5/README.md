# UR5 Example

This directory contains tools for collecting demonstration data from a UR5 robot and converting it to LeRobot format for training.

## Requirements

Install the required dependencies:

```bash
pip install ur_rtde opencv-python h5py numpy
```

- **ur_rtde**: Real-time communication with Universal Robots
- **opencv-python**: Camera capture and image processing
- **h5py**: HDF5 file format for data storage

## Data Collection

### Quick Start

To collect demonstration data from your UR5 robot:

```bash
# Basic data collection (records current robot state)
python examples/ur5/ur_record.py --robot-ip 192.168.1.100 --output-dir ./ur5_data --num-episodes 10

# With freedrive teleoperation (kinesthetic teaching)
python examples/ur5/ur_record.py --robot-ip 192.168.1.100 --use-teleop --output-dir ./ur5_data --num-episodes 10

# With keyboard control
python examples/ur5/ur_record.py --robot-ip 192.168.1.100 --use-keyboard --output-dir ./ur5_data --num-episodes 10

# With tactile sensors enabled
python examples/ur5/ur_record.py --robot-ip 192.168.1.100 --use-tactile --output-dir ./ur5_data --num-episodes 10

# Test mode (no hardware required)
python examples/ur5/ur_record.py --mock --output-dir ./ur5_data --num-episodes 5
```

### Recorded Data

The script collects:
- **Images**: `cam_world` (external view) and `cam_wrist` (wrist-mounted camera)
- **Tactile** (optional): 5 tactile sensor images (`tactile_0` through `tactile_4`)
- **State**: UR5 joint positions (6 joints) + optional gripper position
- **Action**: Target joint positions

Data is saved in HDF5 format with the following structure:
```
episode_0000.hdf5
├── observations/
│   ├── qpos          # Joint positions [T, 6]
│   ├── qvel          # Joint velocities [T, 6]
│   ├── effort        # Joint currents/efforts [T, 6]
│   ├── gripper       # Gripper position [T, 1] (optional)
│   └── images/
│       ├── cam_world  # World camera [T, H, W, 3]
│       ├── cam_wrist  # Wrist camera [T, H, W, 3]
│       └── tactile_*  # Tactile sensors [T, H, W, 3] (optional)
├── action            # Actions [T, 6+1] (joints + gripper)
└── attrs: {num_steps, task, timestamp, control_freq}
```

### Camera Configuration

Cameras are accessed via OpenCV. You can specify:
- **Device index**: `0`, `1`, `2`, etc. for USB cameras
- **URL**: RTSP or HTTP streams (e.g., `rtsp://192.168.1.10:554/stream`)

```bash
# Using USB cameras (by device index)
python examples/ur5/ur_record.py --cam-world 0 --cam-wrist 1 --robot-ip 192.168.1.100

# Using IP cameras (by URL)
python examples/ur5/ur_record.py \
    --cam-world "rtsp://192.168.1.10:554/stream1" \
    --cam-wrist "rtsp://192.168.1.11:554/stream1" \
    --robot-ip 192.168.1.100
```

### Robot Connection

The script uses `ur_rtde` to communicate with the UR5 robot. Make sure:
1. The robot is powered on and in Remote Control mode
2. The robot's IP address is reachable from your computer
3. The External Control URCap is installed (for real-time control)

```bash
# Specify robot IP
python examples/ur5/ur_record.py --robot-ip 192.168.1.100 --output-dir ./ur5_data
```

## Convert to LeRobot Format

After collecting data, convert it to LeRobot format for training:

```bash
# Basic conversion
uv run examples/ur5/convert_ur5_data_to_lerobot.py \
    --raw-dir ./ur5_data \
    --repo-id your_username/ur5_dataset \
    --task "Pick and place task"

# With push to HuggingFace Hub
uv run examples/ur5/convert_ur5_data_to_lerobot.py \
    --raw-dir ./ur5_data \
    --repo-id your_username/ur5_dataset \
    --task "Pick and place task" \
    --push-to-hub
```

## Training Configuration

Below we provide an outline of how to implement the key components mentioned in the "Finetune on your data" section of the [README](../README.md) for finetuning on UR5 datasets.

First, we will define the `UR5Inputs` and `UR5Outputs` classes, which map the UR5 environment to the model and vice versa. Check the corresponding files in `src/openpi/policies/libero_policy.py` for comments explaining each line.

```python

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        state = np.concatenate([data["joints"], data["gripper"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}

```

Next, we will define the `UR5DataConfig` class, which defines how to process raw UR5 data from LeRobot dataset for training. For a full example, see the `LeRobotLiberoDataConfig` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

```python

@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

```

Finally, we define the TrainConfig for our UR5 dataset. Here, we define a config for fine-tuning pi0 on our UR5 dataset. See the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py) for more examples, e.g. for pi0-FAST or for LoRA fine-tuning.

```python
TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_dataset",
        # This config lets us reload the UR5 normalization stats from the base model checkpoint.
        # Reloading normalization stats can help transfer pre-trained models to new environments.
        # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. The recommended setting is True.
            prompt_from_task=True,
        ),
    ),
    # Load the pi0 base model checkpoint.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```





