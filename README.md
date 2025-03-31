# Grounded-SAM-2-ROS

This repository integrates **Grounded-SAM2 (Florence2_SAM2)** with **ROS** to process images from the `realsense_ros` image topic.  
이 레포지토리는 **Grounded-SAM2 (Florence2_SAM2)** 를 **ROS**와 연동하여 `realsense_ros` 이미지 토픽을 처리할 수 있도록 구성되어 있습니다.

## 📦 Prerequisites

### 1. Grounded-SAM-2 Module 
   - Download from: [https://github.com/IDEA-Research/Grounded-SAM-2.git](https://github.com/IDEA-Research/Grounded-SAM-2.git)  
   - Follow the *Installation without Docker* instructions provided in the repository.

### 2. ROS Noetic 
   - Install ROS Noetic on Ubuntu by following the official guide:  
     [https://wiki.ros.org/noetic/Installation/Ubuntu](https://wiki.ros.org/noetic/Installation/Ubuntu)

### 3. realsense-ros
   - For ROS Noetic, install the realsense-ros package following the instructions at:  
     [https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy?tab=readme-ov-file](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy?tab=readme-ov-file)
   - If preferred, you can also **install from source** by cloning the repository and building it as described in their README.  
   - Complete the setup as described in the repository.

## 🛠️ Installation Guide

### 1. Create a Catkin Workspace

Open a terminal and run:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/kimyj-sejong/Grounded-SAM-2-ROS.git
cd ../
```

### 2. Update Custom Paths

- Launch File:
In `florence2_sam2_ros/launch/florence2_sam2_realsense.launch`, replace all instances of
`/your/custom/path/to/` with the absolute path to your Grounded-SAM-2 project directory.

- Node Script:
In `florence2_sam2_ros/scripts/florence2_sam2_ros_realsense_node.py`, replace `/your/custom/path/to/` with the same absolute path.

### 3. Build the Workspace

Source ROS Noetic and build:

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin build
```
## 🚀 Running the System

After a successful build, open three separate terminals and execute:

### Terminal 1 – Start ROS Master

```bash
source /opt/ros/noetic/setup.bash
roscore
```

### Terminal 2 – Launch the RealSense Driver

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws   # Adjust if your workspace is in a different location.
source devel/setup.bash
roslaunch realsense2_camera rs_camera.launch
```

### Terminal 3 – Launch Grounded-SAM2-ROS Node

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws   # Adjust if needed.
source devel/setup.bash
roslaunch florence2_sam2_ros florence2_sam2_realsense.launch
```

## ⚠️ Troubleshooting

If you see an error like:

```css
Falling back to all available kernels for scaled_dot_product_attention (which may have a slower speed).
```

Then, in the file `sam2/modeling/sam/transformer.py` (around line 22), add the following lines:

```python
USE_FLASH_ATTN = False
MATH_KERNEL_ON = True
OLD_GPU = True
```

## 📜 License
This project is based on the original Grounded-SAM-2 repository. Please refer to the original repository’s license for details.

이 프로젝트는 원본 Grounded-SAM-2 코드를 기반으로 하며, 원본 라이선스 고지를 준수합니다.
