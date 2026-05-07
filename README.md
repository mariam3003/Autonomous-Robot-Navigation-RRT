---

## How It Works

### 1. Occupancy Map
A floor plan blueprint was used to manually mark walls and generate a grayscale occupancy map. Obstacles are dilated by the robot's physical radius using OpenCV's `cv2.dilate` to ensure safe wall clearance during navigation.

### 2. RRT Path Planning
The RRT algorithm seeds random waypoints through free space and builds a tree structure. The robot's start position is read live from Gazebo's `/odom` topic and connected to a user-defined goal. The shortest path is then computed using **Dijkstra's algorithm**.

### 3. Autonomous Navigation
The robot follows the computed waypoints in real time by publishing velocity commands to `/cmd_vel`, continuously adjusting its linear and angular speed based on current position and heading error.

---

## ROS2 Interface

| Topic | Type | Description |
|-------|------|-------------|
| `/odom` | `nav_msgs/Odometry` | Subscribes, reads robot pose from Gazebo |
| `/cmd_vel` | `geometry_msgs/Twist` | Publishes, sends movement commands |

---

## Robot Model

The `scout-laser.urdf.xacro` defines a differential drive robot with:

- **360 degree Laser Scanner** - 10Hz, range 0.2m to 10m
- **Differential Drive** - left and right wheel joints
- **Caster Wheel** - for stability

---

## Visualization

The OpenCV RRT window displays:

| Color | Meaning |
|-------|---------|
| Orange lines | RRT tree edges |
| Blue lines | Shortest path to goal |
| Red circle | Current robot position |
| Purple circle | Goal position |
| Green dots | Random RRT sample points |

---

## Setup and Usage

### Prerequisites

- ROS2
- Gazebo
- Python 3.8+
- Python packages: `opencv-python`, `numpy`

### 1. Update File Paths

Before running, update the hardcoded paths in `t3.py` to match your workspace:

```python
self.maze = self.loading_maze_visualization('/home/common/Downloads/40px_new.png', dilation)
```

Also update the URDF path in the launch file if needed:

```python
urdf = os.path.join(get_package_share_directory('cpmr_ch4'), 'scout-laser.urdf.xacro')
```

### 2. Launch the Simulation

```bash
ros2 launch cpmr_ch4 launch_laser_robot.launch.py
```

### 3. Run the RRT Navigator

```bash
python3 t3.py
```

### 4. Set Your Goal

Edit the goal coordinates in `t3.py` before running:

```python
GOAL_X, GOAL_Y = 182, 560.5   # pixel coordinates on the map
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_POINTS` | 600 | Number of random points for RRT tree |
| `robot_radius` | 0.269 m | Used for obstacle dilation |
| `pixels` | 50 | Pixels per metre scale |
| `GOAL_X, GOAL_Y` | 182, 560.5 | Goal location in map pixel coordinates |
| `max_vel` | 0.2 m/s | Maximum linear velocity |
| `max_angular_vel` | 0.8 rad/s | Maximum angular velocity |

---

## Results

- The robot's origin maps to the **hallway** in the floor plan at map coordinates **(420, 336)**
- The RRT algorithm successfully computed paths to multiple goal locations
- Obstacle dilation ensured safe clearance from all walls
- The robot successfully navigated to all tested goal positions

---

## Notes

> Some file paths are hardcoded to `/home/common/CPMR3/...` and `/home/common/Downloads/...`. Update these to match your own ROS2 workspace and file locations before running.
