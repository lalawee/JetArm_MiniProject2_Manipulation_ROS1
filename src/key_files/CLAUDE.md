# CLAUDE.md — JetArm Mini Project 2

## Project Overview

RSE4502 Mini Project 2 — JetArm Robot Manipulation (SIT, Prof Yuan Qilong).
Three graded challenges on a physical JetArm 5-DOF (6 servos) serial manipulator,
Jetson Nano, ROS Noetic. All challenges share AprilTag detection, camera-to-base
frame transforms, IK, and Jacobian-based singularity monitoring.

- Course site: https://ql-yuan.github.io/Jetarm_Session2_Instruction/
- GitHub: https://github.com/Hiwonder/JetArm branch Jetson_nano_ros1
- Official docs: https://wiki.hiwonder.com/projects/JetArm/en/jetarm-jetson-nano/

## Workspace & Build

```bash
# Workspace root
cd ~/jetarm

# Build (NOT catkin_make)
catkin build hiwonder_grasp

# Shell is zsh
source devel/setup.zsh

# Main working package
~/jetarm/src/hiwonder_grasp/scripts/
```

## Robot Hardware

JetArm 5-DOF arm with 6 servos (servo 6 is fixed gripper orientation).
Only joints 1-5 are active; joint 6 is fixed at 180°.
5 DOF means not redundant — can't fully control 6D pose independently.
`get_ik` handles this by taking `pitch` as a fixed parameter.

### DH Parameters (Modified DH Convention)

| Joint | α (deg) | a (m)   | d (m)     | θ offset | Range (deg)  |
|-------|---------|---------|-----------|----------|--------------|
| 1     | 0       | 0       | l0=0.10315| 0        | (-120, 120)  |
| 2     | -90     | 0       | 0         | -90      | (-180, 0)    |
| 3     | 0       | l1=0.12942 | 0      | 0        | (-120, 120)  |
| 4     | 0       | l2=0.12942 | 0      | -90      | (-200, 20)   |
| 5     | -90     | 0       | l3=0.05945| 0        | (-120, 120)  |
| 6     | 0       | 0       | l4=0.02545| 180      | fixed        |

Link lengths: l0=0.10315m, l1=0.12942m, l2=0.12942m, l3=0.05945m, l4=0.02545m

### Key Transforms

**Camera in link5:**
```
T_link5_camera = translation([-0.045, 0, 0.02]) * rotz(-90°)
```
Published via tf_hand2camera.launch:
```xml
<node pkg="tf" type="static_transform_publisher" name="eef_to_camera_transform"
      args="-0.045 0.0 0.02 -1.57 0 0 link5 rgbd_cam_color_optical_frame 100"/>
```
NOTE: Slide 22 in the project PDF shows roll=-0.1 but the actual launch file has roll=0.
May need tuning on real hardware.

**EEF in link5:**
```
T_link5_eef = translation([0, 0, 0.11054687369]) * roty(-90°)
```

**Camera intrinsics:** fx=452.533, fy=452.533, cx=325.613, cy=240.352
**AprilTag size:** 0.025m, family: tag36h11

### Transform Chain (critical for all challenges)

Camera is eye-in-hand (mounted on link5). AprilTag detection gives tag pose in
camera frame. To use for IK, need it in base_link frame:

```
base_link → link5        (from robot joint states / TF tree)
link5 → camera frame     (from tf_hand2camera.launch static TF)
```

ROS TF handles this via tf.TransformListener — look up base_link → rgbd_cam_color_optical_frame.
**Requires tf_hand2camera.launch to be running.**

## ROS Interfaces

### Topics
- `/jetarm/object_poses` — String (JSON): `{"tags": [{"id": 1, "x": ..., "y": ..., "z": ...}]}`
- `/controllers/multi_id_pos_dur` — MultiRawIdPosDur (servo commands, joints 1-4)
- `/controllers/id_pos_dur` — RawIdPosDur (gripper control)
- `/joint_states` — sensor_msgs/JointState
- `/rgbd_cam/color/image_rect_color` — sensor_msgs/Image (camera feed)

### Services
- `/kinematics/set_pose_target` — IK solver (returns pulse values)
- `/kinematics/get_current_pose` — FK, returns xyz + quaternion (GetRobotPose)

### Launch Files
- `jetarm_bringup base.launch` — bring up robot
- `jetarm_kinematics kinematics_node_6dof.launch` — IK/FK services
- `tf_hand2camera.launch` — static TF for camera mount (MUST be running)

### Key Python APIs
```python
# IK
from jetarm_kinematics.kinematics_control import set_pose_target
result = set_pose_target(coord, pitch, yaw_range, 1)
# result[1] = servo pulse list; empty = IK failed

# Alternative IK
from jetarm_kinematics.inverse_kinematics import get_ik
res = get_ik(coordinate, pitch, [-180, 180])

# Angle conversion
import jetarm_kinematics.transform as transform
pulse = transform.angle2pulse(angles)
angles = transform.pulse2angle(pulses)

# Servo control (joints 1-4 for arm motion)
from jetarm_sdk import bus_servo_control, gripper_control
bus_servo_control.set_servos(pub, duration_ms, ((1, p1), (2, p2), (3, p3), (4, p4)))
gripper_control.set_grasp(pub, duration_ms, position)  # 200=open, 600=close

# TF lookup
import tf
tf_listener = tf.TransformListener()
(trans, rot) = tf_listener.lookupTransform('base_link', 'rgbd_cam_color_optical_frame', rospy.Time(0))

# AprilTag detection
from dt_apriltags import Detector
detector = Detector(families='tag36h11', nthreads=8, quad_decimate=2.0, ...)
tags = detector.detect(gray_image, True, [452.533, 452.533, 325.613, 240.352], 0.025)
# tag.pose_R = rotation, tag.pose_t = translation (in camera frame)
# tag.corners = 4 corner pixel coordinates
# tag.center = center pixel coordinate
```

## Jacobian

`JetArm_Jacobian.py` contains a symbolic closed-form Jacobian as function of q1-q4:
- `J_v` (3×6): linear velocity Jacobian
- `J_w` (3×6): angular velocity Jacobian
- Full Jacobian: `J = np.vstack([J_v, J_w])` → 6×6

For 5-DOF manipulability checks, use columns 1-5 only (joint 6 is fixed), giving a 6×5 matrix.
Manipulability: `w = sqrt(det(J @ J.T))` with the 6×5 J.

The numerical constants in the code are derived from link lengths:
e.g., 2157 = l1 * 50000/3 ≈ 12942 * 50000/3

## Challenge 1: Pick & Place — NEARLY COMPLETE

**File:** `pick_and_place_challenge_1.py`

Detect an object via AprilTag, transform pose to base frame, execute pick-and-place.

**Flow:** subscribe /jetarm/object_poses → filter by tag ID → transform camera→base via TF →
average 5 samples → open gripper → approach high (+0.10m) → approach low (+0.05m) →
descend to grasp (-0.02m) → close gripper → retreat → move to place goal → lower → open → retreat

**Config:** pitch=-90 (top-down grasp), servos 1-4 via bus_servo_control, wrist (servo 5) set neutral.

**KNOWN BUG:** If the detection node (e.g., a friend's tag_tracking_v2.py) already publishes
poses in base_link frame to /jetarm/object_poses, then pick_and_place_challenge_1.py will
DOUBLE-TRANSFORM because it calls camera_to_base() on data that's already in base_link.
Check which detection node is running and whether it outputs camera-frame or base-frame poses.

**Status:** Code is functional. TF chain was the last blocker — resolved once tf_hand2camera.launch
provides the static transform. Needs validation on real hardware.

## Challenge 2: Task Space Obstacle Avoidance — NOT STARTED

**Objective:** Move object from start to goal coordinates while avoiding obstacles.
Camera detects ALL tags — obstacle tags give obstacle positions, target tag gives object position.
Plan collision-free path in 2D XY task space using A* algorithm.

**Required deliverables:**
- A* path planner in 2D task space
- Collision-free waypoint generation
- Convert each waypoint to joint space via IK
- Plots of task-space and joint-space trajectories
- Singularity checking at each waypoint: w = sqrt(det(J·Jᵀ))
- Physical demonstration

**Architecture:**
```
Camera → detect all tags (obstacles + target)
       → build obstacle map (XY positions + radii)
       → A* from current EEF to target, avoiding obstacles
       → smooth waypoints
       → for each waypoint:
           IK → joint angles
           Jacobian → manipulability check
           if w < threshold: warn/slow/replan
           execute servo command
       → pick at target
       → A* from target to goal, avoiding obstacles
       → place at goal
```

## Challenge 3: Object Tracking with Vision Feedback — NOT STARTED

**Objective:** Closed-loop visual servoing. Moving AprilTag must stay centered in camera image.

**Control loop:**
```
detect tag in image → pixel error from center (cx=325.613, cy=240.352)
                    → proportional control law (at minimum)
                    → map to EEF velocity/pose delta
                    → IK
                    → command servos
                    → repeat (tight real-time loop)
```

**Jacobian role:** Map image-space velocity to joint-space velocity. Monitor singularities.

**Key file:** `tag_tracking_basic.py` — starter code with AprilTag detection and display.
Currently only prints tag poses; needs control loop added.

## Cross-Challenge Grading Criteria

- Singularity detection/avoidance via Jacobian (ALL challenges)
- Clean TF-based coordinate transforms (base_link → link5 → camera)
- Smooth trajectories (not point-to-point jumps)
- Physical demonstrations on real JetArm
- All challenges share: dt_apriltags detection, camera-to-base TF chain,
  IK via set_pose_target or get_ik, servo control via bus_servo_control

## Current Development State

**Simulation setup on Ubuntu 20.04 / ROS Noetic desktop:**
- Cloned repo, catkin build works
- joints_gui.launch works (URDF + joint sliders)
- demo.launch works (MoveIt planning)
- Camera static TF added, rgbd_cam_color_optical_frame visible in rviz
- Gazebo has joint controllers + ground plane but NO camera plugin, NO AprilTag models

**Next steps (Gazebo simulation — chosen path):**
1. Inspect URDF xacro — confirm no camera exists
2. Add Gazebo camera plugin to link5 at camera mount position
3. Create AprilTag cube models (obstacle + target tags) with textures
4. Add table/surface to Gazebo world
5. Test dt_apriltags detection on simulated camera feed
6. Develop all 3 challenges against this sim environment
7. Deploy to real hardware (expect ~3 hours of hardware time)

## File Reference

Key files in this package:
- `pick_and_place_challenge_1.py` — Challenge 1 node (nearly complete)
- `JetArm_Jacobian.py` — Symbolic Jacobian (J_v, J_w) as f(q1,q2,q3,q4)
- `tag_tracking_basic.py` — Starter code for AprilTag detection + display
- `ik_english.py` — Example IK usage with servo control
- `call_service_sample.py` — Example: call get_current_pose service
- `matrix_calculation_demo.py` — Manual transform computation examples
- `tf_hand2camera.launch` — Static TF: link5 → camera frame
- `execute_joint_angles_import_data_from_file.py` — Replay joint trajectories from file

## Conventions

- All coordinates in metres
- Joint angles in radians (internally), degrees in DH table
- Servo pulse range: 0-1000
- Top-down grasp: pitch = -90°
- Gripper: 200 = open, 600 = closed
- Duration units for servo commands: milliseconds
