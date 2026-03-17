# hiwonder_grasp — JetArm MiniProject 2

ROS Noetic package for the JetArm 5-DOF manipulator.
Covers Challenge 1 (Pick & Place) and Challenge 2 (Obstacle Avoidance with A*).

---

## Prerequisites

- ROS Noetic + Gazebo installed
- Package built: `catkin build hiwonder_grasp`
- All terminals sourced: `source devel/setup.bash`

---

## Challenge 1 — Pick & Place

### Simulation

**Terminal 1 — launch the sim stack:**
```bash
roslaunch hiwonder_grasp sim_pick_place.launch
```
Wait ~5 seconds for Gazebo, controllers, TF, and the tag publisher to fully start.

**Terminal 2 — run the pick & place script:**
```bash
rosrun hiwonder_grasp pick_place.py _sim_mode:=true
```

**What happens:**
1. Arm moves to home position
2. Script scans for visible AprilTags (3 s window) and lists their IDs
3. Enter the tag ID you want to pick (e.g. `1`)
4. Script collects 5 pose samples, averages them, then executes:
   - Approach above the tag → descend → grasp → retract to home
   - Transit to place goal → release → retract to home
5. Loops — ready for the next pick

**Reset between runs:**
```bash
rosrun hiwonder_grasp sim_reset.py
```
Teleports all cubes back to their spawn positions and commands the arm back to the start pose.

**Move the target cube to a new pick position:**
1. In Gazebo, press `T` (translate tool) and drag `apriltag_cube` to a new position
2. Re-run Terminal 2 — the script will detect the new position automatically

### Hardware

```bash
rosrun hiwonder_grasp pick_place.py _sim_mode:=false
```
Ensure the AprilTag detection node is running separately and publishing to `/jetarm/object_poses`.

---

## Challenge 2 — Task Space Obstacle Avoidance (A*)

### Scene layout

| Model | Tag ID | Colour | Role | Position (base frame) |
|-------|--------|--------|------|-----------------------|
| `apriltag_cube` | 1 | Orange top | **Target** (pick this) | [0.22, 0.00] |
| `apriltag_cube_2` | 2 | Blue top | Obstacle | [0.20, 0.10] |
| `apriltag_cube_3` | 3 | Green top | Obstacle (blocks direct path) | [0.17, −0.06] |

### Simulation

**Terminal 1 — launch the sim stack:**
```bash
roslaunch hiwonder_grasp sim_obstacle_avoidance.launch
```
Wait ~5 seconds.

**Terminal 2 — run the obstacle avoidance script:**
```bash
roslaunch hiwonder_grasp obstacle_avoidance.launch sim_mode:=true
```

**What happens:**
1. Arm moves to home position
2. Script scans for all visible AprilTags (3 s window)
3. Enter the target tag ID (e.g. `1`) — all other detected tags become obstacles
4. A* plans a collision-free path in the XY workspace, avoiding 7 cm safety circles around each obstacle
5. A task-space plot is shown (`/tmp/c2_task_space.png`) — press Enter to execute
6. Arm transits through all A* waypoints at a fixed safe height above the workspace
7. A joint-space trajectory plot is saved (`/tmp/c2_joint_space.png`)
8. C1 pick sequence runs at the target position (descend → grasp → retract)
9. C1 place sequence runs (move to goal → release → retract)

**Reset between runs:**
```bash
rosrun hiwonder_grasp sim_reset.py
```

### Offline A* test (no ROS required)

Verify the planner and generate a sample plot without launching Gazebo:
```bash
/usr/bin/python3 scripts/obstacle_avoidance.py --test-astar
# Plot saved to /tmp/c2_astar_test.png
```

### Output plots

| File | Content |
|------|---------|
| `/tmp/c2_task_space.png` | XY workspace with obstacle circles, raw A* path (blue), smoothed waypoints (green), start and goal markers |
| `/tmp/c2_joint_space.png` | Servo pulse values for joints 1–5 across all transit waypoints |

### Tunable parameters (via rosparams)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~safety_radius` | `0.07` | Obstacle exclusion radius in metres |
| `~scan_duration` | `3.0` | Tag scan window in seconds |
| `~pose_timeout` | `15.0` | Max wait for pose samples in seconds |

Example:
```bash
roslaunch hiwonder_grasp obstacle_avoidance.launch sim_mode:=true safety_radius:=0.06
```

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/pick_place.py` | Challenge 1 — pick & place |
| `scripts/obstacle_avoidance.py` | Challenge 2 — A* obstacle avoidance + pick & place |
| `scripts/sim_reset.py` | Reset all cubes and arm between sim runs |
| `scripts/sim_tag_publisher.py` | Reads Gazebo model positions → `/jetarm/object_poses` |
| `scripts/servo_bridge.py` | Converts servo pulses → Gazebo joint Float64 commands |
| `scripts/sim_grasp_node.py` | Simulated grasp attach/detach via Gazebo fixed joints |

## Key launch files

| Launch file | Purpose |
|-------------|---------|
| `launch/sim_pick_place.launch` | Sim stack for Challenge 1 |
| `launch/obstacle_avoidance.launch` | Launches obstacle_avoidance.py (sim or hardware) |
