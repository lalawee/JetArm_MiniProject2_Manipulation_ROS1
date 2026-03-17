#!/usr/bin/env python3
# encoding: utf-8
# Challenge 1: Pick & Place using Visual Perception
# -------------------------------------------------
# Subscribes to /jetarm/object_poses (JSON String) which provides
# detected AprilTag poses in the CAMERA frame.
# Format: {"tags": [{"id": 1, "x": 0.15, "y": 0.0, "z": 0.02}, ...]}
#
# Transforms tag pose from camera frame to base frame via:
#   T_base_tag = T_base_link5 × T_link5_camera × T_camera_tag
# where T_base_link5 comes from /kinematics/get_current_pose service
# and T_link5_camera is the known fixed camera mount offset.
#
# Only picks the object matching TARGET_TAG_ID (rosparam ~target_tag_id).
# Executes a pick-and-place sequence with motion compensation.

import rospy
import json
import math
import numpy as np
from std_msgs.msg import String
from hiwonder_interfaces.msg import MultiRawIdPosDur, RawIdPosDur
from jetarm_sdk import bus_servo_control, gripper_control
from jetarm_kinematics.kinematics_control import set_pose_target

# ============================================================
# CONFIGURATION
# ============================================================
# Place goal in base frame [x, y, z] (metres)
PLACE_GOAL = [0.15, -0.10, 0.02]

# Target AprilTag ID — overridden by rosparam ~target_tag_id
TARGET_TAG_ID = 1

# Gripper pulse values
GRIPPER_OPEN = 200
GRIPPER_CLOSE = 600

# IK parameters
PICK_PITCH = -90        # top-down approach (gripper pointing down)
YAW_RANGE = [-90, 90]   # matching grasp_trajectory.py

# Heights above tag for approach/retreat (metres)
APPROACH_HIGH = 0.10
APPROACH_LOW  = 0.05

# How far below the tag centre to grasp (metres)
# Tag is on top of object — gripper needs to go lower to wrap around it
GRASP_Z_BELOW = 0.02

# Servo motion durations (ms)
DURATION_MOVE = 1500
DURATION_FINE = 1000
DURATION_GRIP = 500

# Number of pose readings to average before picking
NUM_SAMPLES = 5

# ============================================================
# GLOBALS
# ============================================================
pose_samples = []
object_pose = None   # averaged [x, y, z] once ready
joints_pub = None
gripper_pub = None


# ============================================================
# HELPERS
# ============================================================
def parse_tags(msg_str):
    """
    Parse JSON string from /jetarm/object_poses.
    Format: {"tags": [{"id": 1, "x": 0.15, "y": 0.0, "z": 0.02}, ...]}
    Returns list of (tag_id, [x, y, z]) tuples — in CAMERA frame.
    """
    data = json.loads(msg_str)
    results = []
    for tag in data.get("tags", []):
        tag_id = int(tag["id"])
        xyz = [float(tag["x"]), float(tag["y"]), float(tag["z"])]
        results.append((tag_id, xyz))
    return results


# ---- Transform: camera frame → base frame via ROS TF ----
# Requires tf_hand2camera.launch to be running, which publishes
# the static transform from link5 to rgbd_cam_color_optical_frame.
# The robot's internal TF already publishes base_link → link5.
# So TF tree gives us: base_link → link5 → rgbd_cam_color_optical_frame

import tf
import tf.transformations as tft

tf_listener = None  # initialized in main


def camera_to_base(tag_xyz):
    """
    Transform tag position from camera frame to base frame using ROS TF.
    Looks up transform: base_link → rgbd_cam_color_optical_frame
    """
    try:
        # Wait for the transform to be available
        tf_listener.waitForTransform(
            'base_link', 'rgbd_cam_color_optical_frame',
            rospy.Time(0), rospy.Duration(2.0))
        
        # Get the full transform (translation + rotation)
        (trans, rot) = tf_listener.lookupTransform(
            'base_link', 'rgbd_cam_color_optical_frame', rospy.Time(0))
        
        # Build 4x4 transform matrix
        T_base_camera = tft.quaternion_matrix(rot)
        T_base_camera[0, 3] = trans[0]
        T_base_camera[1, 3] = trans[1]
        T_base_camera[2, 3] = trans[2]
        
        # Transform tag position
        p_cam = np.array([tag_xyz[0], tag_xyz[1], tag_xyz[2], 1.0])
        p_base = T_base_camera @ p_cam
        
        result = p_base[:3].tolist()
        rospy.loginfo("  Camera frame: %s -> Base frame: [%.4f, %.4f, %.4f]",
                      tag_xyz, result[0], result[1], result[2])
        return result
    except (tf.LookupException, tf.ConnectivityException, 
            tf.ExtrapolationException) as e:
        rospy.logerr("TF lookup failed: %s", e)
        return None


def solve_and_move(coord, pitch=PICK_PITCH, duration=DURATION_MOVE):
    """
    Call set_pose_target to get pulse values, then command servos 1-4.
    Matches grasp_trajectory.py pattern — only moves joints 1-4.
    Returns the IK result, or None if failed.
    """
    result = set_pose_target(coord, pitch, YAW_RANGE, 1)

    # result[1] is the servo pulse list; empty list means IK failed
    if result is None or result[1] == []:
        rospy.logwarn("IK failed for coord=%s pitch=%s", coord, pitch)
        return None

    servo_data = result[1]
    rospy.loginfo("  IK solution (pulses): %s", servo_data)

    # Command servos 1-4 simultaneously (same as grasp_trajectory)
    bus_servo_control.set_servos(joints_pub, duration,
        ((1, servo_data[0]),
         (2, servo_data[1]),
         (3, servo_data[2]),
         (4, servo_data[3])))
    rospy.sleep(duration / 1000.0 + 0.3)

    return result


def set_gripper(position, duration=DURATION_GRIP):
    """Open or close the gripper via gripper_control."""
    gripper_control.set_grasp(gripper_pub, duration, position)
    rospy.sleep(duration / 1000.0 + 0.3)


def set_wrist(angle_pulse=500, duration=500):
    """Set servo 5 (wrist rotation). 500 = neutral."""
    bus_servo_control.set_servos(joints_pub, duration, ((5, angle_pulse),))
    rospy.sleep(duration / 1000.0 + 0.2)


# ============================================================
# PICK SEQUENCE
# ============================================================
def execute_pick(tag_pos):
    """
    Top-down pick sequence.
    tag_pos: [x, y, z] of the object in base frame.
    """
    x, y, z = tag_pos

    rospy.loginfo("=== PICK SEQUENCE START ===")
    rospy.loginfo("Tag position (base): [%.4f, %.4f, %.4f]", x, y, z)

    # Step 0 — Open gripper + neutral wrist
    rospy.loginfo("Step 0: Open gripper, neutral wrist")
    set_gripper(GRIPPER_OPEN)
    set_wrist(500)

    # Step 1 — Move above tag (high)
    rospy.loginfo("Step 1: Approach high (%.2fm above)", APPROACH_HIGH)
    result = solve_and_move([x, y, z + APPROACH_HIGH], duration=DURATION_MOVE)
    if result is None:
        return False

    # Step 2 — Descend to low approach
    rospy.loginfo("Step 2: Approach low (%.2fm above)", APPROACH_LOW)
    result = solve_and_move([x, y, z + APPROACH_LOW], duration=DURATION_FINE)
    if result is None:
        return False

    # Step 3 — Descend to grasp height (below tag to wrap around object)
    rospy.loginfo("Step 3: Descend to grasp (%.3fm below tag)", GRASP_Z_BELOW)
    result = solve_and_move([x, y, z - GRASP_Z_BELOW], duration=DURATION_FINE)
    if result is None:
        return False

    # Step 4 — Close gripper
    rospy.loginfo("Step 4: Closing gripper")
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(0.5)

    # Step 5 — Retreat (lift up)
    rospy.loginfo("Step 5: Retreat")
    result = solve_and_move([x, y, z + APPROACH_HIGH], duration=DURATION_MOVE)
    if result is None:
        return False

    rospy.loginfo("=== PICK COMPLETE ===")
    return True


# ============================================================
# PLACE SEQUENCE
# ============================================================
def execute_place(goal_pos):
    """
    Top-down place sequence — mirrors pick.
    goal_pos: [x, y, z] of the place target in base frame.
    """
    x, y, z = goal_pos

    rospy.loginfo("=== PLACE SEQUENCE START ===")
    rospy.loginfo("Goal position (base): [%.4f, %.4f, %.4f]", x, y, z)

    # Step 1 — Approach high above goal
    rospy.loginfo("Step 1: Approach goal high")
    result = solve_and_move([x, y, z + APPROACH_HIGH], duration=DURATION_MOVE)
    if result is None:
        return False

    # Step 2 — Lower to place height
    rospy.loginfo("Step 2: Lower to place height")
    result = solve_and_move([x, y, z], duration=DURATION_FINE)
    if result is None:
        return False

    # Release
    rospy.loginfo("Step 3: Opening gripper — releasing object")
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(0.5)

    # Step 4 — Retreat
    rospy.loginfo("Step 4: Retreat from goal")
    result = solve_and_move([x, y, z + APPROACH_HIGH], duration=DURATION_MOVE)
    if result is None:
        return False

    rospy.loginfo("=== PLACE COMPLETE ===")
    return True


# ============================================================
# POSE CALLBACK — accumulate samples then trigger pick & place
# ============================================================
def object_pose_callback(msg):
    global pose_samples, object_pose

    # If we already have a locked pose, ignore further messages
    if object_pose is not None:
        return

    try:
        tags = parse_tags(msg.data)
    except Exception as e:
        rospy.logwarn("Failed to parse tags: %s — %s", msg.data, e)
        return

    # Find the target tag in this message and transform to base frame
    for tag_id, pos_cam in tags:
        if tag_id == TARGET_TAG_ID:
            # Transform from camera frame to base frame
            pos_base = camera_to_base(pos_cam)
            if pos_base is None:
                rospy.logwarn("Skipping sample — transform failed")
                break
            pose_samples.append(pos_base)
            rospy.loginfo("Tag %d — sample %d/%d (base): %s",
                          tag_id, len(pose_samples), NUM_SAMPLES, pos_base)
            break

    if len(pose_samples) >= NUM_SAMPLES:
        # Average the samples
        arr = np.array(pose_samples)
        object_pose = arr.mean(axis=0).tolist()
        rospy.loginfo("Averaged object pose (tag %d): %s", TARGET_TAG_ID, object_pose)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    rospy.init_node('pick_and_place', anonymous=True)

    # Initialize TF listener — needs time to fill buffer
    tf_listener = tf.TransformListener()
    rospy.sleep(2.0)  # give TF buffer time to fill

    # Load target tag ID from rosparam (default: 1)
    TARGET_TAG_ID = rospy.get_param('~target_tag_id', TARGET_TAG_ID)
    rospy.loginfo("Target AprilTag ID: %d", TARGET_TAG_ID)

    # Publishers — matching grasp_trajectory.py pattern
    joints_pub = rospy.Publisher('/controllers/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
    gripper_pub = rospy.Publisher('/controllers/id_pos_dur', RawIdPosDur, queue_size=1)

    # Wait for publisher to connect
    rospy.sleep(1.0)

    # Subscribe to object poses
    rospy.Subscriber('/jetarm/object_poses', String, object_pose_callback)

    rospy.loginfo("Waiting for %d object pose samples...", NUM_SAMPLES)

    # Spin until we have enough samples
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and object_pose is None:
        rate.sleep()

    if object_pose is None:
        rospy.logerr("Shutting down — no object pose received")
    else:
        # Execute pick
        pick_ok = execute_pick(object_pose)

        if pick_ok:
            # Execute place
            execute_place(PLACE_GOAL)

    rospy.loginfo("Done.")
