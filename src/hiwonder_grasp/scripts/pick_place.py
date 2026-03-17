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
import threading
from std_msgs.msg import String
from hiwonder_interfaces.msg import MultiRawIdPosDur, RawIdPosDur
from jetarm_sdk import bus_servo_control, gripper_control
from jetarm_kinematics.kinematics_control import set_pose_target, get_current_pose
import jetarm_kinematics.transform as transform
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from singularity import manipulability, is_singular, SINGULARITY_THRESHOLD
from std_srvs.srv import Trigger

# ============================================================
# CONFIGURATION — Simulation vs Hardware
# ============================================================
# Select at runtime via:  rosrun hiwonder_grasp pick_place.py _sim_mode:=true
# Defaults to hardware mode (sim_mode=false).

_SIM_CONFIG = {
    # Place goal (base frame [x, y, z])
    'PLACE_GOAL':     [0.005, -0.148, 0.001],

    # EEF calibration: compensates IK model vs Gazebo geometry mismatch (~1.5 cm in +x)
    'EEF_OFFSET_X':   0.015,
    'EEF_OFFSET_Y':   0.0,
    'EEF_OFFSET_Z':   0.0,

    # Approach heights (metres relative to tag centre)
    'APPROACH_HIGH':  0.04,
    'APPROACH_LOW':   0.04,
    'GRASP_Z_BELOW':  0.02,

    # Mid-transit waypoint — prevents full-extension swing between pick and place
    'TRANSIT_POS':    [0.18, 0.0, 0.08],

    # Servo durations (ms) — longer in sim to let PID settle
    'DURATION_MOVE':  3500,
    'DURATION_FINE':  3000,
    'DURATION_GRIP':  2500,

    # Home joint angles [q1..q5] radians — must match joint_initializer TARGET_Q
    'HOME_Q':         [0.09, 0.21, -2.1, -0.67, -0.12],

    # Singularity threshold — calibrated to sim's observed w range (0.002–0.004)
    'SINGULARITY_THRESHOLD': 0.003,
}

_HW_CONFIG = {
    # Place goal (base frame [x, y, z]) — tune for real workspace
    'PLACE_GOAL':     [0.125, -0.127, 0.02],

    # No EEF calibration offset needed on real hardware
    'EEF_OFFSET_X':   0.0,
    'EEF_OFFSET_Y':   0.0,
    'EEF_OFFSET_Z':   0.0,

    # Approach heights (metres relative to tag centre)
    'APPROACH_HIGH':  0.06,
    'APPROACH_LOW':   0.02,
    'GRASP_Z_BELOW':  0.02,

    # No mid-transit needed on hardware (real IK handles the workspace better)
    'TRANSIT_POS':    None,

    # Servo durations (ms) — hardware servos need time to physically move
    'DURATION_MOVE':  1500,
    'DURATION_FINE':  1000,
    'DURATION_GRIP':  500,

    # Home joint angles [q1..q5] radians — safe overview pose for real arm
    'HOME_Q':         [0.0, -1.57, 0.0, -1.57, 0.0],

    # Singularity threshold — slightly stricter for real hardware
    'SINGULARITY_THRESHOLD': 0.005,
}

# ── Constants shared by both modes ───────────────────────────────────────────
TARGET_TAG_ID = 1
GRIPPER_OPEN  = 200
GRIPPER_CLOSE = 600
PICK_PITCH    = -90
YAW_RANGE     = [-90, 90]
PITCH_FALLBACKS = [0, +5, -5, +10, -10]
NUM_SAMPLES   = 5

# ── Active config (populated in main from ~sim_mode param) ───────────────────
PLACE_GOAL    = None
EEF_OFFSET_X  = None
EEF_OFFSET_Y  = None
EEF_OFFSET_Z  = None
APPROACH_HIGH = None
APPROACH_LOW  = None
GRASP_Z_BELOW = None
TRANSIT_POS   = None
DURATION_MOVE = None
DURATION_FINE = None
DURATION_GRIP = None
HOME_Q        = None
SIM_MODE      = False

# ============================================================
# GLOBALS
# ============================================================
pose_samples = []
object_pose = None   # averaged [x, y, z] once ready
joints_pub = None
gripper_pub = None
DEBUG_EEF = False    # set via ~debug_eef param in launch file


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


_last_used_pitch = PICK_PITCH   # tracks the pitch accepted on the previous solve_and_move call


def check_reachability(coord, label, pitch=PICK_PITCH):
    """
    Dry-run IK for coord — no servos commanded.
    Prints best manipulability w across all pitch fallbacks so you can
    see upfront whether the target is reachable before committing to a move.
    """
    corrected = [coord[0] - EEF_OFFSET_X,
                 coord[1] - EEF_OFFSET_Y,
                 coord[2] - EEF_OFFSET_Z]
    best_w, best_pitch = -1.0, pitch
    for offset in PITCH_FALLBACKS:
        try_pitch = pitch + offset
        result = set_pose_target(corrected, try_pitch, YAW_RANGE, 1)
        if result is None or result[1] == []:
            continue
        try:
            q = list(transform.pulse2angle(result[1][:5]))
        except Exception:
            continue
        w = manipulability(q)
        if w > best_w:
            best_w, best_pitch = w, try_pitch
        if w >= SINGULARITY_THRESHOLD:
            break
    status = "OK" if best_w >= SINGULARITY_THRESHOLD else "NEAR-SINGULAR"
    rospy.loginfo("  REACHABILITY [%s] target=%s  best_w=%.5f  best_pitch=%.1f°  [%s]",
                  label, [round(c, 4) for c in coord], best_w, best_pitch, status)


def solve_and_move(coord, pitch=PICK_PITCH, duration=DURATION_MOVE, no_fallback=False,
                   dry_run=False):
    """
    Call set_pose_target to get pulse values, check for singularity,
    retry with pitch perturbations if needed, then command servos 1-4.
    Returns the IK result, or None if no valid solution found.

    no_fallback=True: try only the given pitch (no ±offset attempts).
    Use this to keep the same arm configuration between consecutive waypoints.
    dry_run=True: find the best IK solution and record _last_used_pitch but do NOT
    command any servos. Use to pre-solve IK before motion starts.
    """
    global _last_used_pitch
    # Apply EEF calibration correction before solving IK
    corrected = [coord[0] - EEF_OFFSET_X,
                 coord[1] - EEF_OFFSET_Y,
                 coord[2] - EEF_OFFSET_Z]

    best_result  = None
    best_w       = -1.0
    best_pitch   = pitch

    fallbacks = [0] if no_fallback else PITCH_FALLBACKS
    for offset in fallbacks:
        try_pitch = pitch + offset
        result = set_pose_target(corrected, try_pitch, YAW_RANGE, 1)

        if result is None or result[1] == []:
            continue

        # Convert pulses → radians to evaluate manipulability
        pulses = result[1]
        try:
            q = list(transform.pulse2angle(pulses[:5]))
        except Exception:
            continue

        w = manipulability(q)

        if offset == 0:
            rospy.loginfo("  IK pitch=%.1f°  manipulability w=%.5f  (threshold=%.5f)",
                          try_pitch, w, SINGULARITY_THRESHOLD)
        else:
            rospy.loginfo("  Singularity fallback pitch=%.1f°  w=%.5f", try_pitch, w)

        if w > best_w:
            best_w      = w
            best_result = result
            best_pitch  = try_pitch

        if w >= SINGULARITY_THRESHOLD:
            break   # good enough — stop searching

    if best_result is None:
        rospy.logwarn("IK failed for coord=%s (all pitch offsets tried)", coord)
        return None

    _last_used_pitch = best_pitch

    if best_w < SINGULARITY_THRESHOLD:
        rospy.logwarn("  Near-singularity warning: best w=%.5f < threshold=%.5f "
                      "(pitch=%.1f°) — proceeding with best available solution",
                      best_w, SINGULARITY_THRESHOLD, best_pitch)
    else:
        rospy.loginfo("  Accepted: pitch=%.1f°  w=%.5f", best_pitch, best_w)

    if dry_run:
        rospy.loginfo("  dry_run=True — IK solved, pitch locked to %.1f°, no servos commanded",
                      best_pitch)
        return best_result

    servo_data = best_result[1]
    rospy.loginfo("  IK solution (pulses): %s", servo_data)

    bus_servo_control.set_servos(joints_pub, duration,
        ((1, servo_data[0]),
         (2, servo_data[1]),
         (3, servo_data[2]),
         (4, servo_data[3])))
    rospy.sleep(duration / 1000.0 + 0.3)

    return best_result


def set_gripper(position, duration=None):
    """Open or close the gripper via gripper_control."""
    dur = int(DURATION_GRIP if duration is None else duration)
    gripper_control.set_grasp(gripper_pub, dur, position)
    rospy.sleep(dur / 1000.0 + 0.3)


def log_eef(label):
    """Print current EEF xyz from FK service. No-op unless ~debug_eef is true."""
    if not DEBUG_EEF:
        return
    try:
        ok, _, pose = get_current_pose()
        if ok:
            rospy.loginfo("  EEF [%s]: x=%.4f  y=%.4f  z=%.4f",
                          label, pose.position.x, pose.position.y, pose.position.z)
        else:
            rospy.logwarn("  EEF [%s]: FK call returned failure", label)
    except Exception as e:
        rospy.logwarn("  EEF [%s]: FK call failed — %s", label, e)


def _eef_logger_thread():
    """Background thread: logs EEF pose every second. Only started when ~debug_eef is true."""
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        try:
            ok, _, pose = get_current_pose()
            if ok:
                rospy.loginfo("EEF: x=%.4f  y=%.4f  z=%.4f",
                              pose.position.x, pose.position.y, pose.position.z)
        except Exception:
            pass
        rate.sleep()


def set_wrist(angle_pulse=500, duration=500):
    """Set servo 5 (wrist rotation). 500 = neutral."""
    bus_servo_control.set_servos(joints_pub, duration, ((5, angle_pulse),))
    rospy.sleep(duration / 1000.0 + 0.2)


def go_home(duration=1500, keep_gripper=False):
    """Move arm to the home/overview position (HOME_Q joint angles).
    keep_gripper=True: do not command the gripper (use when holding an object)."""
    pulses = transform.angle2pulse([HOME_Q])[0]
    rospy.loginfo("go_home: moving to start pose (pulses: %s)", list(pulses))
    bus_servo_control.set_servos(joints_pub, duration,
        ((1, pulses[0]),
         (2, pulses[1]),
         (3, pulses[2]),
         (4, pulses[3]),
         (5, pulses[4])))
    if not keep_gripper:
        set_gripper(GRIPPER_OPEN, duration=500)
    rospy.sleep(duration / 1000.0 + 0.5)


def sim_attach():
    """Call /sim_grasp/attach if running in simulation (silently skip on hardware)."""
    try:
        attach = rospy.ServiceProxy('/sim_grasp/attach', Trigger)
        resp = attach()
        if resp.success:
            rospy.loginfo("sim_grasp: cube attached to EEF")
        else:
            rospy.logwarn("sim_grasp: attach failed — %s", resp.message)
    except rospy.ServiceException:
        pass  # not in sim — hardware run, skip silently


def sim_detach():
    """Call /sim_grasp/detach if running in simulation (silently skip on hardware)."""
    try:
        detach = rospy.ServiceProxy('/sim_grasp/detach', Trigger)
        resp = detach()
        if resp.success:
            rospy.loginfo("sim_grasp: cube detached")
    except rospy.ServiceException:
        pass  # not in sim — hardware run, skip silently


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
    log_eef("pick-start")

    # In simulation: pre-solve IK BEFORE step 0 to eliminate the fallback-loop
    # drift window (Gazebo PID can cause the arm to sag during 5s of IK calls).
    # On hardware: arm holds position rigidly, so pre-solve is unnecessary overhead.
    check_reachability([x, y, z + APPROACH_HIGH], "approach")
    check_reachability([x, y, z - GRASP_Z_BELOW], "grasp")
    approach_pitch = PICK_PITCH
    if SIM_MODE:
        rospy.loginfo("SIM: Pre-solving approach IK before step 0...")
        pre_result = solve_and_move([x, y, z + APPROACH_HIGH], dry_run=True)
        if pre_result is None:
            rospy.logerr("Pre-solve failed for approach target — aborting pick")
            return False
        approach_pitch = _last_used_pitch
        rospy.loginfo("Pre-solve done: approach pitch=%.1f°", approach_pitch)

    # Step 0 — Open gripper + neutral wrist
    rospy.loginfo("Step 0: Open gripper, neutral wrist")
    set_gripper(GRIPPER_OPEN)
    set_wrist(500)
    log_eef("after-step0")

    # Step 1 — Pre-grasp approach
    rospy.loginfo("Step 1: Approach (%.2fm above tag) → target [%.4f, %.4f, %.4f]",
                  APPROACH_HIGH, x, y, z + APPROACH_HIGH)
    result = solve_and_move([x, y, z + APPROACH_HIGH], pitch=approach_pitch,
                             duration=DURATION_MOVE, no_fallback=SIM_MODE)
    if result is None:
        return False
    log_eef("after-step1-approach")

    # Step 2 — Descend to grasp height
    # Force the same pitch as Step 1 so the IK stays in the same arm configuration
    rospy.loginfo("Step 2: Descend to grasp (%.3fm below tag) → target [%.4f, %.4f, %.4f]",
                  GRASP_Z_BELOW, x, y, z - GRASP_Z_BELOW)
    result = solve_and_move([x, y, z - GRASP_Z_BELOW], pitch=_last_used_pitch,
                             duration=DURATION_FINE, no_fallback=True)
    if result is None:
        return False
    log_eef("after-step2-grasp")

    # Step 3 — Close gripper
    rospy.loginfo("Step 3: Closing gripper")
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(0.5)
    sim_attach()   # lock cube to EEF in simulation
    log_eef("after-step3-grip")

    # Step 4 — Retract to home position (gripper stays closed)
    rospy.loginfo("Step 4: Retract to home position")
    go_home(duration=DURATION_MOVE, keep_gripper=True)
    log_eef("after-step4-home")

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

    check_reachability([x, y, z], "place-goal")
    place_pitch = PICK_PITCH
    if SIM_MODE:
        rospy.loginfo("SIM: Pre-solving place IK...")
        pre_result = solve_and_move([x, y, z], dry_run=True)
        if pre_result is None:
            rospy.logerr("Pre-solve failed for place goal — aborting place")
            return False
        place_pitch = _last_used_pitch
        rospy.loginfo("Pre-solve done: place pitch=%.1f°", place_pitch)
    log_eef("place-start")

    # Step 1 — Transit waypoint (if configured) to avoid arm swinging wildly
    if TRANSIT_POS is not None:
        rospy.loginfo("Step 1a: Transit waypoint → [%.4f, %.4f, %.4f]", *TRANSIT_POS)
        result = solve_and_move(TRANSIT_POS, duration=DURATION_MOVE)
        if result is None:
            rospy.logwarn("Transit waypoint IK failed — skipping transit")
        else:
            log_eef("after-transit")

    # Step 1b — Move to place height
    rospy.loginfo("Step 1b: Move to place height → target [%.4f, %.4f, %.4f]", x, y, z)
    result = solve_and_move([x, y, z], pitch=place_pitch,
                             duration=DURATION_MOVE, no_fallback=SIM_MODE)
    if result is None:
        return False
    log_eef("after-place-step1")

    # Step 2 — Release
    rospy.loginfo("Step 2: Opening gripper — releasing object")
    sim_detach()   # release cube from EEF in simulation
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(0.5)
    log_eef("after-place-step2-release")

    # Step 3 — Retract to home
    rospy.loginfo("Step 3: Retract to home")
    go_home(duration=DURATION_MOVE)
    log_eef("after-place-step3-home")

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

    # ── Select sim or hardware config ────────────────────────────────────────
    SIM_MODE = rospy.get_param('~sim_mode', False)
    cfg = _SIM_CONFIG if SIM_MODE else _HW_CONFIG
    rospy.loginfo("Running in %s mode", "SIMULATION" if SIM_MODE else "HARDWARE")

    PLACE_GOAL    = cfg['PLACE_GOAL']
    EEF_OFFSET_X  = cfg['EEF_OFFSET_X']
    EEF_OFFSET_Y  = cfg['EEF_OFFSET_Y']
    EEF_OFFSET_Z  = cfg['EEF_OFFSET_Z']
    APPROACH_HIGH = cfg['APPROACH_HIGH']
    APPROACH_LOW  = cfg['APPROACH_LOW']
    GRASP_Z_BELOW = cfg['GRASP_Z_BELOW']
    TRANSIT_POS   = cfg['TRANSIT_POS']
    DURATION_MOVE = cfg['DURATION_MOVE']
    DURATION_FINE = cfg['DURATION_FINE']
    DURATION_GRIP = cfg['DURATION_GRIP']
    HOME_Q        = cfg['HOME_Q']
    SINGULARITY_THRESHOLD = cfg['SINGULARITY_THRESHOLD']

    # Initialize TF listener — needs time to fill buffer
    tf_listener = tf.TransformListener()
    rospy.sleep(2.0)  # give TF buffer time to fill

    # Start background EEF logger only when debug_eef is enabled
    DEBUG_EEF = rospy.get_param('~debug_eef', False)
    if DEBUG_EEF:
        rospy.loginfo("debug_eef=True — EEF logging active (1 Hz + step labels)")
        threading.Thread(target=_eef_logger_thread, daemon=True).start()

    # Load target tag ID from rosparam (default: 1)
    TARGET_TAG_ID = rospy.get_param('~target_tag_id', TARGET_TAG_ID)
    rospy.loginfo("Default Target AprilTag ID: %d", TARGET_TAG_ID)

    # Publishers — matching grasp_trajectory.py pattern
    joints_pub = rospy.Publisher('/controllers/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
    gripper_pub = rospy.Publisher('/controllers/id_pos_dur', RawIdPosDur, queue_size=1)

    # Wait for publisher to connect
    rospy.sleep(1.0)

    # Move to home/overview position before doing anything
    go_home()

    POSE_TIMEOUT = rospy.get_param('~pose_timeout', 15.0)   # seconds
    SCAN_DURATION = rospy.get_param('~scan_duration', 3.0)  # seconds

    # ── Scan helper — collect all visible tag IDs from the topic ─────────────
    def scan_for_tags():
        """Listen to /jetarm/object_poses for SCAN_DURATION seconds.
        Returns sorted list of visible tag IDs."""
        seen = set()
        def _cb(msg):
            try:
                for tid, _ in parse_tags(msg.data):
                    seen.add(tid)
            except Exception:
                pass
        sub = rospy.Subscriber('/jetarm/object_poses', String, _cb, queue_size=5)
        rospy.sleep(SCAN_DURATION)
        sub.unregister()
        return sorted(seen)

    while not rospy.is_shutdown():
        # ── Scan to find what tags are visible ───────────────────────────────
        rospy.loginfo("Scanning for visible tags (%.0fs)...", SCAN_DURATION)
        visible = scan_for_tags()

        if not visible:
            rospy.logerr("No tags visible — exiting.")
            break

        print("\nVisible tag IDs: %s" % visible)
        print("Enter tag ID to pick (or Enter to exit): ", end='', flush=True)
        try:
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            break

        try:
            TARGET_TAG_ID = int(user_input)
        except ValueError:
            rospy.logwarn("Invalid tag ID '%s' — exiting.", user_input)
            break

        if TARGET_TAG_ID not in visible:
            rospy.logerr("Tag ID %d not visible — exiting.", TARGET_TAG_ID)
            break

        # ── Subscribe, collect samples, then unsubscribe ─────────────────────
        pose_samples = []
        object_pose  = None
        rospy.loginfo("Collecting %d samples for tag ID %d (timeout: %.0fs)...",
                      NUM_SAMPLES, TARGET_TAG_ID, POSE_TIMEOUT)

        pose_sub = rospy.Subscriber('/jetarm/object_poses', String,
                                    object_pose_callback, queue_size=5)

        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(POSE_TIMEOUT)
        while not rospy.is_shutdown() and object_pose is None:
            if rospy.Time.now() > deadline:
                break
            rate.sleep()

        pose_sub.unregister()

        if object_pose is None:
            rospy.logerr("Failed to collect pose for tag ID %d — exiting.", TARGET_TAG_ID)
            break

        # ── Execute pick & place ──────────────────────────────────────────────
        pick_ok = execute_pick(object_pose)
        if pick_ok:
            execute_place(PLACE_GOAL)

    rospy.loginfo("Done.")
