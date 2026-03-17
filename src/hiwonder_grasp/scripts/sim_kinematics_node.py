#!/usr/bin/env python3
"""
sim_kinematics_node.py — Simulation-only kinematics node

Replaces the hardware kinematics_node_6dof.py for Gazebo.
  - No dependency on the aarch64-only inverse_kinematics.so / forward_kinematics.so
  - FK:  URDF-derived FK (matches Gazebo joint angles exactly)
  - IK:  scipy.optimize numerical solver with pitch-range fallback
  - Joint states read from /jetarm/joint_states (Gazebo joint_state_controller)

Services provided (identical interface to hardware node):
  /kinematics/set_pose_target   (hiwonder_interfaces/SetRobotPose)
  /kinematics/get_current_pose  (hiwonder_interfaces/GetRobotPose)

FK derivation (from URDF joint origins):
  joint1: Trans(0,0,L0) · Rz(q1)
  joint2: Rx(+π/2)      · Rz(q2)               [no translation, rpy=(π/2,0,0)]
  joint3: Trans(0,L1,0) · Rz(q3)
  joint4: Trans(0,L2,0) · Rz(q4)
  joint5: Trans(0,L3,0) · Rx(−π/2) · Rz(q5)
  endpt:  Trans(0,0,L4)                          [fixed]

  q_i are the Gazebo joint positions (== transform.pulse2angle() output, radians).
  At neutral pulses [500,500,500,500,500] → q=[0,−π/2,0,−π/2,0]
  → EEF pos ≈ (+0.259, 0, −0.067 m), pitch = −90° (top-down, arm in front).
"""

import rospy
import numpy as np
from math import cos, sin, atan2, sqrt, radians, degrees, pi
from scipy.optimize import minimize
import threading

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from hiwonder_interfaces.srv import SetRobotPose, SetRobotPoseResponse
from hiwonder_interfaces.srv import GetRobotPose, GetRobotPoseResponse
import jetarm_kinematics.transform as transform

# ──────────────────────────────────────────────────────────────────────────────
# Kinematic constants (from URDF)
# ──────────────────────────────────────────────────────────────────────────────
L0 = 0.10314916202   # base_link height (joint1 origin z)
L1 = 0.12941763737   # joint3 origin y  (upper arm)
L2 = 0.12941763737   # joint4 origin y  (forearm)
L3 = 0.05945312631   # joint5 origin y  (wrist)
L4 = 0.11054687369   # endpoint origin z (end-effector / gripper)

# Joint motor-angle limits (degrees, matching jetarm_6dof_params.py)
_LIM_DEG = [(-120.0, 120.0),   # joint1
            (-180.0,   0.0),   # joint2
            (-120.0, 120.0),   # joint3
            (-200.0,  20.0),   # joint4
            (-120.0, 120.0)]   # joint5
JOINT_LIMITS = [(radians(lo), radians(hi)) for lo, hi in _LIM_DEG]

# Neutral joint angles (pulse 500 for all joints)
Q_NEUTRAL = [0.0, -pi/2, 0.0, -pi/2, 0.0]

# Gazebo joint name order
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']


# ──────────────────────────────────────────────────────────────────────────────
# URDF-based Forward Kinematics
# ──────────────────────────────────────────────────────────────────────────────

def _trans(x, y, z):
    T = np.eye(4); T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T

def _Rx(a):
    T = np.eye(4)
    T[1, 1] = cos(a);  T[1, 2] = -sin(a)
    T[2, 1] = sin(a);  T[2, 2] =  cos(a)
    return T

def _Rz(a):
    T = np.eye(4)
    T[0, 0] = cos(a);  T[0, 1] = -sin(a)
    T[1, 0] = sin(a);  T[1, 1] =  cos(a)
    return T


def fk(q):
    """
    Forward kinematics (URDF convention).

    q : [q1, q2, q3, q4, q5] — motor angles in radians.
        These are the values published by /jetarm/joint_states and
        returned by transform.pulse2angle().

    Returns 4×4 homogeneous transform: base_link → endpoint.

    Verified:
      q = [0, -π/2, 0, -π/2, 0] (neutral) → pos ≈ (+0.259, 0, -0.067), pitch = -90°
      q = [0,  0,   0,  0,   0]            → pos = (0, 0, +0.532),        pitch = +90°
    """
    T = np.eye(4)
    T = T @ _trans(0, 0, L0) @ _Rz(q[0])          # joint1
    T = T @ _Rx(pi / 2)      @ _Rz(q[1])           # joint2  (rpy = π/2,0,0)
    T = T @ _trans(0, L1, 0) @ _Rz(q[2])           # joint3
    T = T @ _trans(0, L2, 0) @ _Rz(q[3])           # joint4
    T = T @ _trans(0, L3, 0) @ _Rx(-pi / 2) @ _Rz(q[4])  # joint5 (rpy = -π/2,0,0)
    T = T @ _trans(0, 0, L4)                        # endpoint (fixed)
    return T


def _eef_pitch(T):
    """Pitch angle (degrees) of EEF approach vector (z-column of rotation).
    -90° = pointing straight down (top-down grasp).
    """
    app = T[:3, 2]
    return degrees(atan2(app[2], sqrt(app[0]**2 + app[1]**2)))


def _rot_to_rpy(R):
    """Rotation matrix → [roll, pitch, yaw] in degrees."""
    sy = sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = degrees(atan2(R[2, 1],  R[2, 2]))
        pitch = degrees(atan2(-R[2, 0], sy))
        yaw   = degrees(atan2(R[1, 0],  R[0, 0]))
    else:
        roll  = degrees(atan2(-R[1, 2], R[1, 1]))
        pitch = degrees(atan2(-R[2, 0], sy))
        yaw   = 0.0
    return [roll, pitch, yaw]


# ──────────────────────────────────────────────────────────────────────────────
# Numerical IK
# ──────────────────────────────────────────────────────────────────────────────

# Pre-built set of diverse initial guesses (motor angles, radians)
_INIT_GUESSES = [
    Q_NEUTRAL,
    [0,  radians(-60),  radians(30),  radians(-110), 0],
    [0,  radians(-45),  radians(20),  radians(-120), 0],
    [0,  radians(-50),  radians(10),  radians(-130), 0],
    [0,  radians(-30),  radians(10),  radians(-140), 0],
    [0,  radians(-70),  radians(40),  radians(-100), 0],
    [radians(20),  radians(-55), radians(25), radians(-115), 0],
    [radians(-20), radians(-55), radians(25), radians(-115), 0],
    [radians(30),  radians(-45), radians(15), radians(-125), 0],
    [radians(-30), radians(-45), radians(15), radians(-125), 0],
]


def _ik_single(target_xyz, pitch_deg, q_init, w_pos=1e4, w_pit=200.0):
    """
    Minimise position + pitch error from a single initial guess.
    Returns (q_solution, final_cost) or (None, inf).
    """
    target  = np.array(target_xyz, dtype=float)
    p_tgt   = radians(pitch_deg)

    def cost(q):
        T     = fk(q)
        pos_e = np.linalg.norm(T[:3, 3] - target)
        app   = T[:3, 2]
        p_cur = atan2(app[2], sqrt(app[0]**2 + app[1]**2))
        pit_e = abs(p_cur - p_tgt)
        return w_pos * pos_e**2 + w_pit * pit_e**2

    res = minimize(cost, q_init, method='SLSQP', bounds=JOINT_LIMITS,
                   options={'maxiter': 400, 'ftol': 1e-12})
    return (res.x, res.fun) if res.fun < 1e-3 else (None, float('inf'))


def ik_solve(target_xyz, pitch_deg, pitch_range, resolution, q_current):
    """
    Full IK: try exact pitch first, then sweep pitch_range if needed.

    Returns list of (angle_list, rpy) tuples (same format as original
    get_ik()), or [] if no solution found.
    """
    # Build ordered pitch list: exact first, then range sweep
    pitches = [pitch_deg]
    if pitch_range and len(pitch_range) == 2:
        lo, hi  = sorted(pitch_range)
        step    = max(float(resolution), 2.0)
        p = lo
        while p <= hi + 1e-6:
            if abs(p - pitch_deg) > 1.0:
                pitches.append(p)
            p += step

    all_inits = [q_current] + _INIT_GUESSES
    solutions = []
    seen      = []

    for p in pitches:
        best_q, best_c = None, float('inf')
        for q0 in all_inits:
            q_sol, c = _ik_single(target_xyz, p, list(q0))
            if q_sol is not None and c < best_c:
                best_q, best_c = q_sol, c

        if best_q is None:
            continue

        # Verify position accuracy independently
        T     = fk(best_q)
        pos_e = np.linalg.norm(T[:3, 3] - np.array(target_xyz))
        if pos_e > 0.015:       # >15 mm error — skip
            continue

        # Deduplicate
        is_new = all(np.linalg.norm(best_q - prev) > 0.05 for prev in seen)
        if not is_new:
            continue
        seen.append(np.array(best_q))

        rpy = _rot_to_rpy(T[:3, :3])
        # angle_list must be a list of q-arrays for transform.angle2pulse()
        solutions.append(([list(best_q)], rpy))

        # One good solution is enough for the pick-and-place use case
        if len(solutions) >= 2:
            break

    return solutions


# ──────────────────────────────────────────────────────────────────────────────
# ROS node
# ──────────────────────────────────────────────────────────────────────────────

class SimKinematicsNode:
    def __init__(self):
        rospy.init_node('kinematics')

        self._lock    = threading.Lock()
        self._q_cur   = list(Q_NEUTRAL)   # motor angles (radians)

        rospy.Subscriber('/jetarm/joint_states', JointState,
                         self._joint_states_cb, queue_size=5)

        rospy.Service('/kinematics/set_pose_target',  SetRobotPose,
                      self._set_pose_target)
        rospy.Service('/kinematics/get_current_pose', GetRobotPose,
                      self._get_current_pose)

        rospy.set_param('~init_finish', True)
        rospy.loginfo("sim_kinematics_node: IK/FK services ready  "
                      "(URDF FK + scipy IK, no ARM .so required)")
        rospy.spin()

    # ── Subscriber ────────────────────────────────────────────────────────────

    def _joint_states_cb(self, msg):
        name_pos = dict(zip(msg.name, msg.position))
        with self._lock:
            for i, jname in enumerate(JOINT_NAMES):
                if jname in name_pos:
                    self._q_cur[i] = name_pos[jname]

    # ── Service: set_pose_target ──────────────────────────────────────────────

    def _set_pose_target(self, req):
        position    = list(req.position)
        pitch       = float(req.pitch)
        pitch_range = list(req.pitch_range) if req.pitch_range else [-180.0, 180.0]
        resolution  = float(req.resolution) if req.resolution else 2.0

        with self._lock:
            q_cur = list(self._q_cur)

        solutions = ik_solve(position, pitch, pitch_range, resolution, q_cur)

        resp         = SetRobotPoseResponse()
        resp.success = False

        if not solutions:
            rospy.logwarn("sim_kinematics_node: IK — no solution for "
                          "pos=%s pitch=%.1f°", position, pitch)
            resp.pulse = resp.current_pulse = resp.rpy = []
            resp.min_variation = 0.0
            return resp

        # Current pulses
        cur_pulses = np.array(transform.angle2pulse([q_cur])[0], dtype=float)

        best_pulses    = None
        best_rpy       = []
        best_variation = 1e9

        for angle_list, rpy in solutions:
            for ps in transform.angle2pulse(angle_list):
                ps_arr = np.clip(np.array(ps, dtype=float), 0, 1000)
                delta  = float(np.sum(np.abs(ps_arr - cur_pulses)))
                if delta < best_variation:
                    best_variation = delta
                    best_pulses    = ps_arr.tolist()
                    best_rpy       = rpy

        resp.success       = True
        resp.pulse         = [float(p) for p in best_pulses]
        resp.current_pulse = [float(p) for p in cur_pulses.tolist()]
        resp.rpy           = [float(v) for v in best_rpy]
        resp.min_variation = best_variation

        rospy.loginfo("sim_kinematics_node: IK solved → pulses=%s",
                      [int(p) for p in resp.pulse])
        return resp

    # ── Service: get_current_pose ─────────────────────────────────────────────

    def _get_current_pose(self, req):
        with self._lock:
            q = list(self._q_cur)

        T    = fk(q)
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])

        # Rotation matrix → quaternion (Shepperd's method)
        R  = T[:3, :3]
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / sqrt(tr + 1.0)
            pose.orientation.w = 0.25 / s
            pose.orientation.x = (R[2, 1] - R[1, 2]) * s
            pose.orientation.y = (R[0, 2] - R[2, 0]) * s
            pose.orientation.z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            pose.orientation.w = (R[2, 1] - R[1, 2]) / s
            pose.orientation.x = 0.25 * s
            pose.orientation.y = (R[0, 1] + R[1, 0]) / s
            pose.orientation.z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            pose.orientation.w = (R[0, 2] - R[2, 0]) / s
            pose.orientation.x = (R[0, 1] + R[1, 0]) / s
            pose.orientation.y = 0.25 * s
            pose.orientation.z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            pose.orientation.w = (R[1, 0] - R[0, 1]) / s
            pose.orientation.x = (R[0, 2] + R[2, 0]) / s
            pose.orientation.y = (R[1, 2] + R[2, 1]) / s
            pose.orientation.z = 0.25 * s

        resp          = GetRobotPoseResponse()
        resp.success  = True
        resp.solution = True
        resp.pose     = pose
        return resp


if __name__ == '__main__':
    try:
        SimKinematicsNode()
    except rospy.ROSInterruptException:
        pass
