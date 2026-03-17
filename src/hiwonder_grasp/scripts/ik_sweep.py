#!/usr/bin/env python3
"""
ik_sweep.py — Sweep a grid of XYZ positions and report IK manipulability.

Use this to find good place goal coordinates before committing them to pick_place.py.
Positions with high w (well above SINGULARITY_THRESHOLD) are safe to use.

Usage:
  rosrun hiwonder_grasp ik_sweep.py

Edit the SWEEP_* constants below to change the grid range and step.
"""

import rospy
import numpy as np
import jetarm_kinematics.transform as transform
from jetarm_kinematics.kinematics_control import set_pose_target

# ── Sweep grid ────────────────────────────────────────────────────────────────
X_RANGE  = np.arange(0.10, 0.26, 0.03)   # metres
Y_RANGE  = np.arange(-0.20, 0.21, 0.04)  # metres
Z_VALUES = [0.001, 0.02, 0.04]           # specific heights to check

PITCH        = -90          # degrees — top-down grasp
PITCH_RANGE  = [-100, -80]  # fallback range
RESOLUTION   = 1

SINGULARITY_THRESHOLD = 0.003   # matches sim config

# ── Jacobian (from singularity.py) ───────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from singularity import manipulability


def check(x, y, z):
    """Return (best_w, best_pitch) for position [x,y,z], or (-1, None) on failure."""
    best_w, best_pitch = -1.0, None
    for try_pitch in range(PITCH, PITCH - 11, -1):   # -90 down to -100
        res = set_pose_target([x, y, z], try_pitch, [try_pitch, try_pitch], 1)
        if res is None or res[1] == []:
            continue
        try:
            q = list(transform.pulse2angle(res[1][:5]))
        except Exception:
            continue
        w = manipulability(q)
        if w > best_w:
            best_w, best_pitch = w, try_pitch
        if w >= SINGULARITY_THRESHOLD:
            break
    return best_w, best_pitch


if __name__ == '__main__':
    rospy.init_node('ik_sweep', anonymous=True)
    rospy.loginfo("IK sweep starting — waiting for kinematics service...")
    rospy.wait_for_service('/kinematics/set_pose_target', timeout=10.0)
    rospy.loginfo("Ready. Sweeping grid...")

    results = []

    for z in Z_VALUES:
        for x in X_RANGE:
            for y in Y_RANGE:
                if rospy.is_shutdown():
                    break
                w, pitch = check(x, y, z)
                status = "OK  " if w >= SINGULARITY_THRESHOLD else "SING"
                results.append((w, x, y, z, pitch, status))
                rospy.loginfo("  [%s] x=%+.3f  y=%+.3f  z=%.3f  w=%.5f  pitch=%s",
                              status, x, y, z, w, pitch)

    rospy.loginfo("\n\n=== TOP 20 POSITIONS (highest manipulability) ===")
    results.sort(reverse=True)
    for w, x, y, z, pitch, status in results[:20]:
        rospy.loginfo("  w=%.5f  [%.3f, %.3f, %.3f]  pitch=%d°  [%s]",
                      w, x, y, z, pitch, status)

    rospy.loginfo("\n=== POSITIONS NEAR PLACE ZONE (x<0.18, y<-0.08) ===")
    place_candidates = [(w, x, y, z, pitch) for w, x, y, z, pitch, s in results
                        if x <= 0.18 and y <= -0.08]
    place_candidates.sort(reverse=True)
    for w, x, y, z, pitch in place_candidates[:10]:
        rospy.loginfo("  w=%.5f  [%.3f, %.3f, %.3f]  pitch=%d°  %s",
                      w, x, y, z, pitch,
                      "[OK]" if w >= SINGULARITY_THRESHOLD else "[NEAR-SINGULAR]")

    rospy.loginfo("Sweep complete.")
