#!/usr/bin/env python3
"""
singularity.py — JetArm manipulability checker

Uses the closed-form Jacobian (J_v, J_w) derived from the DH parameters.
Manipulability measure:  w = sqrt(det(J_5col @ J_5col.T))
where J_5col is the 6×5 Jacobian (first 5 columns — joint 6 is fixed).

w → 0  :  near or at a singularity
w large :  good dexterity

Usage:
    from singularity import manipulability, is_singular
    w = manipulability(q)          # q = [q1, q2, q3, q4, q5] radians
    if is_singular(q):
        ...
"""

import numpy as np

# Threshold below which the configuration is considered near-singular.
# Tunable — decrease to be more permissive, increase to be stricter.
SINGULARITY_THRESHOLD = 0.003


def _jacobian(q1, q2, q3, q4):
    """
    Closed-form Jacobian for JetArm (from JetArm_Jacobian.py).
    Returns J_v (3×6) and J_w (3×6).
    q5 / q6 terms appear only in the last two Jacobian columns which are
    zero for J_v and very small for J_w — we use only the first 5 columns.
    """
    J_v = np.array([
        [-(3*np.sin(q1)*(1415*np.sin(q2+q3+q4) + 2157*np.sin(q2+q3) + 2157*np.sin(q2)))/50000,
          (3*np.cos(q1)*(1415*np.cos(q2+q3+q4) + 2157*np.cos(q2+q3) + 2157*np.cos(q2)))/50000,
          (6471*np.cos(q1+q2+q3))/100000 + (6471*np.cos(q2-q1+q3))/100000 + (849*np.cos(q1+q2+q3+q4))/20000 + (849*np.cos(q2-q1+q3+q4))/20000,
          (849*np.cos(q1+q2+q3+q4))/20000 + (849*np.cos(q2-q1+q3+q4))/20000,
          0, 0],

        [ (3*np.cos(q1)*(1415*np.sin(q2+q3+q4) + 2157*np.sin(q2+q3) + 2157*np.sin(q2)))/50000,
          (3*np.sin(q1)*(1415*np.cos(q2+q3+q4) + 2157*np.cos(q2+q3) + 2157*np.cos(q2)))/50000,
          (6471*np.sin(q1+q2+q3))/100000 - (6471*np.sin(q2-q1+q3))/100000 + (849*np.sin(q1+q2+q3+q4))/20000 - (849*np.sin(q2-q1+q3+q4))/20000,
          (849*np.sin(q1+q2+q3+q4))/20000 - (849*np.sin(q2-q1+q3+q4))/20000,
          0, 0],

        [ 0,
         -(849*np.sin(q2+q3+q4))/10000 - (6471*np.sin(q2+q3))/50000 - (6471*np.sin(q2))/50000,
         -(849*np.sin(q2+q3+q4))/10000 - (6471*np.sin(q2+q3))/50000,
         -(849*np.sin(q2+q3+q4))/10000,
          0, 0]
    ])

    J_w = np.array([
        [0, -np.sin(q1), -np.sin(q1), -np.sin(q1),
          np.sin(q1+q2+q3+q4)/2 + np.sin(q2-q1+q3+q4)/2,
          np.sin(q1+q2+q3+q4)/2 + np.sin(q2-q1+q3+q4)/2],
        [0,  np.cos(q1),  np.cos(q1),  np.cos(q1),
          np.cos(q2-q1+q3+q4)/2 - np.cos(q1+q2+q3+q4)/2,
          np.cos(q2-q1+q3+q4)/2 - np.cos(q1+q2+q3+q4)/2],
        [1, 0, 0, 0,
          np.cos(q2+q3+q4),
          np.cos(q2+q3+q4)]
    ])

    return J_v, J_w


def manipulability(q):
    """
    Compute manipulability measure w = sqrt(det(J_5col @ J_5col.T)).
    q : array-like [q1, q2, q3, q4, q5] in radians (q5 not used in Jacobian).
    Returns float w >= 0.
    """
    q1, q2, q3, q4 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    J_v, J_w = _jacobian(q1, q2, q3, q4)

    # Full 6×6 Jacobian, then take first 5 columns (joint 6 is fixed)
    J_full = np.vstack([J_v, J_w])   # 6×6
    J5 = J_full[:, :5]               # 6×5  (tall matrix)

    # For a tall matrix (m > n), use J.T @ J (5×5) — J @ J.T (6×6) is always rank-deficient
    det_val = np.linalg.det(J5.T @ J5)
    # Clamp to avoid sqrt of tiny negative from floating point noise
    w = np.sqrt(max(det_val, 0.0))
    return w


def is_singular(q, threshold=SINGULARITY_THRESHOLD):
    """
    Returns True if the configuration q is near a singularity.
    q : [q1, q2, q3, q4, q5] radians.
    """
    return manipulability(q) < threshold
