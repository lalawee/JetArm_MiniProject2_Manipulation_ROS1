#!/usr/bin/env python3
# encoding: utf-8
# Challenge 2: Task Space Obstacle Avoidance with A*
# --------------------------------------------------
# Scans all visible AprilTags; user selects the target ID.
# Remaining detected tags are treated as obstacles at their base-frame XY positions.
#
# A* plans a collision-free path in 2D XY task space at a fixed transit height.
# Each waypoint is converted to joint space via IK, singularity-checked, then executed.
# After the transit the C1 pick/place sequence runs unchanged.
#
# Produces matplotlib plots saved next to this script:
#   c2_task_space.png   — XY path with obstacle circles (transit 1 preview)
#   c2_all_transits.png — all 3 A* transits overlaid
#   c2_joint_space.png  — joint pulse values across all transit waypoints
#
# Usage (sim):
#   roslaunch hiwonder_grasp sim_pick_place.launch
#   rosrun hiwonder_grasp obstacle_avoidance.py _sim_mode:=true
#
# Offline A* test (no ROS needed):
#   python3 obstacle_avoidance.py --test-astar

import sys
import os
import json
import math
import heapq
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg' if 'DISPLAY' in os.environ else 'Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Directory containing this script — plots are saved here
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Guard: offline A* test must run before rospy import ──────────────────────
_TEST_ASTAR = '--test-astar' in sys.argv

if not _TEST_ASTAR:
    import rospy
    from std_msgs.msg import String
    from hiwonder_interfaces.msg import MultiRawIdPosDur, RawIdPosDur
    from jetarm_sdk import bus_servo_control, gripper_control
    from jetarm_kinematics.kinematics_control import set_pose_target, get_current_pose
    import jetarm_kinematics.transform as transform
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from singularity import manipulability, SINGULARITY_THRESHOLD
    from std_srvs.srv import Trigger
    import tf
    import tf.transformations as tft


# ============================================================
# CONFIGURATION — Simulation vs Hardware
# ============================================================

_SIM_CONFIG = {
    'PLACE_GOAL':            [0.005, -0.148, 0.001],
    'EEF_OFFSET_X':          0.015,
    'EEF_OFFSET_Y':          0.0,
    'EEF_OFFSET_Z':          0.0,
    'APPROACH_HIGH':         0.04,
    'GRASP_Z_BELOW':         0.02,
    'TRANSIT_POS':           [0.18, 0.0, 0.08],
    'DURATION_MOVE':         3500,
    'DURATION_FINE':         3000,
    'DURATION_GRIP':         2500,
    'HOME_Q':                [0.09, 0.21, -2.1, -0.67, -0.12],
    'SINGULARITY_THRESHOLD': 0.003,
}

_HW_CONFIG = {
    'PLACE_GOAL':            [0.125, -0.127, 0.02],
    'EEF_OFFSET_X':          0.0,
    'EEF_OFFSET_Y':          0.0,
    'EEF_OFFSET_Z':          0.0,
    'APPROACH_HIGH':         0.06,
    'GRASP_Z_BELOW':         0.02,
    'TRANSIT_POS':           None,
    'DURATION_MOVE':         1500,
    'DURATION_FINE':         1000,
    'DURATION_GRIP':         500,
    'HOME_Q':                [0.0, -1.57, 0.0, -1.57, 0.0],
    'SINGULARITY_THRESHOLD': 0.005,
}

# ── Shared constants ──────────────────────────────────────────────────────────
GRIPPER_OPEN    = 200
GRIPPER_CLOSE   = 600
PICK_PITCH      = -90
YAW_RANGE       = [-90, 90]
PITCH_FALLBACKS = [0, +5, -5, +10, -10]
NUM_SAMPLES     = 5

# ── Challenge 2 planner constants ────────────────────────────────────────────
OBSTACLE_SAFETY_RADIUS = 0.07   # metres — obstacle exclusion radius
GRID_RESOLUTION        = 0.005  # metres per cell (5 mm)
WAYPOINT_STEP_M        = 0.03   # metres between kept waypoints after smoothing
WORKSPACE_X            = (0.05, 0.35)
WORKSPACE_Y            = (-0.22, 0.22)

# ── Active config (populated in main) ────────────────────────────────────────
PLACE_GOAL    = None
EEF_OFFSET_X  = None
EEF_OFFSET_Y  = None
EEF_OFFSET_Z  = None
APPROACH_HIGH = None
GRASP_Z_BELOW = None
TRANSIT_POS   = None
DURATION_MOVE = None
DURATION_FINE = None
DURATION_GRIP = None
HOME_Q        = None
SIM_MODE      = False

# ── Globals ───────────────────────────────────────────────────────────────────
joints_pub      = None
gripper_pub     = None
tf_listener     = None
_last_used_pitch = PICK_PITCH


# ============================================================
# A* PLANNER
# ============================================================

class AStarPlanner:
    """
    2D grid-based A* planner over the JetArm XY workspace.

    Obstacle positions (base-frame XY) are inflated by safety_radius when
    set_obstacles() is called.  plan() returns the raw cell path as a list of
    real-world (x, y) tuples; smooth_path() downsamples to one waypoint per
    WAYPOINT_STEP_M metres of arc length.
    """

    def __init__(self, x_bounds=WORKSPACE_X, y_bounds=WORKSPACE_Y,
                 resolution=GRID_RESOLUTION):
        self.res   = resolution
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.nx    = int(round((self.x_max - self.x_min) / self.res)) + 1
        self.ny    = int(round((self.y_max - self.y_min) / self.res)) + 1
        self.obstacle_grid       = np.zeros((self.nx, self.ny), dtype=bool)
        self.obstacles           = []   # [(x, y)] stored for plotting
        self._last_safety_radius = OBSTACLE_SAFETY_RADIUS

    # ── coordinate conversion ─────────────────────────────────────────────────

    def _xy_to_grid(self, x, y):
        gx = int(round((x - self.x_min) / self.res))
        gy = int(round((y - self.y_min) / self.res))
        return gx, gy

    def _grid_to_xy(self, gx, gy):
        return (self.x_min + gx * self.res,
                self.y_min + gy * self.res)

    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.nx and 0 <= gy < self.ny

    # ── obstacle setup ────────────────────────────────────────────────────────

    def set_obstacles(self, obstacle_positions, safety_radius=OBSTACLE_SAFETY_RADIUS):
        """Mark cells within safety_radius of each obstacle centre as blocked."""
        self.obstacles = list(obstacle_positions)
        self._last_safety_radius = safety_radius
        r_cells = int(math.ceil(safety_radius / self.res))
        for ox, oy in obstacle_positions:
            ogx, ogy = self._xy_to_grid(ox, oy)
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    gx, gy = ogx + dx, ogy + dy
                    if not self._in_bounds(gx, gy):
                        continue
                    wx, wy = self._grid_to_xy(gx, gy)
                    if math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2) <= safety_radius:
                        self.obstacle_grid[gx, gy] = True

    # ── planning ──────────────────────────────────────────────────────────────

    def plan(self, start_xy, goal_xy):
        """
        Run A* from start_xy to goal_xy.
        Returns list of (x, y) real-world tuples, [] if no path found.
        """
        sx, sy = self._xy_to_grid(*start_xy)
        gx, gy = self._xy_to_grid(*goal_xy)

        # Clamp to grid bounds
        sx = max(0, min(self.nx - 1, sx))
        sy = max(0, min(self.ny - 1, sy))
        gx = max(0, min(self.nx - 1, gx))
        gy = max(0, min(self.ny - 1, gy))

        # If goal is inside an obstacle zone, find nearest free cell
        if self.obstacle_grid[gx, gy]:
            if not _TEST_ASTAR:
                rospy.logwarn("A*: goal cell is inside obstacle zone — nudging to nearest free cell")
            found = False
            for radius in range(1, 30):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) != radius and abs(dy) != radius:
                            continue
                        ngx, ngy = gx + dx, gy + dy
                        if self._in_bounds(ngx, ngy) and not self.obstacle_grid[ngx, ngy]:
                            gx, gy = ngx, ngy
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        # Clear a full safety_radius bubble around start — the arm is physically
        # already there, so no collision is possible regardless of obstacle proximity.
        # This ensures A* can always escape from the home position even if an obstacle
        # tag happens to be nearby.
        clear_r = int(math.ceil(self._last_safety_radius / self.res))
        cleared_cells = []
        for dx in range(-clear_r, clear_r + 1):
            for dy in range(-clear_r, clear_r + 1):
                gx_, gy_ = sx + dx, sy + dy
                if self._in_bounds(gx_, gy_) and self.obstacle_grid[gx_, gy_]:
                    self.obstacle_grid[gx_, gy_] = False
                    cleared_cells.append((gx_, gy_))

        # 8-connected A*
        DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]

        open_heap = []
        h0 = self._heuristic(sx, sy, gx, gy)
        heapq.heappush(open_heap, (h0, 0, (sx, sy)))

        came_from  = {}
        g_score    = {(sx, sy): 0.0}
        visited    = set()

        path = []
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            cx, cy = current
            if cx == gx and cy == gy:
                # Reconstruct
                node = current
                while node in came_from:
                    path.append(self._grid_to_xy(*node))
                    node = came_from[node]
                path.append(self._grid_to_xy(sx, sy))
                path.reverse()
                break

            for dx, dy in DIRS:
                nx_, ny_ = cx + dx, cy + dy
                if not self._in_bounds(nx_, ny_) or self.obstacle_grid[nx_, ny_]:
                    continue
                step_cost = math.sqrt(dx * dx + dy * dy) * self.res
                tg = g_score[current] + step_cost
                neighbor = (nx_, ny_)
                if tg < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tg
                    f = tg + self._heuristic(nx_, ny_, gx, gy)
                    heapq.heappush(open_heap, (f, tg, neighbor))

        for gx_, gy_ in cleared_cells:
            self.obstacle_grid[gx_, gy_] = True

        if not path and not _TEST_ASTAR:
            rospy.logerr("A*: no path found from %s to %s", start_xy, goal_xy)
        return path

    def _heuristic(self, gx1, gy1, gx2, gy2):
        return math.sqrt((gx2 - gx1) ** 2 + (gy2 - gy1) ** 2) * self.res

    # ── path smoothing ────────────────────────────────────────────────────────

    def smooth_path(self, path, step_m=WAYPOINT_STEP_M):
        """
        Downsample raw A* path to waypoints spaced ~step_m metres apart.
        Always includes the first and last point.
        """
        if not path:
            return []
        waypoints = [path[0]]
        accumulated = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            accumulated += math.sqrt(dx * dx + dy * dy)
            if accumulated >= step_m:
                waypoints.append(path[i])
                accumulated = 0.0
        if waypoints[-1] != path[-1]:
            waypoints.append(path[-1])
        return waypoints


# ============================================================
# PLOTTING
# ============================================================

def _draw_obstacle_circles(ax, obstacle_positions, safety_radius):
    """Helper: draw obstacle exclusion circles and centre markers on ax."""
    for i, (ox, oy) in enumerate(obstacle_positions):
        ax.add_patch(plt.Circle((ox, oy), safety_radius, color='red', alpha=0.25))
        ax.plot(ox, oy, 'rx', markersize=9, markeredgewidth=2,
                label='Obstacle' if i == 0 else '_nolegend_')
        ax.annotate(f'obs ({ox:.2f},{oy:.2f})', xy=(ox, oy),
                    xytext=(ox + 0.01, oy + 0.012), fontsize=7, color='darkred')


def plot_task_space(start_xy, goal_xy, obstacle_positions, raw_path, smoothed,
                    safety_radius=OBSTACLE_SAFETY_RADIUS,
                    save_path=os.path.join(SCRIPT_DIR, 'c2_task_space.png')):
    """XY plot: workspace, obstacles, raw A* path, smoothed waypoints (transit 1 preview)."""
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.add_patch(mpatches.Rectangle(
        (WORKSPACE_X[0], WORKSPACE_Y[0]),
        WORKSPACE_X[1] - WORKSPACE_X[0], WORKSPACE_Y[1] - WORKSPACE_Y[0],
        fill=False, edgecolor='gray', linewidth=1.2, linestyle='--', label='Workspace'))

    _draw_obstacle_circles(ax, obstacle_positions, safety_radius)

    if raw_path:
        px, py = zip(*raw_path)
        ax.plot(px, py, 'b-', linewidth=0.8, alpha=0.35, label='A* raw path')
    if smoothed:
        wx, wy = zip(*smoothed)
        ax.plot(wx, wy, 'g-o', linewidth=2.0, markersize=6, label='Smoothed waypoints')

    ax.plot(start_xy[0], start_xy[1], 'go', markersize=13, label='Start (home EEF)',
            markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax.plot(goal_xy[0], goal_xy[1], 'b^', markersize=13, label='Goal (target tag)',
            markeredgecolor='darkblue', markeredgewidth=1.5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Challenge 2: A* Transit 1 — Home → Target', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Task space plot saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


def plot_all_transits(home_xy, target_xy, place_xy, obstacle_positions,
                      paths, safety_radius=OBSTACLE_SAFETY_RADIUS,
                      save_path=os.path.join(SCRIPT_DIR, 'c2_all_transits.png')):
    """
    XY plot showing all 3 A* transits on one axes.
    paths: list of (raw_path, smoothed, color, label) tuples.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(mpatches.Rectangle(
        (WORKSPACE_X[0], WORKSPACE_Y[0]),
        WORKSPACE_X[1] - WORKSPACE_X[0], WORKSPACE_Y[1] - WORKSPACE_Y[0],
        fill=False, edgecolor='gray', linewidth=1.2, linestyle='--', label='Workspace'))

    _draw_obstacle_circles(ax, obstacle_positions, safety_radius)

    for raw_path, smoothed, color, label in paths:
        if raw_path:
            px, py = zip(*raw_path)
            ax.plot(px, py, '-', color=color, linewidth=0.8, alpha=0.3)
        if smoothed:
            wx, wy = zip(*smoothed)
            ax.plot(wx, wy, '-o', color=color, linewidth=2.0, markersize=6, label=label)

    ax.plot(home_xy[0],   home_xy[1],   'go', markersize=13, label='Home',
            markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax.plot(target_xy[0], target_xy[1], 'b^', markersize=13, label='Target (pick)',
            markeredgecolor='darkblue', markeredgewidth=1.5)
    ax.plot(place_xy[0],  place_xy[1],  'ms', markersize=13, label='Place goal',
            markeredgecolor='darkmagenta', markeredgewidth=1.5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Challenge 2: Full A* Trajectory (all transits)', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Full transit plot saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


def plot_joint_space(joint_trajectory, save_path=os.path.join(SCRIPT_DIR, 'c2_joint_space.png')):
    """Time-series of servo pulse values for each joint across all waypoints."""
    if not joint_trajectory:
        print("No joint trajectory data to plot.")
        return
    traj = np.array(joint_trajectory)   # (N_waypoints, 5)
    N = traj.shape[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(min(traj.shape[1], 5)):
        ax.plot(np.arange(N), traj[:, j], '-o', markersize=5, label=f'Joint {j + 1}')
    ax.set_xlabel('Waypoint index', fontsize=12)
    ax.set_ylabel('Servo pulse', fontsize=12)
    ax.set_title('Challenge 2: Joint Space Trajectory', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Joint space plot saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ============================================================
# ROS HELPERS  (copied from pick_place.py with minor changes)
# ============================================================

def camera_to_base(tag_xyz):
    """Transform tag position from camera frame → base_link via ROS TF."""
    try:
        tf_listener.waitForTransform(
            'base_link', 'rgbd_cam_color_optical_frame',
            rospy.Time(0), rospy.Duration(2.0))
        (trans, rot) = tf_listener.lookupTransform(
            'base_link', 'rgbd_cam_color_optical_frame', rospy.Time(0))
        T = tft.quaternion_matrix(rot)
        T[0, 3], T[1, 3], T[2, 3] = trans[0], trans[1], trans[2]
        p_cam  = np.array([tag_xyz[0], tag_xyz[1], tag_xyz[2], 1.0])
        p_base = T @ p_cam
        return p_base[:3].tolist()
    except (tf.LookupException, tf.ConnectivityException,
            tf.ExtrapolationException) as e:
        rospy.logerr("TF lookup failed: %s", e)
        return None


def solve_and_move(coord, pitch=PICK_PITCH, duration=None, no_fallback=False,
                   dry_run=False):
    """
    IK → singularity check → servo command.
    Returns IK result dict or None.  Identical logic to pick_place.py.
    """
    global _last_used_pitch
    if duration is None:
        duration = DURATION_MOVE
    corrected = [coord[0] - EEF_OFFSET_X,
                 coord[1] - EEF_OFFSET_Y,
                 coord[2] - EEF_OFFSET_Z]
    best_result, best_w, best_pitch = None, -1.0, pitch
    fallbacks = [0] if no_fallback else PITCH_FALLBACKS
    for offset in fallbacks:
        try_pitch = pitch + offset
        result = set_pose_target(corrected, try_pitch, YAW_RANGE, 1)
        if result is None or result[1] == []:
            continue
        try:
            q = list(transform.pulse2angle(result[1][:5]))
        except Exception:
            continue
        w = manipulability(q)
        if offset == 0:
            rospy.loginfo("  IK pitch=%.1f°  w=%.5f  (thr=%.5f)",
                          try_pitch, w, SINGULARITY_THRESHOLD)
        else:
            rospy.loginfo("  Fallback pitch=%.1f°  w=%.5f", try_pitch, w)
        if w > best_w:
            best_w, best_result, best_pitch = w, result, try_pitch
        if w >= SINGULARITY_THRESHOLD:
            break

    if best_result is None:
        rospy.logwarn("IK failed for coord=%s", coord)
        return None

    _last_used_pitch = best_pitch
    if best_w < SINGULARITY_THRESHOLD:
        rospy.logwarn("  Near-singularity: w=%.5f < thr — proceeding with best solution",
                      best_w)
    else:
        rospy.loginfo("  Accepted pitch=%.1f°  w=%.5f", best_pitch, best_w)

    if dry_run:
        rospy.loginfo("  dry_run — pitch locked to %.1f°, no servos commanded", best_pitch)
        return best_result

    servo_data = best_result[1]
    bus_servo_control.set_servos(joints_pub, duration,
        ((1, servo_data[0]),
         (2, servo_data[1]),
         (3, servo_data[2]),
         (4, servo_data[3])))
    rospy.sleep(duration / 1000.0 + 0.3)
    return best_result


def set_gripper(position, duration=None):
    dur = int(DURATION_GRIP if duration is None else duration)
    gripper_control.set_grasp(gripper_pub, dur, position)
    rospy.sleep(dur / 1000.0 + 0.3)


def set_wrist(angle_pulse=500, duration=500):
    bus_servo_control.set_servos(joints_pub, duration, ((5, angle_pulse),))
    rospy.sleep(duration / 1000.0 + 0.2)


def go_home(duration=1500, keep_gripper=False):
    pulses = transform.angle2pulse([HOME_Q])[0]
    rospy.loginfo("go_home: pulses=%s", list(pulses))
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
    try:
        resp = rospy.ServiceProxy('/sim_grasp/attach', Trigger)()
        if resp.success:
            rospy.loginfo("sim_grasp: cube attached")
        else:
            rospy.logwarn("sim_grasp: attach failed — %s", resp.message)
    except rospy.ServiceException:
        pass


def sim_detach():
    try:
        resp = rospy.ServiceProxy('/sim_grasp/detach', Trigger)()
        if resp.success:
            rospy.loginfo("sim_grasp: cube detached")
    except rospy.ServiceException:
        pass


# ============================================================
# PICK / PLACE  — vertical-only sequences
# A* handles all horizontal transits; these functions only move
# vertically (descend → act → ascend) at a fixed XY position.
# The arm is assumed to already be at [x, y, transit_z] on entry.
# ============================================================

def get_current_eef_xy():
    """Return current EEF (x, y) from FK, or None on failure."""
    ok, _, pose = get_current_pose()
    if ok:
        return (pose.position.x, pose.position.y)
    rospy.logwarn("get_current_eef_xy: FK call failed")
    return None


def execute_pick_at(tag_pos, transit_z):
    """
    Vertical pick sequence. Arm must already be at [x, y, transit_z].
    Steps: descend → grasp → ascend back to transit_z.
    Returns True on success.
    """
    x, y, z = tag_pos
    rospy.loginfo("=== PICK ===  [%.4f, %.4f, %.4f]  transit_z=%.4f", x, y, z, transit_z)

    # Pre-solve descent IK in sim before commanding (avoids PID drift during solve)
    grasp_pitch = PICK_PITCH
    if SIM_MODE:
        pre = solve_and_move([x, y, z - GRASP_Z_BELOW], dry_run=True)
        if pre is None:
            rospy.logerr("Pre-solve failed for descent — aborting pick")
            return False
        grasp_pitch = _last_used_pitch

    # Descend to grasp height
    rospy.loginfo("Step 1: descend → [%.4f, %.4f, %.4f]", x, y, z - GRASP_Z_BELOW)
    result = solve_and_move([x, y, z - GRASP_Z_BELOW], pitch=grasp_pitch,
                             duration=DURATION_FINE, no_fallback=True)
    if result is None:
        return False

    # Grasp
    rospy.loginfo("Step 2: close gripper")
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(0.5)
    sim_attach()

    # Ascend back to transit height so A* transit 2 can start cleanly
    rospy.loginfo("Step 3: ascend → [%.4f, %.4f, %.4f]", x, y, transit_z)
    solve_and_move([x, y, transit_z], pitch=_last_used_pitch,
                   duration=DURATION_MOVE, no_fallback=True)

    rospy.loginfo("=== PICK COMPLETE ===")
    return True


def execute_place_at(goal_pos, transit_z):
    """
    Vertical place sequence. Arm must already be at [x, y, transit_z].
    Steps: descend → release → ascend back to transit_z.
    Returns True on success.
    """
    x, y, z = goal_pos
    rospy.loginfo("=== PLACE ===  [%.4f, %.4f, %.4f]  transit_z=%.4f", x, y, z, transit_z)

    place_pitch = PICK_PITCH
    if SIM_MODE:
        pre = solve_and_move([x, y, z], dry_run=True)
        if pre is None:
            rospy.logerr("Pre-solve failed for place — aborting")
            return False
        place_pitch = _last_used_pitch

    # Descend to place height
    rospy.loginfo("Step 1: descend → [%.4f, %.4f, %.4f]", x, y, z)
    result = solve_and_move([x, y, z], pitch=place_pitch,
                             duration=DURATION_MOVE, no_fallback=SIM_MODE)
    if result is None:
        return False

    # Release
    rospy.loginfo("Step 2: release")
    sim_detach()
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(0.5)

    # Ascend back to transit height so A* transit 3 can start cleanly
    rospy.loginfo("Step 3: ascend → [%.4f, %.4f, %.4f]", x, y, transit_z)
    solve_and_move([x, y, transit_z], pitch=_last_used_pitch,
                   duration=DURATION_MOVE, no_fallback=True)

    rospy.loginfo("=== PLACE COMPLETE ===")
    return True


# ============================================================
# TAG SCANNING
# ============================================================

def scan_all_tags(duration=3.0):
    """
    Listen to /jetarm/object_poses for `duration` seconds.
    Returns dict {tag_id: [x_cam, y_cam, z_cam]} — camera frame.
    If a tag appears multiple times, keeps the latest reading.
    """
    tag_data = {}

    def _cb(msg):
        try:
            data = json.loads(msg.data)
            for tag in data.get("tags", []):
                tid = int(tag["id"])
                xyz = [float(tag["x"]), float(tag["y"]), float(tag["z"])]
                tag_data[tid] = xyz
        except Exception:
            pass

    sub = rospy.Subscriber('/jetarm/object_poses', String, _cb, queue_size=5)
    rospy.sleep(duration)
    sub.unregister()
    return tag_data


def collect_pose_samples(target_id, num_samples=NUM_SAMPLES, timeout=15.0):
    """
    Collect `num_samples` base-frame poses for `target_id` and return the mean.
    Returns [x, y, z] in base frame, or None on failure.
    """
    samples = []
    done    = threading.Event()

    def _cb(msg):
        if done.is_set():
            return
        try:
            data = json.loads(msg.data)
            for tag in data.get("tags", []):
                if int(tag["id"]) == target_id:
                    xyz_cam  = [float(tag["x"]), float(tag["y"]), float(tag["z"])]
                    xyz_base = camera_to_base(xyz_cam)
                    if xyz_base is not None:
                        samples.append(xyz_base)
                        rospy.loginfo("Sample %d/%d for tag %d: %s",
                                      len(samples), num_samples, target_id, xyz_base)
                    if len(samples) >= num_samples:
                        done.set()
                    break
        except Exception:
            pass

    sub = rospy.Subscriber('/jetarm/object_poses', String, _cb, queue_size=5)
    done.wait(timeout=timeout)
    sub.unregister()

    if not samples:
        return None
    return np.array(samples).mean(axis=0).tolist()


# ============================================================
# TRANSIT EXECUTION
# ============================================================

def execute_transit(smoothed_waypoints, transit_z):
    """
    Move the EEF through all A* waypoints at fixed transit_z.
    Returns joint_trajectory: list of 5-element pulse lists (one per waypoint).
    Waypoints where IK fails are skipped with a warning.
    """
    joint_trajectory = []
    n = len(smoothed_waypoints)
    for i, (wx, wy) in enumerate(smoothed_waypoints):
        rospy.loginfo("Transit %d/%d → [%.4f, %.4f, %.4f]", i + 1, n, wx, wy, transit_z)
        result = solve_and_move([wx, wy, transit_z], duration=DURATION_MOVE)
        if result is None:
            rospy.logwarn("  IK failed for waypoint %d — skipping", i + 1)
            continue
        joint_trajectory.append(list(result[1][:5]))
    return joint_trajectory


# ============================================================
# MAIN
# ============================================================

def main():
    global joints_pub, gripper_pub, tf_listener
    global PLACE_GOAL, EEF_OFFSET_X, EEF_OFFSET_Y, EEF_OFFSET_Z
    global APPROACH_HIGH, GRASP_Z_BELOW, TRANSIT_POS
    global DURATION_MOVE, DURATION_FINE, DURATION_GRIP, HOME_Q
    global SIM_MODE, SINGULARITY_THRESHOLD, _last_used_pitch

    rospy.init_node('obstacle_avoidance', anonymous=True)

    SIM_MODE = rospy.get_param('~sim_mode', False)
    cfg = _SIM_CONFIG if SIM_MODE else _HW_CONFIG
    rospy.loginfo("Running in %s mode", "SIMULATION" if SIM_MODE else "HARDWARE")

    PLACE_GOAL    = cfg['PLACE_GOAL']
    EEF_OFFSET_X  = cfg['EEF_OFFSET_X']
    EEF_OFFSET_Y  = cfg['EEF_OFFSET_Y']
    EEF_OFFSET_Z  = cfg['EEF_OFFSET_Z']
    APPROACH_HIGH = cfg['APPROACH_HIGH']
    GRASP_Z_BELOW = cfg['GRASP_Z_BELOW']
    TRANSIT_POS   = cfg['TRANSIT_POS']
    DURATION_MOVE = cfg['DURATION_MOVE']
    DURATION_FINE = cfg['DURATION_FINE']
    DURATION_GRIP = cfg['DURATION_GRIP']
    HOME_Q        = cfg['HOME_Q']
    SINGULARITY_THRESHOLD = cfg['SINGULARITY_THRESHOLD']

    # Planners params (override via rosparams if desired)
    safety_radius  = rospy.get_param('~safety_radius', OBSTACLE_SAFETY_RADIUS)
    scan_duration  = rospy.get_param('~scan_duration', 3.0)
    pose_timeout   = rospy.get_param('~pose_timeout',  15.0)

    # TF listener
    tf_listener = tf.TransformListener()
    rospy.sleep(2.0)

    # Publishers
    joints_pub  = rospy.Publisher('/controllers/multi_id_pos_dur',
                                  MultiRawIdPosDur, queue_size=1)
    gripper_pub = rospy.Publisher('/controllers/id_pos_dur',
                                  RawIdPosDur, queue_size=1)
    rospy.sleep(1.0)

    # ── Step 1: Go to home ────────────────────────────────────────────────────
    rospy.loginfo("Moving to home position...")
    go_home()

    # ── Step 2: Scan for all visible tags ─────────────────────────────────────
    rospy.loginfo("Scanning for visible tags (%.0fs)...", scan_duration)
    tag_cam_data = scan_all_tags(duration=scan_duration)

    if not tag_cam_data:
        rospy.logerr("No tags visible — check sim is running and cubes are in view.")
        return

    print("\nVisible tag IDs: %s" % sorted(tag_cam_data.keys()))
    print("Enter TARGET tag ID to pick: ", end='', flush=True)
    try:
        user_input = input().strip()
        target_id  = int(user_input)
    except (ValueError, EOFError, KeyboardInterrupt):
        rospy.logwarn("Invalid input — exiting.")
        return

    if target_id not in tag_cam_data:
        rospy.logerr("Tag %d not visible — exiting.", target_id)
        return

    obstacle_cam_data = {tid: xyz for tid, xyz in tag_cam_data.items()
                         if tid != target_id}
    rospy.loginfo("Target tag: %d  |  Obstacles: %s", target_id,
                  sorted(obstacle_cam_data.keys()))

    # ── Step 3: Transform obstacle positions to base frame ────────────────────
    obstacle_positions = []   # [(x_base, y_base)]
    for tid, xyz_cam in obstacle_cam_data.items():
        xyz_base = camera_to_base(xyz_cam)
        if xyz_base is None:
            rospy.logwarn("TF failed for obstacle tag %d — skipping", tid)
            continue
        rospy.loginfo("Obstacle tag %d → base [%.4f, %.4f, %.4f]", tid, *xyz_base)
        obstacle_positions.append((xyz_base[0], xyz_base[1]))

    # ── Step 4: Collect target pose samples ───────────────────────────────────
    rospy.loginfo("Collecting %d pose samples for tag %d...", NUM_SAMPLES, target_id)
    target_pose = collect_pose_samples(target_id, timeout=pose_timeout)
    if target_pose is None:
        rospy.logerr("Failed to collect target pose — exiting.")
        return
    rospy.loginfo("Target pose (base): %s", target_pose)

    # ── Step 5: Get home EEF XY from FK (arm is at home right now) ───────────
    home_xy = get_current_eef_xy()
    if home_xy is None:
        rospy.logwarn("FK failed — using fallback home XY estimate")
        home_xy = (0.15, 0.0)
    rospy.loginfo("Home EEF XY: [%.4f, %.4f]", *home_xy)

    target_xy = (target_pose[0], target_pose[1])
    place_xy  = (PLACE_GOAL[0],  PLACE_GOAL[1])
    transit_z = target_pose[2] + APPROACH_HIGH

    # ── Step 6: Build planner (reused for all 3 transits) ────────────────────
    rospy.loginfo("Building A* planner (safety_radius=%.3f m)...", safety_radius)
    planner = AStarPlanner()
    planner.set_obstacles(obstacle_positions, safety_radius=safety_radius)

    def plan(start, goal, label):
        raw  = planner.plan(start, goal)
        if not raw:
            rospy.logerr("A* found no path for %s — aborting", label)
            return None, None
        smooth = planner.smooth_path(raw)
        rospy.loginfo("%s: %d cells → %d waypoints", label, len(raw), len(smooth))
        return raw, smooth

    # ── Step 7: Plan all 3 transits ──────────────────────────────────────────
    raw1, smooth1 = plan(home_xy,   target_xy, "Transit 1 (home→target)")
    if smooth1 is None:
        return
    raw2, smooth2 = plan(target_xy, place_xy,  "Transit 2 (target→place)")
    if smooth2 is None:
        return
    raw3, smooth3 = plan(place_xy,  home_xy,   "Transit 3 (place→home)")
    if smooth3 is None:
        return

    # ── Step 8: Preview transit 1 plot before execution ──────────────────────
    plot_task_space(home_xy, target_xy, obstacle_positions,
                    raw1, smooth1, safety_radius)

    print("\nPlot generated.  Press Enter to execute (or Ctrl-C to abort): ",
          end='', flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        rospy.loginfo("Aborted by user.")
        return

    # ── Step 9: Open gripper + neutral wrist before first transit ─────────────
    set_gripper(GRIPPER_OPEN)
    set_wrist(500)

    # ── Step 10: TRANSIT 1 — home → above target ─────────────────────────────
    rospy.loginfo("=== TRANSIT 1: home → target (%d waypoints, z=%.4f) ===",
                  len(smooth1), transit_z)
    jt1 = execute_transit(smooth1, transit_z)

    # ── Step 11: PICK (descend → grasp → ascend) ─────────────────────────────
    pick_ok = execute_pick_at(target_pose, transit_z)
    if not pick_ok:
        rospy.logerr("Pick failed — aborting")
        go_home(keep_gripper=False)
        return

    # ── Step 12: TRANSIT 2 — above target → above place goal ─────────────────
    rospy.loginfo("=== TRANSIT 2: target → place (%d waypoints, z=%.4f) ===",
                  len(smooth2), transit_z)
    jt2 = execute_transit(smooth2, transit_z)

    # ── Step 13: PLACE (descend → release → ascend) ───────────────────────────
    execute_place_at(PLACE_GOAL, transit_z)

    # ── Step 14: TRANSIT 3 — above place → home ──────────────────────────────
    rospy.loginfo("=== TRANSIT 3: place → home (%d waypoints, z=%.4f) ===",
                  len(smooth3), transit_z)
    jt3 = execute_transit(smooth3, transit_z)

    # ── Step 15: Final go_home to reset joint configuration ──────────────────
    go_home()

    # ── Step 16: Plots ────────────────────────────────────────────────────────
    all_jt = jt1 + jt2 + jt3
    plot_joint_space(all_jt)

    plot_all_transits(
        home_xy, target_xy, place_xy, obstacle_positions,
        paths=[
            (raw1, smooth1, 'green',  'Transit 1: home → target'),
            (raw2, smooth2, 'blue',   'Transit 2: target → place'),
            (raw3, smooth3, 'orange', 'Transit 3: place → home'),
        ],
        safety_radius=safety_radius)

    rospy.loginfo("Challenge 2 complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    if _TEST_ASTAR:
        # ── Offline A* test — no ROS required ────────────────────────────────
        print("=== Offline A* test ===")
        # start: approximate home EEF XY (arm folded in compact pose)
        # goal:  approximate target tag position
        start  = (0.10, 0.02)
        goal   = (0.22, 0.0)
        obs    = [(0.17, -0.06), (0.20, 0.10)]
        radius = 0.07

        planner = AStarPlanner()
        planner.set_obstacles(obs, safety_radius=radius)

        path     = planner.plan(start, goal)
        smoothed = planner.smooth_path(path)

        print(f"Raw path:        {len(path)} cells")
        print(f"Smoothed path:   {len(smoothed)} waypoints")
        for i, (x, y) in enumerate(smoothed):
            print(f"  wp {i:02d}: [{x:.4f}, {y:.4f}]")

        plot_task_space(start, goal, obs, path, smoothed, radius,
                        save_path=os.path.join(SCRIPT_DIR, 'c2_astar_test.png'))
        sys.exit(0)

    try:
        main()
    except rospy.ROSInterruptException:
        pass
