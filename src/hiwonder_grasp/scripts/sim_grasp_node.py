#!/usr/bin/env python3
"""
sim_grasp_node.py — Mock grasp attachment for Gazebo simulation.

Gazebo has no friction-based grasping. This node simulates a "fixed joint"
between the gripper and the cube by continuously moving the cube to follow
the EEF TF frame at the offset computed when attach was called.

Services:
  /sim_grasp/attach  (std_srvs/Trigger) — lock cube to EEF
  /sim_grasp/detach  (std_srvs/Trigger) — release cube

pick_place.py calls these after closing / before opening the gripper.
On real hardware the services won't exist — pick_place.py silently skips them.
"""

import rospy
import tf
import numpy as np
import threading

from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

CUBE_MODELS = ['apriltag_cube', 'apriltag_cube_2']
EEF_FRAME   = 'link5'        # last active arm link — stable TF reference
BASE_FRAME  = 'base_link'
UPDATE_HZ   = 20


class SimGraspNode:
    def __init__(self):
        rospy.init_node('sim_grasp_node')

        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)  # let TF buffer fill

        # Gazebo service for moving the cube
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10.0)
        self._set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Latest cube poses from Gazebo (world frame): {model_name: (pos, quat)}
        self._cube_poses = {}   # {str: (np.array[3], list[4])}
        self._active_cube = None  # model name currently attached
        self._model_sub  = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self._model_states_cb, queue_size=1)

        # Attachment state
        self._attached       = False
        self._offset_pos     = None   # cube position in EEF frame at attach time
        self._update_thread  = None
        self._stop_event     = threading.Event()

        rospy.Service('/sim_grasp/attach', Trigger, self._attach_cb)
        rospy.Service('/sim_grasp/detach', Trigger, self._detach_cb)

        rospy.loginfo("sim_grasp_node: ready — /sim_grasp/attach and /sim_grasp/detach")
        rospy.spin()

    # ── Gazebo model states callback ─────────────────────────────────────────
    def _model_states_cb(self, msg):
        for model_name in CUBE_MODELS:
            if model_name in msg.name:
                idx = msg.name.index(model_name)
                p = msg.pose[idx].position
                o = msg.pose[idx].orientation
                self._cube_poses[model_name] = (
                    np.array([p.x, p.y, p.z]),
                    [o.x, o.y, o.z, o.w]
                )

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _get_eef_world_pos(self):
        """Look up current EEF position in world (base_link) frame."""
        try:
            (trans, _) = self.tf_listener.lookupTransform(
                BASE_FRAME, EEF_FRAME, rospy.Time(0))
            return np.array(trans)
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return None

    def _move_cube(self, pos, quat):
        """Teleport the cube to pos/quat in world frame."""
        state = ModelState()
        state.model_name = self._active_cube
        state.reference_frame = 'world'
        state.pose.position.x = float(pos[0])
        state.pose.position.y = float(pos[1])
        state.pose.position.z = float(pos[2])
        state.pose.orientation.x = float(quat[0])
        state.pose.orientation.y = float(quat[1])
        state.pose.orientation.z = float(quat[2])
        state.pose.orientation.w = float(quat[3])
        state.twist.linear.x  = 0.0
        state.twist.linear.y  = 0.0
        state.twist.linear.z  = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        try:
            self._set_state(state)
        except Exception:
            pass

    # ── Follow loop ───────────────────────────────────────────────────────────
    def _follow_loop(self):
        """Run at UPDATE_HZ — moves active cube to follow EEF while attached."""
        rate = rospy.Rate(UPDATE_HZ)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            eef_pos = self._get_eef_world_pos()
            if (eef_pos is not None and self._offset_pos is not None
                    and self._active_cube is not None
                    and self._active_cube in self._cube_poses):
                _, quat = self._cube_poses[self._active_cube]
                cube_pos = eef_pos + self._offset_pos
                self._move_cube(cube_pos, quat)
            rate.sleep()

    # ── Service callbacks ─────────────────────────────────────────────────────
    def _attach_cb(self, req):
        if not self._cube_poses:
            return TriggerResponse(success=False, message="No cube poses received yet")

        eef_pos = self._get_eef_world_pos()
        if eef_pos is None:
            return TriggerResponse(success=False, message="EEF TF not available")

        # Attach to whichever cube is closest to the EEF
        closest, closest_dist = None, float('inf')
        for model_name, (pos, _) in self._cube_poses.items():
            d = np.linalg.norm(pos - eef_pos)
            if d < closest_dist:
                closest_dist, closest = d, model_name

        self._active_cube = closest
        cube_pos, _ = self._cube_poses[closest]
        self._offset_pos = cube_pos - eef_pos
        self._attached   = True

        # Start follow thread
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self._update_thread.start()

        rospy.loginfo("sim_grasp_node: attached '%s' (dist=%.3fm), offset=%s",
                      closest, closest_dist, self._offset_pos)
        return TriggerResponse(success=True, message="Attached")

    def _detach_cb(self, req):
        self._stop_event.set()
        self._attached    = False
        self._offset_pos  = None
        self._active_cube = None
        rospy.loginfo("sim_grasp_node: detached — cube released")
        return TriggerResponse(success=True, message="Detached")


if __name__ == '__main__':
    SimGraspNode()
