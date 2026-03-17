#!/usr/bin/env python3
"""
sim_tag_publisher.py — Simulation only

Reads the AprilTag cube's position from Gazebo (/gazebo/model_states),
transforms it from base_link frame to the camera frame using TF, then
publishes it to /jetarm/object_poses in the same JSON format that
tag_tracking_v2.py would produce.

This means pick_place.py's camera_to_base() will correctly transform
the position BACK to base_link — no double-transform.

The cube's position in Gazebo world frame == base_link frame because the
robot URDF is fixed at the world origin.

Usage:
  Run alongside the Gazebo sim + tf_hand2camera.launch + kinematics node.
  Drag the 'apriltag_cube' model in Gazebo (press T to translate), then
  re-run pick_place.py to test a new pick position.

Published:
  /jetarm/object_poses  (std_msgs/String)  JSON: {"tags": [{"id":1, "x":..., ...}]}

Subscribed:
  /gazebo/model_states  (gazebo_msgs/ModelStates)
"""

import json
import rospy
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates

# Mapping: Gazebo model name → AprilTag ID
CUBE_MODELS = {
    'apriltag_cube':   1,   # target (orange top)
    'apriltag_cube_2': 2,   # obstacle (blue top)
    'apriltag_cube_3': 3,   # obstacle (green top) — added for Challenge 2
}

# TF frame names
BASE_FRAME   = 'base_link'
CAMERA_FRAME = 'rgbd_cam_color_optical_frame'

# How often to publish (Hz).  pick_place.py needs 5 samples; 5 Hz gives a
# clean 1-second acquisition at startup.
PUBLISH_RATE = 5.0


class SimTagPublisher:
    def __init__(self):
        rospy.init_node('sim_tag_publisher')

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.poses_pub = rospy.Publisher('/jetarm/object_poses', String, queue_size=1)

        # Latest model poses in world (= base_link) frame: {model_name: (x,y,z)}
        self._model_pos = {}

        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self._model_states_cb, queue_size=2)

        rospy.loginfo("sim_tag_publisher: tracking models: %s",
                      {m: t for m, t in CUBE_MODELS.items()})
        rospy.loginfo("sim_tag_publisher: waiting for TF: %s → %s",
                      BASE_FRAME, CAMERA_FRAME)

    def _model_states_cb(self, msg):
        """Cache positions for all tracked models."""
        for model_name in CUBE_MODELS:
            if model_name in msg.name:
                idx = msg.name.index(model_name)
                p = msg.pose[idx].position
                self._model_pos[model_name] = (p.x, p.y, p.z)

    def _transform_base_to_camera(self, x_base, y_base, z_base):
        """
        Transform a point from base_link to the camera frame using TF.
        Returns (x, y, z) in camera frame, or None if TF unavailable.
        """
        try:
            # Look up transform: camera ← base_link
            trans = self.tf_buffer.lookup_transform(
                CAMERA_FRAME, BASE_FRAME, rospy.Time(0), rospy.Duration(0.1))

            point = geometry_msgs.msg.PointStamped()
            point.header.frame_id = BASE_FRAME
            point.header.stamp    = rospy.Time(0)
            point.point.x = x_base
            point.point.y = y_base
            point.point.z = z_base

            point_cam = tf2_geometry_msgs.do_transform_point(point, trans)
            return (point_cam.point.x, point_cam.point.y, point_cam.point.z)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0,
                "sim_tag_publisher: TF lookup failed (%s→%s): %s. "
                "Is tf_hand2camera.launch running?", BASE_FRAME, CAMERA_FRAME, e)
            return None

    def spin(self):
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            tags = []
            for model_name, tag_id in CUBE_MODELS.items():
                if model_name not in self._model_pos:
                    continue
                xb, yb, zb = self._model_pos[model_name]
                cam_pos = self._transform_base_to_camera(xb, yb, zb)
                if cam_pos is None:
                    continue
                xc, yc, zc = cam_pos
                tags.append({"id": tag_id,
                             "x": round(xc, 6),
                             "y": round(yc, 6),
                             "z": round(zc, 6)})
                rospy.logdebug(
                    "sim_tag_publisher: tag%d base=(%.4f,%.4f,%.4f) "
                    "→ camera=(%.4f,%.4f,%.4f)",
                    tag_id, xb, yb, zb, xc, yc, zc)

            if tags:
                msg = String()
                msg.data = json.dumps({"tags": tags})
                self.poses_pub.publish(msg)

            rate.sleep()


if __name__ == '__main__':
    try:
        node = SimTagPublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass
