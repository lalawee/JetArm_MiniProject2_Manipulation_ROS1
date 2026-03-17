#!/usr/bin/env python3
"""
Fake AprilTag publisher for desktop testing (no camera needed).
Mimics the output of tag_tracking_v2.py by publishing to /jetarm/object_poses.

The positions are in BASE_LINK frame (tag_tracking_v2.py already does
the camera->base transform before publishing).

Usage:
  rosrun <your_package> fake_tag_publisher.py
  or just: python3 fake_tag_publisher.py

You can also publish a TF marker so the tag is visible in rviz.
"""

import rospy
import json
import tf2_ros
import geometry_msgs.msg
from std_msgs.msg import String
from visualization_msgs.msg import Marker


# ============================================================
# CONFIGURATION — edit these to test different scenarios
# ============================================================
# Tag pose in base_link frame (from your real hardware log)
TAGS = [
    {"id": 1, "x": -0.005966, "y": 0.039293, "z": 0.131664},
    # Add more tags here to test multi-tag scenarios:
    # {"id": 2, "x": 0.10, "y": -0.05, "z": 0.02},
]

PUBLISH_RATE = 2.0  # Hz — how often to publish (real node runs at camera fps)


def create_tag_marker(tag, marker_id):
    """Create an rviz Marker so you can see the tag position in the 3D view."""
    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = rospy.Time.now()
    m.ns = "fake_tags"
    m.id = marker_id
    m.type = Marker.CUBE
    m.action = Marker.ADD

    # Position
    m.pose.position.x = tag["x"]
    m.pose.position.y = tag["y"]
    m.pose.position.z = tag["z"]
    m.pose.orientation.w = 1.0

    # Size of the tag block (roughly 2.5cm AprilTag on a small cube)
    m.scale.x = 0.03
    m.scale.y = 0.03
    m.scale.z = 0.03

    # Color: green for tag 1, red for tag 2, blue for tag 3
    colors = {1: (0.0, 1.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 0.0, 1.0)}
    r, g, b = colors.get(tag["id"], (1.0, 1.0, 0.0))
    m.color.r = r
    m.color.g = g
    m.color.b = b
    m.color.a = 0.8

    m.lifetime = rospy.Duration(0)  # persistent
    return m


def create_tag_label(tag, marker_id):
    """Create a text label above the tag marker."""
    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = rospy.Time.now()
    m.ns = "fake_tag_labels"
    m.id = marker_id
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD

    m.pose.position.x = tag["x"]
    m.pose.position.y = tag["y"]
    m.pose.position.z = tag["z"] + 0.04  # slightly above the cube
    m.pose.orientation.w = 1.0

    m.scale.z = 0.02  # text height
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 1.0

    m.text = "Tag %d\n(%.3f, %.3f, %.3f)" % (tag["id"], tag["x"], tag["y"], tag["z"])
    m.lifetime = rospy.Duration(0)
    return m


if __name__ == '__main__':
    rospy.init_node('fake_tag_publisher', anonymous=True)

    # Publisher matching tag_tracking_v2.py output
    poses_pub = rospy.Publisher('/jetarm/object_poses', String, queue_size=1)

    # Marker publisher for rviz visualization
    marker_pub = rospy.Publisher('/fake_tag_markers', Marker, queue_size=10)

    rospy.sleep(1.0)  # let publishers connect

    rate = rospy.Rate(PUBLISH_RATE)
    rospy.loginfo("Fake tag publisher started. Publishing %d tag(s) at %.1f Hz", len(TAGS), PUBLISH_RATE)
    rospy.loginfo("Tags (base_link frame): %s", TAGS)
    rospy.loginfo("Add a Marker display in rviz, topic: /fake_tag_markers")

    while not rospy.is_shutdown():
        # Publish JSON message (same format as tag_tracking_v2.py)
        msg = String()
        msg.data = json.dumps({"tags": TAGS})
        poses_pub.publish(msg)

        # Publish rviz markers
        for i, tag in enumerate(TAGS):
            marker_pub.publish(create_tag_marker(tag, i))
            marker_pub.publish(create_tag_label(tag, i + 100))

        rate.sleep()
