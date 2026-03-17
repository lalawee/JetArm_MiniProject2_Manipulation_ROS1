#!/usr/bin/env python3
"""
gripper_mimic_node.py

Gazebo does not honour <mimic> joint tags natively.
This node subscribes to the r_joint command and republishes
to each mimic joint controller with the correct multiplier.

Mimic relationships (from gripper.urdf.xacro):
  l_joint    : multiplier = -1  (mirrors r_joint)
  r_in_joint : multiplier = -1
  r_out_joint: multiplier = +1
  l_in_joint : multiplier = -1
  l_out_joint: multiplier = +1
"""

import rospy
from std_msgs.msg import Float64

MIMIC_JOINTS = [
    ('l_joint',     -1.0),
    ('r_in_joint',  -1.0),
    ('r_out_joint',  1.0),
    ('l_in_joint',  -1.0),
    ('l_out_joint',  1.0),
]

pubs = {}

def r_joint_cb(msg):
    for joint_name, multiplier in MIMIC_JOINTS:
        out = Float64()
        out.data = msg.data * multiplier
        pubs[joint_name].publish(out)

if __name__ == '__main__':
    rospy.init_node('gripper_mimic_node')

    for joint_name, _ in MIMIC_JOINTS:
        topic = '/jetarm/{}_position_controller/command'.format(joint_name)
        pubs[joint_name] = rospy.Publisher(topic, Float64, queue_size=1)

    rospy.Subscriber(
        '/jetarm/r_joint_position_controller/command',
        Float64,
        r_joint_cb,
        queue_size=1)

    rospy.loginfo("gripper_mimic_node: running — mirroring r_joint to %d finger joints",
                  len(MIMIC_JOINTS))
    rospy.spin()
