#!/usr/bin/env python3
"""
sim_reset.py — Quick simulation reset between pick-and-place test runs.

Does two things without restarting the sim stack:
  1. Teleports the apriltag_cube back to its original spawn position.
  2. Commands the arm joints back to the start pose (same as joint_initializer).

Usage:
  rosrun hiwonder_grasp sim_reset.py
"""

import rospy
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# ── Cube home position (matches world file spawn pose) ──────────────────────
CUBE_MODEL  = 'apriltag_cube'
CUBE_POSE   = {'x': 0.22, 'y': 0.00, 'z': 0.025,
               'rx': 0.0,  'ry': 0.0, 'rz': 0.0, 'rw': 1.0}

# ── Arm start pose (matches joint_initializer TARGET_Q) ─────────────────────
TARGET_Q     = [0.09, 0.21, -2.1, -0.67, -0.12]
GRIPPER_OPEN = 1.57   # r_joint open position

# How long to keep publishing joint commands so the arm settles
HOLD_DURATION = 5.0   # seconds
PUBLISH_HZ    = 20


def reset_cube():
    rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    state = ModelState()
    state.model_name      = CUBE_MODEL
    state.reference_frame = 'world'

    state.pose.position.x    = CUBE_POSE['x']
    state.pose.position.y    = CUBE_POSE['y']
    state.pose.position.z    = CUBE_POSE['z']
    state.pose.orientation.x = CUBE_POSE['rx']
    state.pose.orientation.y = CUBE_POSE['ry']
    state.pose.orientation.z = CUBE_POSE['rz']
    state.pose.orientation.w = CUBE_POSE['rw']

    # Zero velocity so cube doesn't drift after teleport
    state.twist.linear.x  = 0.0
    state.twist.linear.y  = 0.0
    state.twist.linear.z  = 0.0
    state.twist.angular.x = 0.0
    state.twist.angular.y = 0.0
    state.twist.angular.z = 0.0

    resp = set_state(state)
    if resp.success:
        rospy.loginfo("sim_reset: cube teleported to home position")
    else:
        rospy.logwarn("sim_reset: cube reset failed — %s", resp.status_message)


def reset_arm():
    joint_pubs = {}
    for i in range(1, 6):
        topic = '/jetarm/joint{}_position_controller/command'.format(i)
        joint_pubs[i] = rospy.Publisher(topic, Float64, queue_size=1)

    gripper_pub = rospy.Publisher(
        '/jetarm/r_joint_position_controller/command', Float64, queue_size=1)

    # Give publishers time to connect
    rospy.sleep(0.3)

    rospy.loginfo("sim_reset: commanding arm to start pose %s", TARGET_Q)

    rate  = rospy.Rate(PUBLISH_HZ)
    t_end = rospy.Time.now() + rospy.Duration(HOLD_DURATION)

    while not rospy.is_shutdown() and rospy.Time.now() < t_end:
        for i, q in enumerate(TARGET_Q):
            msg = Float64()
            msg.data = q
            joint_pubs[i + 1].publish(msg)

        grip = Float64()
        grip.data = GRIPPER_OPEN
        gripper_pub.publish(grip)

        rate.sleep()

    rospy.loginfo("sim_reset: arm reset complete")


if __name__ == '__main__':
    rospy.init_node('sim_reset')
    reset_cube()
    reset_arm()
    rospy.loginfo("sim_reset: done — ready for next run")
