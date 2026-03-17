#!/usr/bin/env python3
"""
joint_initializer.py — Simulation only

Moves the arm to the desired start pose as soon as the Gazebo joint
position controllers come online.

Strategy (avoids the -J teleport race condition):
  1. Robot spawns with all joints at 0 rad (arm pointing straight UP,
     away from the workspace — no risk of hitting the cube).
  2. This node waits for the joint1 controller command topic to appear,
     which means the controllers are running and accepting commands.
  3. Immediately publishes TARGET_Q at 20 Hz for 8 seconds, holding the
     arm at the desired start pose throughout the settle period.

The arm moves smoothly from 0 rad → TARGET_Q under PID control.
Because it starts pointing up (away from the cube), it does NOT sweep
through the workspace during this transition.
"""

import rospy
from std_msgs.msg import Float64

# ── Desired start pose ─────────────────────────────────────────────────────
# [joint1, joint2, joint3, joint4, joint5]  (radians)
TARGET_Q = [0.09, 0.21, -2.1, -0.67, -0.12]

GRIPPER_OPEN = 1.57   # r_joint upper limit

HOLD_DURATION = 8.0   # seconds to keep publishing after controllers appear
PUBLISH_HZ    = 20    # publish rate while holding

# ── Controller topic to wait for ────────────────────────────────────────────
WATCH_TOPIC = '/jetarm/joint1_position_controller/command'

# ──────────────────────────────────────────────────────────────────────────

rospy.init_node('joint_initializer')

# Build publishers
joint_pubs = {}
for i in range(1, 6):
    topic = '/jetarm/joint{}_position_controller/command'.format(i)
    joint_pubs[i] = rospy.Publisher(topic, Float64, queue_size=1)

gripper_pub = rospy.Publisher(
    '/jetarm/r_joint_position_controller/command', Float64, queue_size=1)

# Pre-build messages
target_msgs = []
for q in TARGET_Q:
    m = Float64()
    m.data = q
    target_msgs.append(m)

gripper_msg = Float64()
gripper_msg.data = GRIPPER_OPEN


def publish_target():
    for i, msg in enumerate(target_msgs):
        joint_pubs[i + 1].publish(msg)
    gripper_pub.publish(gripper_msg)


# Wait for the controller topic to appear (= controllers are running).
# We do this by trying to get the list of published topics from the master.
rospy.loginfo("joint_initializer: waiting for controllers (%s)...", WATCH_TOPIC)
while not rospy.is_shutdown():
    published = [t for t, _ in rospy.get_published_topics()]
    if WATCH_TOPIC in published:
        break
    rospy.sleep(0.05)

rospy.loginfo("joint_initializer: controllers online — commanding start pose %s", TARGET_Q)

# Hold the target for HOLD_DURATION seconds
rate  = rospy.Rate(PUBLISH_HZ)
t_end = rospy.Time.now() + rospy.Duration(HOLD_DURATION)
while not rospy.is_shutdown() and rospy.Time.now() < t_end:
    publish_target()
    rate.sleep()

rospy.loginfo("joint_initializer: done")
