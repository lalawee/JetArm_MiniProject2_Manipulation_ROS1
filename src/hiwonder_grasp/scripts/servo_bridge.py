#!/usr/bin/env python3
"""
servo_bridge.py — Simulation only

Bridges pick_place.py servo commands to Gazebo ros_control joint controllers.

pick_place.py publishes hardware-style servo messages:
  /controllers/multi_id_pos_dur  (MultiRawIdPosDur)  — arm joints 1-5
  /controllers/id_pos_dur        (RawIdPosDur)        — gripper (servo id=10)

This node converts those pulse values to radians and forwards them to
Gazebo's ros_control position controllers:
  /jetarm/joint{1-5}_position_controller/command  (Float64)
  /jetarm/r_joint_position_controller/command     (Float64)

Gripper mapping  (r_joint limits: lower=0, upper=1.57 rad):
  pulse 200 (open)  → 1.57 rad
  pulse 600 (closed) → 0.00 rad
"""

import rospy
from std_msgs.msg import Float64
from hiwonder_interfaces.msg import MultiRawIdPosDur, RawIdPosDur
import jetarm_kinematics.transform as transform

# Gripper pulse range from gripper_control.py
GRIPPER_PULSE_OPEN  = 200   # fully open
GRIPPER_PULSE_CLOSE = 600   # fully closed

# Corresponding r_joint angles (URDF: lower=0, upper=1.57)
GRIPPER_ANGLE_OPEN  = 1.57  # rad
GRIPPER_ANGLE_CLOSE = 0.00  # rad

joint_pubs  = {}   # {joint_id (1-5): rospy.Publisher}
gripper_pub = None


def _gripper_pulse_to_angle(pulse):
    """Linear map: pulse [200, 600] → angle [1.57, 0.0] rad."""
    pulse   = max(GRIPPER_PULSE_OPEN, min(GRIPPER_PULSE_CLOSE, int(pulse)))
    ratio   = (pulse - GRIPPER_PULSE_OPEN) / float(GRIPPER_PULSE_CLOSE - GRIPPER_PULSE_OPEN)
    return GRIPPER_ANGLE_OPEN + ratio * (GRIPPER_ANGLE_CLOSE - GRIPPER_ANGLE_OPEN)


def multi_servo_callback(msg):
    """
    Receive MultiRawIdPosDur, extract joints 1-5, convert to radians,
    publish to Gazebo position controllers.
    """
    # Start with neutral pulses so missing joints don't jump
    pulses = [500, 500, 500, 500, 500]

    for servo in msg.id_pos_dur_list:
        sid = int(servo.id)
        if 1 <= sid <= 5:
            pulses[sid - 1] = int(servo.position)

    try:
        angles = transform.pulse2angle(pulses)
    except Exception as e:
        rospy.logwarn_throttle(5.0, "servo_bridge: pulse2angle failed: %s", e)
        return

    for i, angle in enumerate(angles):
        joint_id = i + 1
        if joint_id in joint_pubs:
            msg_out       = Float64()
            msg_out.data  = float(angle)
            joint_pubs[joint_id].publish(msg_out)

    rospy.logdebug("servo_bridge: pulses=%s angles=%s", pulses, list(angles))


def gripper_callback(msg):
    """
    Receive RawIdPosDur for gripper (id=10), convert pulse → angle,
    publish to r_joint_position_controller.
    """
    if int(msg.id) != 10:
        return
    angle         = _gripper_pulse_to_angle(msg.position)
    msg_out       = Float64()
    msg_out.data  = angle
    gripper_pub.publish(msg_out)
    rospy.logdebug("servo_bridge: gripper pulse=%d → %.3f rad", msg.position, angle)


if __name__ == '__main__':
    rospy.init_node('servo_bridge')

    # Publish to Gazebo joint position controllers
    for i in range(1, 6):
        topic = '/jetarm/joint{}_position_controller/command'.format(i)
        joint_pubs[i] = rospy.Publisher(topic, Float64, queue_size=1)

    gripper_pub = rospy.Publisher(
        '/jetarm/r_joint_position_controller/command', Float64, queue_size=1)

    rospy.sleep(0.5)  # wait for publishers to connect

    rospy.Subscriber('/controllers/multi_id_pos_dur', MultiRawIdPosDur,
                     multi_servo_callback, queue_size=5)
    rospy.Subscriber('/controllers/id_pos_dur', RawIdPosDur,
                     gripper_callback, queue_size=5)

    rospy.loginfo("servo_bridge: ready. Bridging /controllers/* → /jetarm/*_controller/command")
    rospy.spin()
