#!/usr/bin/env python3

import rospy
import numpy as np
import math
from jetarm_sdk import common
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, EmptyResponse
import threading

import jetarm_kinematics.transform as transform
from jetarm_kinematics.inverse_kinematics import get_ik, set_link, get_link, set_joint_range, get_joint_range
from jetarm_kinematics.forward_kinematics import ForwardKinematics 

from hiwonder_interfaces.srv import SetRobotPose, GetRobotPose, SetLink, GetLink, SetJointValue, SetJointRange, GetJointRange
from hiwonder_interfaces.msg import ServoState, ServoStateList, JointsRange, Link
from jetarm_sdk import controller_client

#from scipy.spatial.transform import Rotation as R
#from spatialmath import *


class KinematicsNode:
    def __init__(self):
        rospy.init_node("kinematics", log_level=rospy.INFO)
        self.controller = controller_client.JointControllerClient(wait_for_service='/jetarm_sdk/serial_servo/ping')
        self.servo_states = rospy.Subscriber('/servo_states', ServoStateList, self.servo_states_callback)
        #self.joints_sub = rospy.Subscriber('/joint_states', JointState, self.servo_state_callback)
        self.robot_dof = rospy.get_param('~robot_dof', 6)
        self.current_joints_state = []
        self.lock = threading.RLock()
        self.fk = ForwardKinematics(debug=False)

        rospy.sleep(0.5)
        self.controller.set_servo(1, 500, 2)
        self.controller.set_servo(2, 560, 2)
        self.controller.set_servo(3, 115,  2)
        self.controller.set_servo(4, 130, 2)
        self.controller.set_servo(5, 500, 2)
        self.controller.set_servo(10, 200, 2)
        rospy.sleep(2)

        rospy.Service('~set_link', SetLink, self.set_link_srv)
        rospy.Service('~get_link', GetLink, self.get_link_srv)
        rospy.Service('~set_joint_range', SetJointRange, self.set_joint_range_srv)
        rospy.Service('~get_joint_range', GetJointRange, self.get_joint_range_srv)
        rospy.Service('~set_pose_target', SetRobotPose, self.set_pose_target)
        rospy.Service('~get_current_pose', GetRobotPose, self.get_current_pose)
        rospy.Service('~set_joint_value_target', SetJointValue, self.set_joint_value_target)
        common.loginfo('kinematics init finish') 
        rospy.set_param('~init_finish', True)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

        rospy.loginfo("Kinematics node is running...")
    
    def set_link_srv(self, msg):
        # 设置link长度(set link length)
        base_link = msg.data.base_link
        link1 = msg.data.link1
        link2 = msg.data.link2
        link3 = msg.data.link3
        end_effector_link = msg.data.end_effector_link
        set_link(base_link, link1, link2, link3, end_effector_link)
        self.fk.set_link(base_link, link1, link2, link3, end_effector_link)

        return [True, 'set_link']

    def get_link_srv(self, msg):
        # 获取各个link长度(get each link length)
        data = get_link()
        data1 = self.fk.get_link()
        link = Link()
        if data == data1:
            link.base_link = data[0]
            link.link1 = data[1]
            link.link2 = data[2]
            link.link3 = data[3]
            link.end_effector_link = data[4]
            return [True, link]
        else:
            return [True, []]

    def set_joint_range_srv(self, msg):
        # 设置关节范围(set joint range)
        joint1 = msg.data.joint1
        joint2 = msg.data.joint2
        joint3 = msg.data.joint3
        joint4 = msg.data.joint4
        joint5 = msg.data.joint5
        set_joint_range([joint1.min, joint1.max], [joint2.min, joint2.max], [joint3.min, joint3.max], [joint4.min, joint4.max], [joint5.min, joint5.max], 'deg')
        self.fk.set_joint_range([joint1.min, joint1.max], [joint2.min, joint2.max], [joint3.min, joint3.max], [joint4.min, joint4.max], [joint5.min, joint5.max], 'deg')

        return [True, 'set_joint_range']

    def get_joint_range_srv(self, msg):
        # 获取各个关节范围(get the range of each joint)
        data = get_joint_range('deg')
        data1 = self.fk.get_joint_range('deg')
        joint_range = JointsRange()
        joint_range.joint1.min = data[0][0]
        joint_range.joint1.max = data[0][1]
        joint_range.joint2.min = data[1][0]
        joint_range.joint2.max = data[1][1]
        joint_range.joint3.min = data[2][0]
        joint_range.joint3.max = data[2][1]
        joint_range.joint4.min = data[3][0]
        joint_range.joint4.max = data[3][1]
        joint_range.joint5.min = data[4][0]
        joint_range.joint5.max = data[4][1]
        if data == data1:
            return [True, joint_range]
        else:
            return [True, []]

    def set_joint_value_target(self, msg):
        # 正运动学解(forward kinematics solution)
        #print(msg)
        joint_value = msg.joint_value
        angle = transform.pulse2angle(joint_value)
        res = self.fk.get_fk(angle)
        pose = Pose() 
        if res:
            pose.position.x = res[0][0]
            pose.position.y = res[0][1]
            pose.position.z = res[0][2]
            pose.orientation = res[1]
            return [True, True, pose]
        else:
            return [True, False, pose]

    def get_current_pose(self, msg):
        # 获取机械臂当前位置(get the current position of robotic arm)
        angle = transform.pulse2angle(self.current_servo_positions)
        res = self.fk.get_fk(angle)
        pose = Pose() 
        if res:
            pose.position.x = res[0][0]
            pose.position.y = res[0][1]
            pose.position.z = res[0][2]
            pose.orientation = res[1]
            return [True, True, pose]
        else:
            return [True, False, pose]

    def servo_states_callback(self, msg):
        # 获取舵机当前角度(get the current angle of servo)
        #rospy.loginfo(msg)
        servo_positions = []
        for i in msg.servo_states:
            servo_positions.append(i.position)
        self.current_servo_positions = np.array(servo_positions[:5]) #[::-1]) 
        #print(self.current_servo_positions)

    def set_pose_target(self, msg):
        # 逆运动学解，获取最优解(所有电机转动最小)(inverse kinematics solution, get the optimal solution)
        position, pitch, pitch_range, resolution = msg.position, msg.pitch, msg.pitch_range, msg.resolution
        position = list(position)

        # t1 = rospy.get_time()
        all_solutions = get_ik(position, pitch, list(pitch_range), resolution)
        # t2 = rospy.get_time()
        # print(t2 - t1)
        # print(all_solutions, self.current_servo_positions)
        if all_solutions != [] and self.current_servo_positions != []:
            rpy = []
            min_d = 1000*5
            optimal_solution = []
            for s in all_solutions:
                pulse_solutions = transform.angle2pulse(s[0])
                try:
                    for i in pulse_solutions:
                        d = np.array(i) - self.current_servo_positions
                        d_abs = np.maximum(d, -d)
                        min_sum = np.sum(d_abs)
                        if min_sum < min_d:
                            min_d = min_sum
                            for k in range(len(i)):
                                if i[k] < 0:
                                    i[k] = 0
                                elif i[k] > 1000:
                                    i[k] = 1000
                            rpy = s[1]
                            optimal_solution = i
                except BaseException as e:
                    print('choose solution error', e)
                    #print(pulse_solutions, current_servo_positions)
                # print(rospy.get_time() - t2)
            return [True, optimal_solution, self.current_servo_positions.tolist(), rpy, min_d]
        else:
            return [True, [], [], [], 0]



if __name__ == '__main__':
    kinematics_node = KinematicsNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.info("Shutting down")
