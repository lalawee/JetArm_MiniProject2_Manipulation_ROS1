#!/usr/bin/env python3
# encoding: utf-8
# Date:2021/08/12
import rospy
import signal
import jetarm_kinematics.transform as transform
from jetarm_kinematics.forward_kinematics import ForwardKinematics
from jetarm_kinematics.inverse_kinematics import get_ik, get_position_ik, set_link, get_link, set_joint_range, get_joint_range
from hiwonder_interfaces.msg import SerialServoMove

from jetarm_sdk import bus_servo_control
from hiwonder_interfaces.msg import MultiRawIdPosDur
import math
import csv

ls_angles_rad=[] # list of arm joint angles
ls_gripper=[] # list of gripper values

# Define the file path (use the correct file path of your saved matrix)
file_path = 'matlab_output.txt'
# Open the text file and read its contents


with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Process each row in the CSV
    for row in reader:
        # Convert each value in the row to a float and apply the transformations
        processed_row = [
            float(row[0]),  # first element (no change)
            float(row[1]) - 1.57,  # second element with -1.57 applied
            float(row[2]),  # third element (no change)
            float(row[3]) - 1.57,  # fourth element with -1.57 applied
            float(row[4]) # fifth element (no change)
        ]
        
        ls_angles_rad.append(processed_row)
        ls_gripper.append(int(row[5]))



rospy.init_node('execute_joint_angles', anonymous=True) #Initialize Node
# set topic publisher
bus_servo_pub = rospy.Publisher('/jetarm_sdk/serial_servo/move', SerialServoMove, queue_size=1)
fk = ForwardKinematics(debug=True)
rospy.wait_for_service('/kinematics/set_pose_target')


print('number of waypoints',len(ls_angles_rad))
for i in range(len(ls_angles_rad)):
	print('waypoint ID',i, len(ls_angles_rad[i]))
	#check joint in range
	#check safety wherever possible

bus_servo_data_detection = False
bus_servo_id_detection = False
def bus_servo_controls(id=0, position=0, duration=0.0):
    # Set the bus servo message type
    data = SerialServoMove()
    data.servo_id = id  # Bus servo ID    
    data.position = position  # Bus servo angle [0-1000]
    data.duration = duration  # Bus servo running time
    bus_servo_pub.publish(data)  # Publish data
    
dt=0.1
dtp=1
for i in range(len(ls_angles_rad)):
	servo_list = transform.angle2pulse(ls_angles_rad)[i]
	servo_list=[math.floor(float(x)) for x in servo_list]

	print("Servo Motor Pulse Value：")
	print(servo_list)  

	print("Converting Pulse Value to angular value in radian：")
	pulse = transform.pulse2angle(servo_list)
	print(pulse)
	res = fk.get_fk(pulse)  #Get the foward kinematics result
	print('foward kinematics solution-coordinates：', res[0])
	print('foward kinematics-quaternion：', res[1])
	print('converting quaternion to rpy：', transform.qua2rpy(res[1]))

	id_list=[1,2,3,4,5]
	for j in id_list:
		bus_servo_controls(id =j,position =servo_list[j-1],duration=500) #publish data, one by one. The robot will response if the pulse value for servo ID 1-5 all published
		rospy.sleep(dt)
		if j == 5: #
			bus_servo_data_detection = True
			break
	bus_servo_controls(id =10,position =ls_gripper[i],duration=500) #publish gripper value
	rospy.sleep(dtp)