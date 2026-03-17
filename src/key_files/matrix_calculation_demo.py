import numpy as np
from geometry_msgs.msg import Pose
import tf.transformations as tft


def T44(R,t):
	T = np.hstack((R, t))  # Combine rotation and translation to the left side
	T = np.vstack((T, np.array([0, 0, 0, 1])))  # Add the bottom row for homogeneous coordinates
	return T
def inv_T44(T2):
	R2_inv = T2[:3, :3].T  # Transpose of the rotation matrix
	t2_inv = -np.dot(R2_inv, T2[:3, 3])  # Negated rotated translation

	# Construct the inverse homogeneous transformation matrix
	T2_inv = np.hstack((R2_inv, t2_inv.reshape(3, 1)))  # Add inverse translation
	T2_inv = np.vstack((T2_inv, np.array([0, 0, 0, 1])))  # Add homogeneous row

	return T2_inv

if __name__ == '__main__':
	# Demo: Pose to T44 Homogenous Transformation Matrix
	p1=Pose()
	p1.position.x=0.1
	p1.position.y=0.1
	p1.position.z=0.1
	p1.orientation.w=1			
	print(p1)
	translation = np.array([p1.position.x, 
	p1.position.y, 
	p1.position.z])

	quaternion = np.array([p1.orientation.x,
	p1.orientation.y,
	p1.orientation.z,
	p1.orientation.w,])

	# Convert quaternion to rotation matrix using tf.transformations
	R = tft.quaternion_matrix(quaternion)[:3, :3]  # Get only the 3x3 rotation matrix part
	# Create the homogeneous transformation matrix
	T = np.eye(4)  # Start with the identity matrix
	T[:3, :3] = R  # Set the rotation part
	T[:3, 3] = translation  # Set the translation part
	
	print(T)	
	
	
	## Demo how to convert rotation angle to rotation matrix
	R_camera_in_link5=np.dot(tft.rotation_matrix(np.radians(-90), [0, 0, 1]) , tft.rotation_matrix(-0.1, [1, 0, 0]))[:3, :3]
	print(R_camera_in_link5)

	P_camera_in_link5=np.array([[-0.045],[0],[0.02]])
	R1=R_camera_in_link5
	t1=P_camera_in_link5
	# T_camera_in_link5
	T_camera_in_link5 = T44(R1,t1)
		
	R_eef_in_link5=tft.rotation_matrix(np.radians(-90), [0, 1, 0])[:3, :3]
	print(R_eef_in_link5)
	P_eef_in_link5=np.array([[0.0],[0.0],[0.11054687369]])
	# Now, let's say we have another transformation T2
	R2 = R_eef_in_link5
	t2 = P_eef_in_link5
	T_eef_in_link5 = T44(R2,t2)


	# Matrix multiplication (homogeneous transformation multiplication)
	T_camera_in_eef = np.dot(inv_T44(T_eef_in_link5), T_camera_in_link5)
	print(T_camera_in_eef)	
