import rospy
import numpy as np
from hiwonder_interfaces.srv import GetRobotPose  # Import the service type
import tf.transformations as tft
	

class GetEEFPoseNode:
    def __init__(self):
        rospy.init_node('sample_node')  

    def get_eef_pose_base(self):
        # Wait for the service to be available
        rospy.wait_for_service('/kinematics/get_current_pose')
        
        try:
            # Create a ServiceProxy to call the service
            get_jetarm_current_pose = rospy.ServiceProxy('/kinematics/get_current_pose', GetRobotPose)
            
            # Call the service (pass any required parameters)
            response = get_jetarm_current_pose()  # Modify if the service takes arguments
        
            # Print or handle the response
            print("Current Pose: ", response.pose)
            
            translation = np.array([response.pose.position.x, 
                response.pose.position.y, 
                response.pose.position.z])

            quaternion = np.array([response.pose.orientation.x,
                response.pose.orientation.y,
                response.pose.orientation.z,
                response.pose.orientation.w,])
            yaw, pitch, roll = tft.euler_from_quaternion(quaternion.tolist())
            # Convert quaternion to rotation matrix using tf.transformations
            R = tft.quaternion_matrix(quaternion)[:3, :3]  # Get only the 3x3 rotation matrix part	
            # Create the homogeneous transformation matrix
            T = np.eye(4)  # Start with the identity matrix
            T[:3, :3] = R  # Set the rotation part
            T[:3, 3] = translation  # Set the translation part
            return T,yaw,pitch,roll
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)            

if __name__ == '__main__':
	geteef=GetEEFPoseNode()
	geteef.get_eef_pose_base()

