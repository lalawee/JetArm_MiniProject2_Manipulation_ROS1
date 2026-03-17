#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Global variable to store the latest image

current_image = None
count=0
# Initialize CvBridge
bridge = CvBridge()

# Define the callback function for mouse clicks
def save_image(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN and current_image is not None:
        # Save the current frame when the left mouse button is clicked
        filename = str(count)+".png"
        cv2.imwrite(filename, current_image)
        print(f"Image saved as {filename}")
        count+=1

def capture_and_save_image():
    rospy.init_node('image_capture_node', anonymous=True)
    
    # Wait for a single message on the /rgbd_cam/color/image_rect_color topic
    rospy.loginfo("Waiting for image message...")
    # Create an OpenCV window and set the mouse callback
    cv2.namedWindow("ROS Image Feed")
    cv2.setMouseCallback("ROS Image Feed", save_image)
        
    while True:
        msg = rospy.wait_for_message('/rgbd_cam/color/image_rect_color', Image)
        
        # Convert the ROS image message to OpenCV format
        try:
            global current_image
            current_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.loginfo("Image received successfully!")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting ROS image: {e}")
            return
    

        # Display the image once received
        cv2.imshow("ROS Image Feed", current_image)
        
        # Wait for a mouse click
        print("Click on the image to save it.")    
        key = cv2.waitKey(1) & 0xFF
        
        # Exit the loop when 'q' is pressed
        if key == ord('q'):
            break
    
    # Close OpenCV window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_image()