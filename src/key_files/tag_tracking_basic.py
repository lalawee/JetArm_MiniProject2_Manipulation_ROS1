#!/usr/bin/env python3
import os
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from vision_utils import fps, draw_tags
from hiwonder_interfaces.msg import MultiRawIdPosDur
from dt_apriltags import Detector
from jetarm_sdk import bus_servo_control, pid

import rospy
import tf2_ros
import geometry_msgs.msg
import tf.transformations as tft

import tf2_geometry_msgs

class TagTrackingNode:
    def __init__(self):
        rospy.init_node('tag_tracking_node')


        self.at_detector = Detector(searchpath=['apriltags'], 
                                    families='tag36h11',
                                    nthreads=8,
                                    quad_decimate=2.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        self.fps = fps.FPS() # frame rate calculator 
        self.servos_pub = rospy.Publisher('/controllers/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        rospy.sleep(3)
        # robot move to this initial configuration with servo values as assigned
        bus_servo_control.set_servos(self.servos_pub, 1000, ((1, 500), (2, 450), (3, 285), (4, 150), (5, 500), (10, 200)))
        rospy.sleep(2)

        # subscriber camera image topic
        source_image_topic = rospy.get_param('~source_image_topic', '/rgbd_cam/color/image_rect_color')
        rospy.loginfo("Subscribing source image node" + source_image_topic)
        self.image_sub = rospy.Subscriber(source_image_topic, Image, self.image_callback, queue_size=2)
        
    
    def image_callback(self, ros_image):
        #rospy.logdebug('Received an image! ')
        # convert image to opencv format
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        result_image = np.copy(rgb_image)
        tags = self.at_detector.detect(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), True, [452.533, 452.533, 325.613, 240.352], 0.025)
        #True means tracking pose of tags
        #camera model: [fx,fy,cx,cy] is [452.533, 452.533, 325.613, 240.352]. You should replace this value based on your camera calibration data.
        tags = sorted(tags, key=lambda tag: tag.tag_id) # sorted
        draw_tags(result_image, tags, corners_color=(0, 0, 255), center_color=(0, 255, 0))
        
        # Example usage with flexible frame names
        R0=[]
        P0=[]
        camera_matrix = np.array([[452.533, 0, 325.613],
                                  [0, 452.533, 240.352],
                                  [0, 0, 1]], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))  # Assuming no distortion
        tag_size = 0.025  # in meters
                
        if len(tags) > 0:
            for tag in tags:
                if tag.tag_id == 1:
                    print('I am tag1')
                elif tag.tag_id == 2:
                    print('I am tag2')
                elif tag.tag_id == 3:
                    print('I am tag3')
                else:
                    print('I am other tags')
                    
                print(tag.pose_R)# Orientation in camera frame
                print(tag.pose_t)# Position in camera frame
                
                R0=tag.pose_R
                P0=tag.pose_t
                
                corners = tag.corners
                # Draw the detected tag's corners
                result_image = cv2.polylines(result_image, [np.int32(corners)], isClosed=True, color=(0, 255, 0), thickness=2)
                # Draw the coordinate frame (axes) on the tag
                cv2.drawFrameAxes(result_image, camera_matrix, dist_coeffs, R0, P0, 0.025)  # 0.025 is the length of the axes in meters






        self.fps.update()
        self.fps.show_fps(result_image)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("tag_tracking_poses", result_image)
        key = cv2.waitKey(1)


if __name__ == '__main__':
    try:
        tag_tracking = TagTrackingNode()
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))

