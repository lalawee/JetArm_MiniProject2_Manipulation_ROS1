#!/usr/bin/env python3
import os
import json
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
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

        # TF buffer and listener for camera-to-base_link transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_frame = 'rgbd_cam_color_optical_frame'
        self.base_frame = 'base_link'

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
        self.object_poses_pub = rospy.Publisher('/jetarm/object_poses', String, queue_size=1)
        rospy.sleep(3)
        # robot move to this initial configuration with servo values as assigned
        bus_servo_control.set_servos(self.servos_pub, 1000, ((1, 500), (2, 450), (3, 285), (4, 150), (5, 500), (10, 200)))
        rospy.sleep(2)

        # subscriber camera image topic
        source_image_topic = rospy.get_param('~source_image_topic', '/rgbd_cam/color/image_rect_color')
        rospy.loginfo("Subscribing source image node" + source_image_topic)
        self.image_sub = rospy.Subscriber(source_image_topic, Image, self.image_callback, queue_size=2)
        
    
    def transform_to_base(self, x_cam, y_cam, z_cam):
        """
        Transform a point from camera frame to base_link using the TF tree.
        Returns (x, y, z) in base_link frame, or None if TF is unavailable.
        """
        try:
            # Look up transform from camera frame to base_link
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.1))

            # Create a PointStamped in camera frame
            point_cam = geometry_msgs.msg.PointStamped()
            point_cam.header.frame_id = self.camera_frame
            point_cam.header.stamp = rospy.Time(0)
            point_cam.point.x = x_cam
            point_cam.point.y = y_cam
            point_cam.point.z = z_cam

            # Transform to base_link
            point_base = tf2_geometry_msgs.do_transform_point(point_cam, trans)
            return (point_base.point.x, point_base.point.y, point_base.point.z)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, "TF lookup failed (camera->base_link): %s. "
                                  "Ensure base.launch and tf_hand2camera.launch are running." % e)
            return None

    def image_callback(self, ros_image):
        #rospy.logdebug('Received an image! ')
        # convert image to opencv format
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        result_image = np.copy(rgb_image)
        tags = self.at_detector.detect(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), True, [495.5465, 498.9231, 323.0810, 239.4526], 0.025)
        #True means tracking pose of tags
        #camera model: [fx,fy,cx,cy] from calibrate_cam.m calibration results.
        tags = sorted(tags, key=lambda tag: tag.tag_id) # sorted
        draw_tags(result_image, tags, corners_color=(0, 0, 255), center_color=(0, 255, 0))
        
        # Example usage with flexible frame names
        R0=[]
        P0=[]
        camera_matrix = np.array([[495.5465, 0, 323.0810],
                                  [0, 498.9231, 239.4526],
                                  [0, 0, 1]], dtype=np.float32)

        dist_coeffs = np.array([[0.1263], [-0.2577], [0.0], [0.0]], dtype=np.float32)  # [k1, k2, p1, p2] from calibration
        tag_size = 0.025  # in meters
                
        detected_tags = []
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
                
                # Transform tag position from camera frame to base_link for publishing
                x_cam = float(tag.pose_t[0][0])
                y_cam = float(tag.pose_t[1][0])
                z_cam = float(tag.pose_t[2][0])
                base_pos = self.transform_to_base(x_cam, y_cam, z_cam)
                if base_pos is not None:
                    detected_tags.append({
                        "id": int(tag.tag_id),
                        "x": round(base_pos[0], 6),
                        "y": round(base_pos[1], 6),
                        "z": round(base_pos[2], 6)
                    })
                
                corners = tag.corners
                # Draw the detected tag's corners
                result_image = cv2.polylines(result_image, [np.int32(corners)], isClosed=True, color=(0, 255, 0), thickness=2)
                # Draw the coordinate frame (axes) on the tag
                cv2.drawFrameAxes(result_image, camera_matrix, dist_coeffs, R0, P0, 0.025)  # 0.025 is the length of the axes in meters

        # Publish detected tag poses as JSON to /jetarm/object_poses
        if detected_tags:
            msg = String()
            msg.data = json.dumps({"tags": detected_tags})
            self.object_poses_pub.publish(msg)






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

