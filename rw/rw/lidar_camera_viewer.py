#!/usr/bin/env python3

"""
Created/Modified files during execution:
   lidar_camera_projector_direct_transform.py
"""

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from cv_bridge import CvBridge
from math import pi

from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from tf_transformations import quaternion_from_euler, quaternion_matrix

class LidarCameraProjectorNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projector')

        # Declare parameters for user-friendly configuration
        self.declare_parameter('camera_topic', '/camera/image')
        self.declare_parameter('lidar_topic', '/scan/points')

        # Retrieve the actual parameter values
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value

        # Subscribers
        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.camera_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            10
        )

        # For converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Store the last camera frame
        self.last_image = None

        # Store LiDAR points
        self.lidar_points = None

        # --- Direct Transformation Definitions from URDF ---

        # Transformation from base_link to camera_link
        camera_joint_xyz = np.array([-0.483, -0.000313, 0.553294])
        camera_joint_rpy = np.array([0.0, pi / 2, 0.0])
        q = quaternion_from_euler(*camera_joint_rpy)
        self.base_to_camera_transform = self.create_homogeneous_matrix(camera_joint_xyz, q)

        # Transformation from camera_link to camera_link_optical
        camera_optical_joint_xyz = np.array([0.0, 0.0, 0.0])
        camera_optical_joint_rpy = np.array([-pi / 2, pi / 2, pi / 2])
        q = quaternion_from_euler(*camera_optical_joint_rpy)
        self.camera_to_camera_optical_transform = self.create_homogeneous_matrix(camera_optical_joint_xyz, q)

        # Transformation from base_link to laser_frame
        laser_joint_xyz = np.array([-0.48, 0.000058, 0.40056])
        laser_joint_rpy = np.array([pi / 2, 0.0, pi / 2])
        q = quaternion_from_euler(*laser_joint_rpy)
        self.base_to_laser_transform = self.create_homogeneous_matrix(laser_joint_xyz, q)

        # Transformation from laser_frame to lidar_link_optical
        lidar_optical_joint_xyz = np.array([0.0, 0.0, 0.0])
        lidar_optical_joint_rpy = np.array([pi / 2, pi / 2, 0.0])
        q = quaternion_from_euler(*lidar_optical_joint_rpy)
        self.laser_to_lidar_optical_transform = self.create_homogeneous_matrix(lidar_optical_joint_xyz, q)

        # --- Combined Transformation: lidar_link_optical to camera_link_optical ---

        # 1. lidar_link_optical to laser_frame (inverse of laser_to_lidar_optical)
        laser_to_lidar_optical_transform_inv = np.linalg.inv(self.laser_to_lidar_optical_transform)

        # 2. laser_frame to base_link (inverse of base_to_laser)
        base_to_laser_transform_inv = np.linalg.inv(self.base_to_laser_transform)

        # 3. base_link to camera_link
        base_to_camera = self.base_to_camera_transform

        # 4. camera_link to camera_link_optical
        camera_to_camera_optical = self.camera_to_camera_optical_transform

        self.lidar_to_camera_optical_transform = camera_to_camera_optical @ base_to_camera @ base_to_laser_transform_inv @ laser_to_lidar_optical_transform_inv

        # Camera intrinsic parameters (replace with your actual values)
        self.camera_matrix = np.array([[500.0, 0.0, 320.0],
                                   [0.0, 500.0, 240.0],
                                   [0.0, 0.0, 1.0]])
        self.distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Assuming no distortion

        # Timer to periodically show our data (OpenCV display)
        self.timer = self.create_timer(0.1, self.show_views)  # ~10 Hz

        self.get_logger().info('LidarCameraProjectorNode initialized.')

    def create_homogeneous_matrix(self, translation, rotation_quaternion):
        """Creates a 4x4 homogeneous transformation matrix from translation and quaternion."""
        matrix = quaternion_matrix(rotation_quaternion)
        matrix[:3, 3] = translation
        return matrix

    def camera_callback(self, msg: Image):
        """
        Where the camera image is received.
        Convert the incoming sensor_msgs/Image to an OpenCV image.
        """
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera image: {e}")

    def lidar_callback(self, msg: PointCloud2):
        """
        Where the LiDAR data is received.
        Convert the incoming sensor_msgs/PointCloud2 into a list of (x,y,z),
        skipping NaNs and infinite values.
        """
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            # Check for infinite values; skip if not finite
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            points.append((x, y, z))

        # Convert list to a NumPy array, ensuring it's 2D even if empty
        if points:
            self.lidar_points = np.array(points, dtype=np.float64)
        else:
            self.lidar_points = np.empty((0, 3), dtype=np.float64) # Initialize as an empty 2D array

    def show_views(self):
        """
        Timer callback, triggered ~10 times/sec. Display the camera image with
        projected LiDAR points.
        """
        if self.last_image is None or self.lidar_points is None:
            return

        # Make a copy of the image to draw on
        img_with_points = self.last_image.copy()

        # Project LiDAR points onto the image
        lidar_points_homogeneous = np.hstack((self.lidar_points, np.ones((self.lidar_points.shape[0], 1))))

        # Transform points from lidar_link_optical to camera_link_optical
        points_camera_optical_frame = (self.lidar_to_camera_optical_transform @ lidar_points_homogeneous.T).T[:, :3]

        # Project points to image plane
        image_points, _ = cv2.projectPoints(
            points_camera_optical_frame,
            np.zeros(3),  # Rotation vector (already in transform)
            np.zeros(3),  # Translation vector (already in transform)
            self.camera_matrix,
            self.distortion_coeffs
        )

        if image_points is not None:
            for point in image_points.reshape(-1, 2):
                x, y = int(point[0]), int(point[1])
                cv2.circle(img_with_points, (x, y), 2, (0, 255, 0), -1)

        # Show the image with projected points
        cv2.imshow("Camera with LiDAR Projection", img_with_points)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectorNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()