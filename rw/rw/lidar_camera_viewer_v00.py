#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class LidarCameraViewerNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_viewer')

        # Declare parameters for user-friendly configuration
        self.declare_parameter('camera_topic', '/camera/image')
        self.declare_parameter('lidar_topic', '/scan/points')

        # Retrieve the actual parameter values
        self.camera_topic = self.get_parameter('camera_topic')\
                                .get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_topic')\
                                .get_parameter_value().string_value

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

        # Store LiDAR points for display
        self.lidar_points = []

        # Timer to periodically show our data (OpenCV display)
        self.timer = self.create_timer(0.1, self.show_views)  # ~10 Hz

        self.get_logger().info('LidarCameraViewerNode initialized.')

    def camera_callback(self, msg: Image):
        """
        Where the camera image is received.
        Convert the incoming sensor_msgs/Image to an OpenCV image.
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_image = cv_img
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

        self.lidar_points = points

    def show_views(self):
        """
        Timer callback, triggered ~10 times/sec. Display the camera image and
        a simple overhead LiDAR view, skipping invalid points.
        """
        # Show camera image
        if self.last_image is not None:
            cv2.imshow("Camera View", self.last_image)

        # Show LiDAR data
        self.draw_lidar_view(self.lidar_points)

        # Update OpenCV GUI
        cv2.waitKey(1)

    def draw_lidar_view(self, points):
        """
        Displays a top-down 2D representation of LiDAR data in a 600x600 image.
        """
        lidar_img = np.zeros((600, 600, 3), dtype=np.uint8)

        if not points:
            cv2.imshow("LiDAR View", lidar_img)
            return

        scale = 20.0
        offset_x, offset_y = 300, 300

        for (x, y, z) in points:
            # Optionally clamp out-of-range values
            if abs(x) > 1e4 or abs(y) > 1e4:
                continue

            px = int(x * scale + offset_x)
            py = int(-y * scale + offset_y)

            # Draw a small circle for each valid point within the image
            if 0 <= px < 600 and 0 <= py < 600:
                cv2.circle(lidar_img, (px, py), 1, (255, 255, 255), -1)

        cv2.imshow("LiDAR View", lidar_img)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraViewerNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()