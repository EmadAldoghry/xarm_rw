#!/usr/bin/env python3

# Created/Modified files during execution:
#   lidar_camera_fusion_node.py

import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_py as tf2

import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped


def strip_to_xyz_cloud(cloud_in: PointCloud2) -> PointCloud2:
    """
    Create a new PointCloud2 that has only the x,y,z fields from 'cloud_in'.
    The order of fields must be x,y,z (type FLOAT32, offsets 0,4,8),
    with point_step = 12 bytes (3 floats).
    """
    # Read all points from the original cloud (skip nans).
    # field_names must exist in the message fields.
    points = pc2.read_points_list(cloud_in, field_names=["x", "y", "z"], skip_nans=True)

    # Create a new PointCloud2
    stripped_cloud = PointCloud2()
    stripped_cloud.header = cloud_in.header

    # Define just x,y,z fields
    stripped_cloud.height = 1
    stripped_cloud.width = len(points)
    stripped_cloud.fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    stripped_cloud.is_bigendian = False
    stripped_cloud.point_step = 12  # 3 floats * 4 bytes each
    stripped_cloud.row_step = stripped_cloud.point_step * stripped_cloud.width
    stripped_cloud.is_dense = True

    # Convert the list of points to a numpy array of float32
    if stripped_cloud.width > 0:
        xyz_array = np.array(points, dtype=np.float32)
        stripped_cloud.data = xyz_array.tobytes()
    else:
        stripped_cloud.data = b''

    return stripped_cloud


class LidarCameraFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion_node')

        # Declare (and retrieve) parameters for topic names, frames, etc.
        self.declare_parameter('camera_image_topic', '/camera/image')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('lidar_points_topic', '/scan/points')
        self.declare_parameter('camera_optical_frame', 'camera_link_optical')
        self.declare_parameter('lidar_optical_frame', 'lidar_link_optical')

        self.camera_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_points_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_optical_frame').get_parameter_value().string_value
        self.lidar_frame = self.get_parameter('lidar_optical_frame').get_parameter_value().string_value

        # TF buffer + listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera intrinsics
        self.intrinsic_matrix = None
        self.distortion_model = None
        self.distortion_coeffs = None
        self.image_width = 0
        self.image_height = 0

        # Subscribers
        self.sub_cam_image = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.sub_cam_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, 10)

        # CV Bridge for converting ROS Image -> OpenCV
        self.bridge = CvBridge()

        # Store latest camera frame
        self.last_image = None
        self.last_image_stamp = None

        # Timer to update visualization
        # We'll show images at about 10 Hz in separate windows
        self.timer = self.create_timer(0.1, self.show_windows)

        self.get_logger().info('LidarCameraFusionNode started.')

    def camera_info_callback(self, msg: CameraInfo):
        """
        Extract camera intrinsic parameters once.
        """
        k = msg.k  # 9-element array in row-major
        self.intrinsic_matrix = np.array(k, dtype=np.float32).reshape((3, 3))
        self.distortion_model = msg.distortion_model
        self.distortion_coeffs = np.array(msg.d, dtype=np.float32)
        self.image_width = msg.width
        self.image_height = msg.height

    def image_callback(self, msg: Image):
        """ Convert the incoming sensor_msgs/Image to an OpenCV image. """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_image = cv_img
            self.last_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")

    def lidar_callback(self, msg: PointCloud2):
        """
        - Strip extra fields from LiDAR cloud to ensure only x, y, z remain.
        - Lookup transform from LiDAR frame to camera frame.
        - Transform the stripped cloud.
        - Project the transformed points onto the camera image if available.
        - Also draw a simple 2D overhead map of LiDAR data in a separate window.
        """
        if self.intrinsic_matrix is None:
            self.get_logger().warn('No CameraInfo received yet; cannot project points.')
            return

        # 1) Strip extra fields (like intensity/ring) -> only x,y,z
        xyz_cloud = strip_to_xyz_cloud(msg)

        # 2) Lookup transform from lidar_link_optical -> camera_link_optical
        try:
            transform_stamped: TransformStamped = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.lidar_frame,
                rclpy.time.Time())
        except (tf2.LookupException, tf2.ExtrapolationException) as ex:
            self.get_logger().warn(f"Transform from {self.lidar_frame} to {self.camera_frame} not found: {ex}")
            return

        # 3) Transform the stripped cloud
        try:
            cloud_in_camera_frame = do_transform_cloud(xyz_cloud, transform_stamped)
        except AssertionError as ex:
            self.get_logger().error(f"Cloud transform error: {ex}")
            return

        # 4) Read the transformed cloud's points
        points_3d = list(pc2.read_points(cloud_in_camera_frame, field_names=["x","y","z"], skip_nans=True))

        # 5) Draw simple overhead map
        self.draw_lidar(points_3d)

        # 6) If we have a camera image, project LiDAR points onto it
        if self.last_image is not None and len(points_3d) > 0:
            fused_image = self.last_image.copy()
            self.project_and_draw_points(fused_image, points_3d)
            cv2.imshow("Fused View", fused_image)

    def draw_lidar(self, points_3d):
        """
        Draw a simple 2D overhead representation of the LiDAR data in a "LiDAR View" window.
        """
        pts = np.array(points_3d, dtype=np.float32)
        lidar_img = np.zeros((600, 600, 3), dtype=np.uint8)

        if pts.size == 0:
            cv2.imshow("LiDAR View", lidar_img)
            return

        # very basic scale & offset for 2D visualization (X,Y)
        scale = 20.0
        offset_x, offset_y = 300, 300

        for (x, y, z) in pts:
            # Convert x,y (forward, left) into pixel coords
            px = int(x * scale + offset_x)
            py = int(-y * scale + offset_y)  # invert y
            if 0 <= px < 600 and 0 <= py < 600:
                cv2.circle(lidar_img, (px, py), 1, (0, 255, 0), -1)

        cv2.imshow("LiDAR View", lidar_img)

    def project_and_draw_points(self, image, points_3d):
        """
        Project each (x, y, z) into image pixel (u,v) using camera intrinsics:
         u = fx*(x/z) + cx
         v = fy*(y/z) + cy
        Then draw a circle at (u, v).

        Only draws those points that land within the image bounds.
        """
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]

        for (x, y, z) in points_3d:
            if z <= 0:
                continue  # behind camera

            u = int((fx * x / z) + cx)
            v = int((fy * y / z) + cy)

            if 0 <= u < self.image_width and 0 <= v < self.image_height:
                cv2.circle(image, (u, v), 2, (0, 0, 255), -1)

    def show_windows(self):
        """
        Called periodically (10 Hz). Show the camera image if we have one.
        """
        if self.last_image is not None:
            cv2.imshow("Camera View", self.last_image)
        # OpenCV requires a short waitKey to update the UI
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()