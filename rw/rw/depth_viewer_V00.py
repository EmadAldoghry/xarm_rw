import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthCameraSubscriber(Node):

    def __init__(self):
        super().__init__('depth_camera_subscriber')
        self.depth_image_subscription = self.create_subscription(
            Image,
            '/camera/depth_image',
            self.depth_image_callback,
            10)
        self.point_cloud_subscription = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.point_cloud_callback,
            10)
        self.bridge = CvBridge()

    def depth_image_callback(self, msg):
        try:
            # Convert the depth image message to a CV2 image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Normalize the depth image to a range suitable for display
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # Display the depth image
            cv2.imshow('Depth Image', depth_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def point_cloud_callback(self, msg):
        # You can access the point cloud data here
        # The data is in a structured format within the msg.data
        # and the structure is defined by msg.fields

        # For basic viewing, you can print some information
        self.get_logger().info(f"Received point cloud with height: {msg.height}, width: {msg.width}, point_step: {msg.point_step}")

        # To actually visualize the point cloud, you would typically use libraries like:
        # - open3d: A popular library for 3D data processing.
        # - pcl (Point Cloud Library): Another widely used library.
        # - matplotlib (for basic 3D scatter plots).

        # Example of accessing the raw data (requires understanding the point format)
        # if msg.height > 0: # Check if the point cloud is not unorganized
        #     point_data = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width, -1)
        #     # Assuming typical XYZ data, you can access points like this:
        #     # print(f"First point: {point_data[0, 0]}")
        # else: # Unorganized point cloud
        #     point_data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, len(msg.fields))
        #     # print(f"First point: {point_data[0]}")

def main(args=None):
    rclpy.init(args=args)
    depth_camera_subscriber = DepthCameraSubscriber()
    rclpy.spin(depth_camera_subscriber)
    depth_camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()