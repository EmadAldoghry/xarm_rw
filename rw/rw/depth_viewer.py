import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class ImageSaver(Node):

    def __init__(self):
        super().__init__('image_saver')
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.rgb_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth_image',
            self.depth_callback,
            10)
        self.bridge = CvBridge()
        self.data_folder = 'data'
        os.makedirs(self.data_folder, exist_ok=True)
        self.get_logger().info(f'Saving images to {os.path.abspath(self.data_folder)}')

    def rgb_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Get timestamp with milliseconds
            filename = os.path.join(self.data_folder, f'rgb_{timestamp}.png')
            cv2.imwrite(filename, rgb_image)
            self.get_logger().info(f'Saved RGB image: {filename}')
        except Exception as e:
            self.get_logger().error(f"Error saving RGB image: {e}")

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Normalize the depth image for saving (optional, but makes it viewable as an image)
            depth_normalized = cv2.normalize(depth_image, 255, 255, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Get timestamp with milliseconds
            filename = os.path.join(self.data_folder, f'depth_{timestamp}.png')
            cv2.imwrite(filename, depth_colormap)
            self.get_logger().info(f'Saved depth image: {filename}')
        except Exception as e:
            self.get_logger().error(f"Error saving depth image: {e}")

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()