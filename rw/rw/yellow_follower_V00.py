import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class YellowObjectDetector(Node):

    def __init__(self):
        super().__init__('yellow_object_detector')
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
        self.ball_pub  = self.create_publisher(Point,"/yellow_object",10)
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.latest_depth_info = None

        # Yellow color thresholds in HSV
        self.lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        self.upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    def rgb_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)

            # Find contours of the yellow object
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour (assuming it's the main yellow object)
                largest_contour = max(contours, key=cv2.contourArea)

                # Calculate the centroid of the largest contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    if self.latest_depth_image is not None:
                        depth_image = self.latest_depth_image

                        # Ensure the centroid is within the depth image bounds
                        if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                            depth_value = depth_image[center_y, center_x]

                            point_msg = Point()
                            point_msg.x = float(center_x)
                            point_msg.y = float(center_y)
                            point_msg.z = float(depth_value)  # Distance in the depth image unit
                            self.ball_pub.publish(point_msg)
                            self.get_logger().info(f'Detected yellow object at pixel: ({center_x}, {center_y}), Distance: {depth_value}')
                        else:
                            self.get_logger().warn('Yellow object centroid outside depth image bounds.')
                    else:
                        self.get_logger().warn('No depth image received yet.')
            else:
                # If no yellow object is detected, you might want to publish a default value or nothing
                # point_msg = Point()
                # point_msg.x = -1.0
                # point_msg.y = -1.0
                # point_msg.z = -1.0
                # self.ball_pub.publish(point_msg)
                self.get_logger().info('No yellow object detected.')

        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_callback(self, msg):
        try:
            # Use 'passthrough' to keep the original depth values
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

def main(args=None):
    rclpy.init(args=args)
    yellow_object_detector = YellowObjectDetector()
    rclpy.spin(yellow_object_detector)
    yellow_object_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()