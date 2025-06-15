import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from cv_bridge import CvBridge
import time

class YellowObjectFollower(Node):

    def __init__(self):
        super().__init__('yellow_object_follower')
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
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.latest_depth_image = None

        # Yellow color thresholds in HSV
        self.declare_parameter("h_min", 20)
        self.declare_parameter("s_min", 100)
        self.declare_parameter("v_min", 100)
        self.declare_parameter("h_max", 40)
        self.declare_parameter("s_max", 255)
        self.declare_parameter("v_max", 255)

        self.h_min = self.get_parameter('h_min').get_parameter_value().integer_value
        self.s_min = self.get_parameter('s_min').get_parameter_value().integer_value
        self.v_min = self.get_parameter('v_min').get_parameter_value().integer_value
        self.h_max = self.get_parameter('h_max').get_parameter_value().integer_value
        self.s_max = self.get_parameter('s_max').get_parameter_value().integer_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().integer_value
        self.lower_yellow = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        self.upper_yellow = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)

        # Control parameters
        self.declare_parameter("rcv_timeout_secs", 1.0)
        self.declare_parameter("angular_chase_multiplier", 0.005)
        self.declare_parameter("forward_chase_speed", 0.1)
        self.declare_parameter("search_angular_speed", 0.3)
        self.declare_parameter("min_distance_threshold", 0.5)
        self.declare_parameter("filter_value", 0.9)
        self.declare_parameter("image_width", 320)

        self.rcv_timeout_secs = self.get_parameter('rcv_timeout_secs').get_parameter_value().double_value
        self.angular_chase_multiplier = self.get_parameter('angular_chase_multiplier').get_parameter_value().double_value
        self.forward_chase_speed = self.get_parameter('forward_chase_speed').get_parameter_value().double_value
        self.search_angular_speed = self.get_parameter('search_angular_speed').get_parameter_value().double_value
        self.min_distance_threshold = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.filter_value = self.get_parameter('filter_value').get_parameter_value().double_value
        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.control_loop)
        self.target_center_offset = 0.0
        self.target_distance = float('inf')
        self.last_detection_time = self.get_clock().now()

    def rgb_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    if self.latest_depth_image is not None:
                        depth_image = self.latest_depth_image
                        if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                            depth_value = depth_image[center_y, center_x]
                            self.update_target(center_x, depth_value)
                        else:
                            self.get_logger().warn('Yellow object centroid outside depth image bounds.')
                    else:
                        self.get_logger().warn('No depth image received yet.')
            else:
                self.get_logger().info('No yellow object detected.')
                self.target_distance = float('inf') # Reset distance when not detected

        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def update_target(self, center_x, distance):
        f = self.filter_value
        image_center = self.image_width / 2
        current_offset = center_x - image_center
        self.target_center_offset = self.target_center_offset * f + current_offset * (1 - f)
        self.target_distance = self.target_distance * f + distance * (1 - f)
        self.last_detection_time = self.get_clock().now()

    def control_loop(self):
        msg = Twist()
        now = self.get_clock().now()
        time_diff_ns = (now - self.last_detection_time).nanoseconds
        time_diff = time_diff_ns / 1e9  # Convert nanoseconds to seconds

        if time_diff < self.rcv_timeout_secs:
            self.get_logger().info(f'Target Offset: {self.target_center_offset}, Distance: {self.target_distance}')

            # Angular control
            msg.angular.z = -self.angular_chase_multiplier * self.target_center_offset

            # Linear control
            if self.target_distance > self.min_distance_threshold:
                msg.linear.x = self.forward_chase_speed
            else:
                msg.linear.x = 0.0
                self.get_logger().info('Reached the yellow object')
        else:
            self.get_logger().info('Yellow object lost')
            msg.linear.x = 0.0
            msg.angular.z = self.search_angular_speed  # Search by rotating

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    yellow_object_follower = YellowObjectFollower()
    rclpy.spin(yellow_object_follower)
    yellow_object_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()