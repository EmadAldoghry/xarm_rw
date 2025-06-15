import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class YellowObjectFollower(Node):
    def __init__(self):
        super().__init__('yellow_object_follower')

        # QoS profile for best effort communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.yellow_object_subscription = self.create_subscription(
            Point,
            '/yellow_object',
            self.yellow_object_callback,
            qos_profile)  # Use the defined QoS profile

        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.target_center_offset = 0.0
        self.target_distance = 0.0  # Initialize to 0.0
        self.last_detection_time = self.get_clock().now()
        self.image_width = 640

        # Declare parameters
        self.declare_parameter('linear_chase_speed', 0.1)
        self.declare_parameter('angular_chase_multiplier', 0.001)
        self.declare_parameter('close_distance_threshold', 0.5)
        self.declare_parameter('lost_timeout', 1.0)
        self.declare_parameter('filter_value', 0.5)

        self.timer = self.create_timer(0.1, self.control_loop)

    def yellow_object_callback(self, msg):
        current_offset = msg.x - self.image_width / 2
        current_distance = msg.z
        self.update_target(current_offset, current_distance)
        self.last_detection_time = self.get_clock().now()

    def update_target(self, offset, distance):
        f = self.get_parameter('filter_value').get_parameter_value().double_value
        self.target_center_offset = self.target_center_offset * f + offset * (1 - f)
        self.target_distance = self.target_distance * f + distance * (1 - f)

    def control_loop(self):
        linear_chase_speed = self.get_parameter('linear_chase_speed').get_parameter_value().double_value
        angular_chase_multiplier = self.get_parameter('angular_chase_multiplier').get_parameter_value().double_value
        close_distance_threshold = self.get_parameter('close_distance_threshold').get_parameter_value().double_value
        lost_timeout = self.get_parameter('lost_timeout').get_parameter_value().double_value

        if (self.get_clock().now() - self.last_detection_time) > rclpy.time.Duration(seconds=lost_timeout):
            self.get_logger().info("Yellow object lost")
            self.publish_velocity(0.0, 0.0)
            return

        linear_speed = 0.0
        angular_speed = 0.0

        if self.target_distance > 0 and self.target_distance < close_distance_threshold:
            linear_speed = 0.0
            angular_speed = 0.0
        elif self.target_distance >= close_distance_threshold:
            linear_speed = linear_chase_speed

        # Adjust angular speed based on target offset - Removed the negative sign
        angular_speed = self.target_center_offset * angular_chase_multiplier
        self.get_logger().info(f"Target Offset: {self.target_center_offset}, Distance: {self.target_distance}")

        self.publish_velocity(linear_speed, angular_speed)

    def publish_velocity(self, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        self.velocity_publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    yellow_object_follower = YellowObjectFollower()
    rclpy.spin(yellow_object_follower)
    yellow_object_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()