import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import threading

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Subscribers for RGB and Depth images
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

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Variables to store the latest frames
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.frame_lock = threading.Lock()  # Lock to ensure thread safety

        # Flag to control the display loop
        self.running = True

        # Start a separate thread for spinning
        self.spin_thread = threading.Thread(target=self.spin_thread_func)
        self.spin_thread.start()

    def spin_thread_func(self):
        """Separate thread function for rclpy spinning."""
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def rgb_callback(self, msg):
        """Callback function to receive and store the latest RGB frame."""
        try:
            with self.frame_lock:
                self.latest_rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"Error converting RGB image: {e}")

    def depth_callback(self, msg):
        """Callback function to receive and store the latest depth frame."""
        try:
            with self.frame_lock:
                # Use 32FC1 for single-channel float depth images
                self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().warn(f"Error converting depth image: {e}")

    def display_image(self):
        """Main loop to process and display the latest frame."""
        while rclpy.ok():
            with self.frame_lock:
                rgb_frame = self.latest_rgb_frame
                depth_frame = self.latest_depth_frame

            if rgb_frame is not None and depth_frame is not None:
                # Get the dimensions of the RGB frame
                height, width = rgb_frame.shape[:2]

                # Process the current image
                mask, contour, crosshair = self.process_image(rgb_frame, depth_frame)

                # Add processed images as small images on top of main image
                result = self.add_small_pictures(rgb_frame, [mask, contour, crosshair])

                # Calculate the desired height of the display window
                small_image_height = 240  # Increased height for small images
                padding = 40  # Increased padding
                display_height = height + small_image_height + padding

                # Create and resize the window dynamically
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

                # Show the latest frame
                cv2.imshow("frame", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()
        self.running = False

    def process_image(self, rgb_img, depth_img):
        """Image processing task to detect and track a longitudinal crack."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        rows, cols = rgb_img.shape[:2]
        center_x = cols // 2
        center_y = rows // 2

        # Threshold to find black objects (RGB)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([0, 0, 0], dtype=np.uint8)
        blackMask = cv2.inRange(rgb_img, lower_black, upper_black)

        stackedMask = np.dstack((blackMask, blackMask, blackMask))
        contourMask = stackedMask.copy()
        crosshairMask = rgb_img.copy()

        # Find contours
        contours, _ = cv2.findContours(blackMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- New Logic for Crack Tracking ---
        if contours:
            # Assuming the largest contour is the crack (you might need more robust filtering)
            crack_contour = max(contours, key=cv2.contourArea)

            # Fit a line to the contour to determine its orientation
            [vx, vy, x, y] = cv2.fitLine(crack_contour, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
            slope = vy / vx if vx != 0 else float('inf')

            # Find the extreme points of the contour (endpoints of the crack)
            leftmost = tuple(crack_contour[crack_contour[:,:,0].argmin()][0])
            rightmost = tuple(crack_contour[crack_contour[:,:,0].argmax()][0])
            topmost = tuple(crack_contour[crack_contour[:,:,1].argmin()][0])
            bottommost = tuple(crack_contour[crack_contour[:,:,1].argmax()][0])

            # Determine the approximate orientation (horizontal or vertical)
            is_horizontal = abs(slope) < 1  # Adjust threshold as needed

            if is_horizontal:
                start_point = leftmost
                end_point = rightmost
            else:
                start_point = topmost
                end_point = bottommost

            # Draw the line and endpoints
            cv2.line(crosshairMask, start_point, end_point, (0, 255, 255), 2)
            cv2.circle(crosshairMask, start_point, 5, (0, 255, 0), -1) # Mark start
            cv2.circle(crosshairMask, end_point, 5, (0, 0, 255), -1)   # Mark end

            # --- Robot Control Logic to Follow the Crack ---
            # Target a point slightly ahead on the crack from the robot's perspective
            target_offset = 50  # Distance ahead to target (adjust as needed)

            if is_horizontal:
                target_x = start_point[0] + target_offset
                target_y = int(start_point[1] + slope * (target_x - start_point[0]))
            else:
                target_y = start_point[1] + target_offset
                target_x = int(start_point[0] + (target_y - start_point[1]) / slope)

            # Ensure target is within image bounds
            target_x = max(0, min(cols - 1, target_x))
            target_y = max(0, min(rows - 1, target_y))

            target_point = (target_x, target_y)
            cv2.circle(crosshairMask, target_point, 5, (255, 0, 255), -1) # Mark target

            # Calculate angle to target
            angle_to_target = np.arctan2(target_point[0] - center_x, center_y - target_point[1])

            # Control logic (adjust these values)
            linear_speed = 0.1
            angular_speed = 0.2

            msg.linear.x = linear_speed
            msg.angular.z = angular_speed * angle_to_target

        else:
            # If no crack is detected, stop
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        # Publish cmd_vel
        self.publisher.publish(msg)

        return blackMask, contourMask, crosshairMask

    def add_small_pictures(self, large_image, small_images):
        """Adds small images horizontally on top of a large image with increased size."""
        large_height, large_width, _ = large_image.shape
        num_small = len(small_images)
        if num_small == 0:
            return large_image

        small_height = 240  # Increased height for small images
        combined_image = large_image.copy()

        # Calculate available width for small images and divide evenly
        available_width = large_width
        small_width_target = available_width // num_small

        for i, img in enumerate(small_images):
            # Resize the small image
            resized_img = cv2.resize(img, (small_width_target, small_height))

            if len(resized_img.shape) == 2:  # Convert grayscale to BGR
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

            current_small_height, current_small_width = resized_img.shape[:2]
            start_x = i * small_width_target
            end_x = start_x + current_small_width

            # Ensure the slice indices are within bounds
            if 0 <= start_x < large_width and 0 < end_x <= large_width:
                combined_image[0:current_small_height, start_x:end_x] = resized_img

        return combined_image

    def stop(self):
        """Stop the node and the spin thread."""
        self.running = False
        self.spin_thread.join()

def main(args=None):
    print("OpenCV version: %s" % cv2.__version__)
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        node.display_image()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()