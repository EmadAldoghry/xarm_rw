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
                small_image_height = 120  # Assuming the height of the small images
                padding = 20  # Add some padding
                display_height = height + small_image_height + padding

                # Create and resize the window dynamically
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("frame", width, display_height)

                # Show the latest frame
                cv2.imshow("frame", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()
        self.running = False

    def process_image(self, rgb_img, depth_img):
        """Image processing task to detect and track the nearest black object, switching targets."""
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
        upper_black = np.array([0, 0, 0], dtype=np.uint8)  # Adjust upper bound if needed
        blackMask = cv2.inRange(rgb_img, lower_black, upper_black)

        stackedMask = np.dstack((blackMask, blackMask, blackMask))
        contourMask = stackedMask.copy()
        crosshairMask = rgb_img.copy()  # Use RGB image for crosshair overlay

        # Find contours
        contours, _ = cv2.findContours(blackMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        nearest_point = None
        min_distance_to_center = float('inf')
        min_object_distance = float('inf')

        # Check if the current target point is still on any of the contours
        is_current_target_present = False
        if hasattr(self, 'current_target_point') and self.current_target_point is not None:
            # Ensure current_target_point is a tuple of floats
            target_point_tuple = tuple(float(x) for x in self.current_target_point)
            for contour in contours:
                if cv2.pointPolygonTest(contour, target_point_tuple, False) >= 0:
                    is_current_target_present = True
                    nearest_point = self.current_target_point
                    if 0 <= int(nearest_point[1]) < depth_img.shape[0] and 0 <= int(nearest_point[0]) < depth_img.shape[1]:
                        min_object_distance = depth_img[int(nearest_point[1]), int(nearest_point[0])]
                    break

        # If the current target is not present, find the new nearest point
        if not is_current_target_present:
            self.current_target_point = None  # Reset the current target
            if len(contours) > 0:
                for contour in contours:
                    for point in contour:
                        px, py = point[0]
                        distance_to_center = np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)

                        if 0 <= py < depth_img.shape[0] and 0 <= px < depth_img.shape[1]:
                            object_distance = depth_img[py, px]
                            if not np.isnan(object_distance):
                                if distance_to_center < min_distance_to_center:
                                    min_distance_to_center = distance_to_center
                                    nearest_point = (float(px), float(py)) # Store as tuple of floats
                                    min_object_distance = object_distance
                if nearest_point is not None:
                    self.current_target_point = nearest_point  # Set the new target

        # Process the nearest point (either the current target or the newly selected one)
        if nearest_point is not None:
            nearest_x, nearest_y = int(nearest_point[0]), int(nearest_point[1])
            cv2.circle(crosshairMask, (nearest_x, nearest_y), 5, (0, 255, 0), -1)  # Mark the nearest point

            # Draw crosshair and display distance
            cv2.line(crosshairMask, (nearest_x, 0), (nearest_x, rows), (0, 0, 255), 2)
            cv2.line(crosshairMask, (0, nearest_y), (cols, nearest_y), (0, 0, 255), 2)
            cv2.line(crosshairMask, (center_x, 0), (center_x, rows), (255, 0, 0), 2)
            cv2.putText(crosshairMask, f"{min_object_distance:.2f} m", (nearest_x + 10, nearest_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Control logic to move towards the nearest point
            if abs(center_x - nearest_x) > 20:
                msg.linear.x = 0.0
                if center_x > nearest_x:
                    msg.angular.z = 0.2
                else:
                    msg.angular.z = -0.2
            else:
                msg.linear.x = 0.2
                msg.angular.z = 0.0
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        # Publish cmd_vel
        self.publisher.publish(msg)

        return blackMask, contourMask, crosshairMask

    # Add small images to the top row of the main image
    def add_small_pictures(self, large_image, small_images):
        """Adds small images horizontally on top of a large image."""
        large_height, large_width, _ = large_image.shape
        num_small = len(small_images)
        if num_small == 0:
            return large_image

        small_height = 120
        small_width = large_width // num_small
        combined_image = large_image.copy()

        for i, img in enumerate(small_images):
            resized_img = cv2.resize(img, (small_width, small_height))
            if len(resized_img.shape) == 2:  # Convert grayscale to BGR
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

            start_x = i * small_width
            end_x = start_x + small_width

            # Ensure the slice indices are within bounds
            if 0 <= start_x < large_width and 0 < end_x <= large_width:
                combined_image[0:small_height, start_x:end_x] = resized_img[:small_height, :small_width]

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