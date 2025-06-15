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
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 800, 600)

        while rclpy.ok():
            with self.frame_lock:
                rgb_frame = self.latest_rgb_frame
                depth_frame = self.latest_depth_frame

            if rgb_frame is not None and depth_frame is not None:
                # Process the current image
                mask, contour, crosshair = self.process_image(rgb_frame, depth_frame)

                # Add processed images as small images on top of main image
                result = self.add_small_pictures(rgb_frame, [mask, contour, crosshair])

                # Show the latest frame
                cv2.imshow("frame", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()
        self.running = False

    def process_image(self, rgb_img, depth_img):
        """Image processing task to detect and track the nearest black object."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        rows, cols = rgb_img.shape[:2]

        # Threshold to find black objects (RGB)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([0, 0, 0], dtype=np.uint8) # Adjust upper bound if needed
        blackMask = cv2.inRange(rgb_img, lower_black, upper_black)

        stackedMask = np.dstack((blackMask, blackMask, blackMask))
        contourMask = stackedMask.copy()
        crosshairMask = rgb_img.copy() # Use RGB image for crosshair overlay

        # Find contours
        contours, _ = cv2.findContours(blackMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        nearest_contour = None
        min_distance = float('inf')

        # Find the nearest black object
        if len(contours) > 0:
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Get distance from depth image
                    distance = depth_img[cy, cx]

                    if not np.isnan(distance) and distance < min_distance:
                        min_distance = distance
                        nearest_contour = contour

            # Process the nearest contour
            if nearest_contour is not None:
                cv2.drawContours(contourMask, [nearest_contour], -1, (0, 255, 0), 2)
                M = cv2.moments(nearest_contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw crosshair and display distance
                cv2.line(crosshairMask, (cx, 0), (cx, rows), (0, 0, 255), 2)
                cv2.line(crosshairMask, (0, cy), (cols, cy), (0, 0, 255), 2)
                cv2.line(crosshairMask, (int(cols / 2), 0), (int(cols / 2), rows), (255, 0, 0), 2)
                cv2.putText(crosshairMask, f"{min_distance:.2f} m", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Control logic to chase the black object
                if abs(cols / 2 - cx) > 20:
                    msg.linear.x = 0.0
                    if cols / 2 > cx:
                        msg.angular.z = 0.2
                    else:
                        msg.angular.z = -0.2
                else:
                    msg.linear.x = 0.2
                    msg.angular.z = 0.0
            else:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        # Publish cmd_vel
        self.publisher.publish(msg)

        return blackMask, contourMask, crosshairMask

    # Removed the grayscale threshold function
    # def threshold_binary_black(self, img, thresh=(0, 100)):
    #     binary = np.zeros_like(img)
    #     binary[(img >= thresh[0]) & (img <= thresh[1])] = 255
    #     return binary

    # Add small images to the top row of the main image
    def add_small_pictures(self, img, small_images, size=(160, 120)):
        result = img.copy()
        main_height, main_width = img.shape[:2]
        x_base_offset = 40
        y_base_offset = 10
        x_offset = x_base_offset
        y_offset = y_base_offset

        for small in small_images:
            small_resized = cv2.resize(small, size)
            if len(small_resized.shape) == 2:
                small_resized = np.dstack((small_resized, small_resized, small_resized))

            h, w = small_resized.shape[:2]
            if y_offset + h <= main_height and x_offset + w <= main_width:
                roi = result[y_offset:y_offset + h, x_offset:x_offset + w]
                if roi.shape == small_resized.shape:
                    result[y_offset:y_offset + h, x_offset:x_offset + w] = small_resized
            x_offset += size[0] + x_base_offset
        return result

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