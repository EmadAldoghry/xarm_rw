#!/home/aldoghry/my_ros2_env/bin/python

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as ROSPoint
import sensor_msgs_py.point_cloud2 as pc2  # ROS2 utility for PointCloud2
import numpy as np
import open3d as o3d

class CrackPathNode(Node):
    def __init__(self):
        super().__init__('crack_path_extractor')
        # Subscriber to depth camera point cloud (PointCloud2)
        self.create_subscription(
            PointCloud2,
            '/camera2/points',    # topic name (adjust if different)
            self.pointcloud_callback,
            qos_profile_sensor_data   # use sensor data QoS for compatible reliability
        )
        # Publisher for visualization Marker
        self.marker_pub = self.create_publisher(Marker, 'crack_path_marker', 10)
        # Stored crack path (list of 3D points in camera frame)
        self.crack_path_points = []  # will hold the latest extracted path as a list of (x,y,z) tuples

        self.get_logger().info("CrackPathNode initialized: waiting for point cloud data...")

    def pointcloud_callback(self, cloud_msg: PointCloud2):
        """Callback function to process incoming PointCloud2 messages."""
        self.get_logger().debug(f"Received point cloud message with {cloud_msg.width * cloud_msg.height} points.")

        # Convert PointCloud2 message to a simple Nx3 NumPy array.
        # skip_nans=True handles NaN values during reading
        points_iter = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        try:
            points = np.array([[x, y, z] for (x, y, z) in points_iter], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Error reading points from PointCloud2: {e}")
            return

        if points.size == 0:
            self.get_logger().warn("Received point cloud resulted in zero points after NaN removal.")
            return

        # --- START: Remove points with infinite values ---
        num_points_before_inf_removal = points.shape[0]
        # Create a boolean mask: True for rows where *all* values are finite
        finite_mask = np.all(np.isfinite(points), axis=1)
        # Apply the mask to keep only finite points
        points = points[finite_mask]
        num_points_after_inf_removal = points.shape[0]
        num_inf_removed = num_points_before_inf_removal - num_points_after_inf_removal

        if num_inf_removed > 0:
            self.get_logger().info(f"Removed {num_inf_removed} points with non-finite (inf or -inf) values.")
        # --- END: Remove points with infinite values ---

        # Check again if the cloud is empty after removing inf values
        if points.size == 0:
            self.get_logger().warn("Point cloud is empty after removing non-finite values.")
            return

        self.get_logger().info(f"Processing {points.shape[0]} finite points.")

        # Downsample the point cloud using a voxel grid filter.
        voxel_size = 0.005  # 5 mm voxels (Adjusted from original comment for consistency)
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            self.get_logger().info(f"Downsampled to {len(pcd_downsampled.points)} points.")
        except Exception as e:
            self.get_logger().error(f"Error during Open3D processing (downsampling): {e}")
            return

        # Check if downsampling resulted in an empty cloud
        if not pcd_downsampled.has_points():
             self.get_logger().warn("Downsampling resulted in an empty point cloud.")
             return

        # Segment the largest plane (ground) using RANSAC.
        distance_threshold = 0.01  # 1 cm threshold
        try:
            plane_model, inliers = pcd_downsampled.segment_plane(distance_threshold=distance_threshold,
                                                       ransac_n=3, num_iterations=100)
        except RuntimeError as e:
             # Catch potential Open3D runtime errors, e.g., not enough points for RANSAC
             self.get_logger().warn(f"RANSAC plane segmentation failed: {e}. Need at least 3 points.")
             return

        # Remove ground plane points (inliers).
        # ground_cloud = pcd_downsampled.select_by_index(inliers) # Not used, commented out
        non_ground_cloud = pcd_downsampled.select_by_index(inliers, invert=True)

        if not non_ground_cloud.has_points():
            self.get_logger().warn("All points were on the ground plane; no potential crack points left.")
            # Clear previous path visualization if no new path is found
            self.publish_empty_marker(cloud_msg.header.frame_id)
            self.crack_path_points = []
            return

        remaining_points = np.asarray(non_ground_cloud.points)
        self.get_logger().info(f"{remaining_points.shape[0]} points remaining after ground removal.")

        # Cluster the remaining points to find crack candidate(s)
        eps_distance = 0.02   # 2 cm cluster search radius (Adjusted from original comment for consistency)
        min_points = 5       # minimum points to form a cluster
        try:
            # Note: cluster_dbscan requires min_points >= 3
            if remaining_points.shape[0] < min_points:
                 self.get_logger().warn(f"Not enough points ({remaining_points.shape[0]}) for DBSCAN clustering (min_points={min_points}).")
                 self.publish_empty_marker(cloud_msg.header.frame_id)
                 self.crack_path_points = []
                 return
            labels = np.array(non_ground_cloud.cluster_dbscan(eps=eps_distance, min_points=min_points, print_progress=False)) # Set print_progress to False/True as needed
        except Exception as e:
            self.get_logger().error(f"Error during DBSCAN clustering: {e}")
            return

        # Filter out noise points (label -1)
        valid_labels = labels[labels >= 0]
        if valid_labels.size == 0:
            self.get_logger().warn("No clusters found (all points classified as noise by DBSCAN).")
            self.publish_empty_marker(cloud_msg.header.frame_id)
            self.crack_path_points = []
            return

        # Identify the largest cluster (most points) as the crack
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_cluster_id = unique_labels[np.argmax(counts)]
        crack_indices = np.where(labels == largest_cluster_id)[0]
        crack_cloud = non_ground_cloud.select_by_index(crack_indices)
        crack_points = np.asarray(crack_cloud.points)

        self.get_logger().info(f"Largest cluster (potential crack) has {crack_points.shape[0]} points.")

        # Order the crack cluster points to form a continuous path.
        if crack_points.shape[0] < 2:
            self.get_logger().warn("Crack cluster has too few points (< 2) to form a path.")
            self.publish_empty_marker(cloud_msg.header.frame_id)
            self.crack_path_points = []
            return

        # --- Path Ordering Logic (using simple nearest neighbor) ---
        try:
            ordered_path = self.order_points_simple_nn(crack_cloud)
            if ordered_path is None or len(ordered_path) < 2:
                 self.get_logger().warn("Failed to order crack points into a path.")
                 self.publish_empty_marker(cloud_msg.header.frame_id)
                 self.crack_path_points = []
                 return
            self.crack_path_points = [(float(x), float(y), float(z)) for x, y, z in ordered_path]
        except Exception as e:
             self.get_logger().error(f"Error during path ordering: {e}")
             self.publish_empty_marker(cloud_msg.header.frame_id)
             self.crack_path_points = []
             return
        # --- End Path Ordering Logic ---

        self.get_logger().info(f"Crack path ({len(self.crack_path_points)} points) extracted.")

        # Publish the path as a visualization Marker (LINE_STRIP)
        self.publish_path_marker(cloud_msg.header.frame_id)

    def order_points_simple_nn(self, point_cloud):
        """Orders points in a point cloud using a simple nearest-neighbor approach."""
        points = np.asarray(point_cloud.points)
        if points.shape[0] < 2:
            return None # Cannot form a path

        # Build KDTree for efficient neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)

        # Start from an arbitrary point (e.g., point with min x)
        start_idx = np.argmin(points[:, 0])

        path_indices = [start_idx]
        used_indices = {start_idx}
        current_idx = start_idx

        while len(used_indices) < points.shape[0]:
            # Find k nearest neighbors (k=5 is arbitrary, adjust if needed)
            # Increase k slightly to reduce chances of getting stuck if nearest is already used
            k = min(10, points.shape[0]) # Search up to 10 neighbors or total points
            [num_found, idxs, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[current_idx], k)

            next_idx = -1
            # Find the closest neighbor *not* already in the path
            for i in range(1, num_found): # Start from 1 to exclude self
                 neighbor_idx = idxs[i]
                 if neighbor_idx not in used_indices:
                     next_idx = neighbor_idx
                     break # Found the closest unused neighbor

            if next_idx != -1:
                path_indices.append(next_idx)
                used_indices.add(next_idx)
                current_idx = next_idx
            else:
                # No unused neighbors found among the k nearest, path might be broken
                self.get_logger().debug(f"Path ordering stopped: No unused neighbors found for point {current_idx}. Path length: {len(path_indices)}/{points.shape[0]}")
                break

        return points[path_indices]

    def publish_path_marker(self, frame_id):
        """Publishes the current crack path as a LINE_STRIP marker."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "crack_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        # Adjust scale and color as needed
        marker.scale.x = 0.015  # Slightly thicker line
        marker.color.a = 1.0    # Alpha (opacity)
        marker.color.r = 1.0    # Red
        marker.color.g = 0.7    # Orange-ish tint
        marker.color.b = 0.0    # Blue
        marker.pose.orientation.w = 1.0 # Avoid potential issues with uninitialized orientation
        # Assign points
        marker.points = [ROSPoint(x=p[0], y=p[1], z=p[2]) for p in self.crack_path_points]
        # Ensure marker is published even if points list is short but > 0
        if marker.points:
             self.marker_pub.publish(marker)
             self.get_logger().debug(f"Published marker with {len(marker.points)} points.")
        else:
             self.get_logger().debug("Attempted to publish marker, but no points were available.")
             self.publish_empty_marker(frame_id) # Ensure old marker is cleared

    def publish_empty_marker(self, frame_id):
         """Publishes an empty marker to clear previous visualizations."""
         marker = Marker()
         marker.header.frame_id = frame_id
         marker.header.stamp = self.get_clock().now().to_msg()
         marker.ns = "crack_path"
         marker.id = 0
         marker.action = Marker.DELETE # Use DELETE action to clear
         self.marker_pub.publish(marker)
         self.get_logger().debug("Published DELETE marker to clear visualization.")


def main(args=None):
    rclpy.init(args=args)
    node = CrackPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up node and shutdown ROS
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()