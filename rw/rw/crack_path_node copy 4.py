#!/home/aldoghry/my_ros2_env/bin/python

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as ROSPoint
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import struct

class CrackPathNode(Node):
    def __init__(self):
        super().__init__('crack_path_extractor')

        # Where to save PCD files (if you want to save in the future)
        self.save_directory = "/home/aldoghry/code/pointClouds"
        os.makedirs(self.save_directory, exist_ok=True)

        # Subscriber
        self.create_subscription(
            PointCloud2,
            '/camera2/points',
            self.pointcloud_callback,
            qos_profile_sensor_data
        )

        # Publishers
        self.marker_pub = self.create_publisher(Marker, 'crack_path_marker', 10)
        self.inlier_pub = self.create_publisher(PointCloud2, 'ground_plane_points', 10)  # Red
        self.outlier_pub = self.create_publisher(PointCloud2, 'non_ground_points', 10)   # Green

        self.crack_path_points = []

        # XYZRGB PointFields
        self.xyzrgb_point_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info("CrackPathNode initialized: waiting for point cloud data...")

    def pack_rgb(self, r, g, b):
        """
        Pack three 8-bit channel values into the 'rgb' float field.
        """
        rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
        packed = struct.pack('I', rgb_int)
        unpacked_float = struct.unpack('f', packed)[0]
        return unpacked_float

    def pointcloud_callback(self, cloud_msg: PointCloud2):
        self.get_logger().debug(
            f"Received point cloud message with {cloud_msg.width * cloud_msg.height} points."
        )
        current_stamp = self.get_clock().now().to_msg()
        current_frame_id = cloud_msg.header.frame_id

        # --- Convert raw PointCloud2 to Numpy array ---
        points_iter = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        try:
            points = np.array([[x, y, z] for (x, y, z) in points_iter], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Error reading points from PointCloud2: {e}")
            return

        if points.size == 0:
            self.get_logger().warn("Point cloud empty after NaN removal.")
            return

        # Remove any inf values
        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]
        if points.size == 0:
            self.get_logger().warn("Point cloud empty after inf removal.")
            # Clear visualizations if processing stops here
            self.publish_empty_colored_clouds(current_frame_id, current_stamp)
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        self.get_logger().info(f"Processing {points.shape[0]} finite points.")

        # ----------------------------------------------------------------
        # CROP: Use an AxisAlignedBoundingBox to restrict the ROI
        # ----------------------------------------------------------------
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Adjust these corners to your actual ROI
        corner_min = (0.2,  -0.31, -0.33)  # smaller x, y, z
        corner_max = (0.73,  0.33,  0.19)  # larger x, y, z
        crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=corner_min, max_bound=corner_max)

        pcd_cropped = pcd.crop(crop_bbox)
        cropped_points = np.asarray(pcd_cropped.points)

        if cropped_points.size == 0:
            self.get_logger().warn("No points left after cropping.")
            self.publish_empty_colored_clouds(current_frame_id, current_stamp)
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        # Use cropped_points for the rest of the pipeline
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cropped_points)

        # --------------------------------------------------------
        # Downsample
        # --------------------------------------------------------
        voxel_size = 0.005
        try:
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            self.get_logger().info(f"Downsampled to {len(pcd_downsampled.points)} points.")
        except Exception as e:
            self.get_logger().error(f"Error during downsampling: {e}")
            # Clear visualizations on error
            self.publish_empty_colored_clouds(current_frame_id, current_stamp)
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        if not pcd_downsampled.has_points():
            self.get_logger().warn("Downsampling resulted in an empty point cloud.")
            self.publish_empty_colored_clouds(current_frame_id, current_stamp)
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        # --------------------------------------------------------
        # RANSAC Plane Segmentation
        # --------------------------------------------------------
        distance_threshold = 0.01
        try:
            plane_model, inliers = pcd_downsampled.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=100
            )
        except RuntimeError as e:
            self.get_logger().warn(f"RANSAC plane segmentation failed: {e}.")
            # Clear visualizations on failure
            self.publish_empty_colored_clouds(current_frame_id, current_stamp)
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        # Separate inliers/outliers
        ground_cloud = pcd_downsampled.select_by_index(inliers)
        non_ground_cloud = pcd_downsampled.select_by_index(inliers, invert=True)
        inlier_points = np.asarray(ground_cloud.points)
        outlier_points = np.asarray(non_ground_cloud.points)
        self.get_logger().info(
            f"Plane segmentation: {inlier_points.shape[0]} inliers (ground), "
            f"{outlier_points.shape[0]} outliers."
        )

        # Create header for colored clouds
        colored_header = Header()
        colored_header.stamp = current_stamp
        colored_header.frame_id = current_frame_id

        # Publish RED inliers
        if inlier_points.shape[0] > 0:
            red_packed = self.pack_rgb(255, 0, 0)
            inlier_colors = np.full((inlier_points.shape[0], 1), red_packed, dtype=np.float32)
            inlier_data_combined = np.hstack((inlier_points, inlier_colors))
            inlier_data_list = inlier_data_combined.tolist()
            inlier_cloud_msg = pc2.create_cloud(colored_header, self.xyzrgb_point_fields, inlier_data_list)
            self.inlier_pub.publish(inlier_cloud_msg)
        else:
            # Publish empty to clear
            inlier_cloud_msg = pc2.create_cloud(colored_header, self.xyzrgb_point_fields, [])
            self.inlier_pub.publish(inlier_cloud_msg)

        # Publish GREEN outliers
        if outlier_points.shape[0] > 0:
            green_packed = self.pack_rgb(0, 255, 0)
            outlier_colors = np.full((outlier_points.shape[0], 1), green_packed, dtype=np.float32)
            outlier_data_combined = np.hstack((outlier_points, outlier_colors))
            outlier_data_list = outlier_data_combined.tolist()
            outlier_cloud_msg = pc2.create_cloud(colored_header, self.xyzrgb_point_fields, outlier_data_list)
            self.outlier_pub.publish(outlier_cloud_msg)
        else:
            # Publish empty to clear
            outlier_cloud_msg = pc2.create_cloud(colored_header, self.xyzrgb_point_fields, [])
            self.outlier_pub.publish(outlier_cloud_msg)

        # --------------------------------------------------------
        # Crack Extraction from Outliers
        # --------------------------------------------------------
        if not non_ground_cloud.has_points():
            self.get_logger().info("No non-ground points left after segmentation.")
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        remaining_points = outlier_points  # alias

        # --- DBSCAN Clustering ---
        eps_distance = 0.02
        min_points = 5
        try:
            if remaining_points.shape[0] < min_points:
                self.get_logger().warn(
                    f"Not enough points ({remaining_points.shape[0]}) for DBSCAN (min={min_points})."
                )
                self.publish_empty_marker(current_frame_id, current_stamp)
                self.crack_path_points = []
                return
            labels = np.array(
                non_ground_cloud.cluster_dbscan(eps=eps_distance, min_points=min_points, print_progress=False)
            )
        except Exception as e:
            self.get_logger().error(f"Error during DBSCAN clustering: {e}")
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        valid_labels = labels[labels >= 0]
        if valid_labels.size == 0:
            self.get_logger().info("No clusters found (all points classified as noise by DBSCAN).")
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        # Find the largest cluster
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_cluster_id = unique_labels[np.argmax(counts)]
        crack_indices = np.where(labels == largest_cluster_id)[0]
        crack_cloud = non_ground_cloud.select_by_index(crack_indices)
        crack_points = np.asarray(crack_cloud.points)
        self.get_logger().info(f"Largest cluster (potential crack) has {crack_points.shape[0]} points.")

        if crack_points.shape[0] < 2:
            self.get_logger().warn("Crack cluster has too few points (< 2) to form a path.")
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        # --- Order the path ---
        try:
            ordered_path = self.order_points_simple_nn(crack_cloud)
            if ordered_path is None or len(ordered_path) < 2:
                self.get_logger().warn("Failed to order crack points into a path.")
                self.publish_empty_marker(current_frame_id, current_stamp)
                self.crack_path_points = []
                return
            self.crack_path_points = [(float(x), float(y), float(z)) for x, y, z in ordered_path]
        except Exception as e:
            self.get_logger().error(f"Error during path ordering: {e}")
            self.publish_empty_marker(current_frame_id, current_stamp)
            self.crack_path_points = []
            return

        self.get_logger().info(f"Crack path ({len(self.crack_path_points)} points) extracted.")
        self.publish_path_marker(current_frame_id, current_stamp)

    def order_points_simple_nn(self, point_cloud):
        """
        Simple nearest-neighbor ordering of the crack points.
        Starts with the point having the smallest X, then
        iteratively picks the next nearest neighbor.
        """
        points = np.asarray(point_cloud.points)
        if points.shape[0] < 2:
            return None

        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        # Start with the point that has the smallest X
        start_idx = np.argmin(points[:, 0])
        path_indices = [start_idx]
        used_indices = {start_idx}
        current_idx = start_idx

        while len(used_indices) < points.shape[0]:
            k = min(10, points.shape[0])
            [num_found, idxs, _] = pcd_tree.search_knn_vector_3d(
                point_cloud.points[current_idx],
                k
            )
            next_idx = -1
            # Skip idxs[0] because it is the current point itself
            for i in range(1, num_found):
                neighbor_idx = idxs[i]
                if neighbor_idx not in used_indices:
                    next_idx = neighbor_idx
                    break
            if next_idx != -1:
                path_indices.append(next_idx)
                used_indices.add(next_idx)
                current_idx = next_idx
            else:
                self.get_logger().debug(f"Path ordering stopped. Path length: {len(path_indices)}/{points.shape[0]}")
                break

        return points[path_indices]

    def publish_path_marker(self, frame_id, stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "crack_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.015  # thickness of the line
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.7
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0

        marker.points = [
            ROSPoint(x=p[0], y=p[1], z=p[2]) for p in self.crack_path_points
        ]

        if marker.points:
            self.marker_pub.publish(marker)
            self.get_logger().debug(f"Published marker with {len(marker.points)} points.")
        else:
            self.get_logger().debug("Attempted to publish marker, but no points.")
            self.publish_empty_marker(frame_id, stamp)

    def publish_empty_marker(self, frame_id, stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "crack_path"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)
        self.get_logger().debug("Published DELETE marker.")

    def publish_empty_colored_clouds(self, frame_id, stamp):
        """
        Publishes empty point clouds for inliers and outliers.
        This helps to clear any previously published data on those topics
        when there's nothing to show.
        """
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        empty_cloud_msg = pc2.create_cloud(header, self.xyzrgb_point_fields, [])
        self.inlier_pub.publish(empty_cloud_msg)
        self.outlier_pub.publish(empty_cloud_msg)
        self.get_logger().debug("Published empty colored point clouds.")

def main(args=None):
    rclpy.init(args=args)
    node = CrackPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


