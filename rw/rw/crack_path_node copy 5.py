#!/home/aldoghry/my_ros2_env/bin/python

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import struct
import time # Import time for unique filenames if needed (optional)

# TF2 imports
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf_transformations # For converting quaternion to matrix etc.
from geometry_msgs.msg import TransformStamped

class GroundPlaneSegmenterNode(Node):
    def __init__(self):
        super().__init__('ground_plane_segmenter_tf_merge_save_node')

        # Target frame for all processing and output
        self.target_frame = "base_link"

        # --- Hollow Cylinder Filter Parameters (defined in target_frame) ---
        self.apply_hollow_cylinder_filter = True # Set to False to disable
        self.cylinder_center = np.array([-0.038302994, 0.0, 0.090695661])
        self.outer_cylinder_radius = 0.50
        self.inner_cylinder_radius = 0.40
        self.cylinder_height = 0.416
        # Pre-calculate squared radii and half-height for efficiency
        self.outer_radius_sq = self.outer_cylinder_radius**2
        self.inner_radius_sq = self.inner_cylinder_radius**2
        self.half_height = self.cylinder_height / 2.0
        # --- End Filter Parameters ---

        # TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Directory to save the merged cloud
        self.save_directory = "/home/aldoghry/code/pointClouds"
        os.makedirs(self.save_directory, exist_ok=True)
        self.first_cloud_saved = False
        self.save_filename = "merged_first_filtered_cloud.ply" # Updated filename

        # Subscriber
        self.create_subscription(
            PointCloud2,
            '/camera2/points', # Assuming this is the correct topic for your lidar
            self.pointcloud_callback,
            qos_profile_sensor_data
        )

        # Publishers (will publish in target_frame)
        self.inlier_pub = self.create_publisher(PointCloud2, 'ground_plane_points', 10)      # Red
        self.outlier_pub = self.create_publisher(PointCloud2, 'non_ground_points', 10)       # Green
        self.projected_pub = self.create_publisher(PointCloud2, 'projected_non_ground_points', 10) # Orange

        # XYZRGB PointFields (used for ROS publishing)
        self.xyzrgb_point_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info(f"GroundPlaneSegmenterNode initialized. Transforming to '{self.target_frame}'.")
        if self.apply_hollow_cylinder_filter:
             self.get_logger().info("Applying hollow cylinder filter BEFORE segmentation.")
             self.get_logger().info(f"  Filter Center: {self.cylinder_center}")
             self.get_logger().info(f"  Outer Radius: {self.outer_cylinder_radius}, Inner Radius: {self.inner_cylinder_radius}, Height: {self.cylinder_height}")
        else:
             self.get_logger().info("Hollow cylinder filter is DISABLED.")
        self.get_logger().info("Publishing: ground (red), non-ground (green), projected non-ground (orange)")
        self.get_logger().info(f"Will save the first filtered & merged cloud to: {os.path.join(self.save_directory, self.save_filename)}")


    def pack_rgb(self, r, g, b):
        """Packs RGB into a float (for ROS PointCloud2 with RGB field)."""
        rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
        packed = struct.pack('I', rgb_int)
        unpacked_float = struct.unpack('f', packed)[0]
        return unpacked_float

    def transform_points(self, points_array, transform_matrix):
        """Applies a 4x4 transformation matrix to an Nx3 numpy array of points."""
        if points_array.shape[0] == 0:
            return points_array
        if points_array.ndim != 2 or points_array.shape[1] != 3:
             self.get_logger().error(f"transform_points expected Nx3 array, got shape {points_array.shape}")
             return np.empty((0,3), dtype=points_array.dtype)

        points_homogeneous = np.hstack((points_array, np.ones((points_array.shape[0], 1))))
        transformed_homogeneous = points_homogeneous @ transform_matrix.T
        return transformed_homogeneous[:, :3]

    def pointcloud_callback(self, cloud_msg: PointCloud2):
        self.get_logger().debug(f"Received point cloud in frame '{cloud_msg.header.frame_id}'")
        current_stamp = cloud_msg.header.stamp
        source_frame = cloud_msg.header.frame_id

        # --- 1. Get Transform ---
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, current_stamp, timeout=Duration(seconds=0.1)
            )
            self.get_logger().debug(f"Successfully found transform from {source_frame} to {self.target_frame}")
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"Could not get transform from '{source_frame}' to '{self.target_frame}' at time {current_stamp.sec}.{current_stamp.nanosec}: {e}. Skipping message."
            )
            self.publish_empty_colored_clouds(self.target_frame, current_stamp)
            return

        # --- 2. Read Raw Points ---
        try:
            points_generator = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
            points_list = [[x, y, z] for x, y, z in points_generator]
            if not points_list:
                 self.get_logger().warn("Point cloud empty (possibly all NaNs or zero points).")
                 self.publish_empty_colored_clouds(self.target_frame, current_stamp)
                 return
            points_orig = np.array(points_list, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Error reading points from PointCloud2: {e}")
            self.publish_empty_colored_clouds(self.target_frame, current_stamp)
            return

        if points_orig.size == 0:
            self.get_logger().warn("Point cloud became empty after reading/conversion.")
            self.publish_empty_colored_clouds(self.target_frame, current_stamp)
            return

        finite_mask = np.all(np.isfinite(points_orig), axis=1)
        points_orig = points_orig[finite_mask]
        if points_orig.size == 0:
            self.get_logger().warn("Point cloud empty after explicit finite check.")
            self.publish_empty_colored_clouds(self.target_frame, current_stamp)
            return

        self.get_logger().info(f"Read {points_orig.shape[0]} finite points from frame '{source_frame}'.")

        # --- 3. Transform Points to Target Frame ---
        t = transform.transform.translation
        q = transform.transform.rotation
        trans_matrix = tf_transformations.translation_matrix([t.x, t.y, t.z])
        rot_matrix = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        transform_matrix = np.dot(trans_matrix, rot_matrix)

        points_transformed = self.transform_points(points_orig, transform_matrix)

        if points_transformed.size == 0:
             self.get_logger().warn("Point cloud empty after transformation.")
             self.publish_empty_colored_clouds(self.target_frame, current_stamp)
             return
        self.get_logger().info(f"Transformed {points_transformed.shape[0]} points to '{self.target_frame}' frame.")

        # --- 4. *** NEW: Apply Hollow Cylinder Filter (in target_frame) *** ---
        points_to_process = points_transformed # Default to using all transformed points
        if self.apply_hollow_cylinder_filter:
            if points_transformed.shape[0] > 0:
                points_relative = points_transformed - self.cylinder_center
                radial_dist_sq = points_relative[:, 0]**2 + points_relative[:, 1]**2
                z_dist = np.abs(points_relative[:, 2])

                # Create boolean masks
                inside_outer_mask = radial_dist_sq <= self.outer_radius_sq
                outside_inner_mask = radial_dist_sq > self.inner_radius_sq
                inside_height_mask = z_dist <= self.half_height

                # Combine masks to find points within the hollow cylinder
                in_hollow_cylinder_mask = inside_outer_mask & outside_inner_mask & inside_height_mask

                points_to_process = points_transformed[in_hollow_cylinder_mask]
                num_original = points_transformed.shape[0]
                num_filtered = points_to_process.shape[0]
                self.get_logger().info(f"Applied hollow cylinder filter: kept {num_filtered} out of {num_original} points.")

                if num_filtered == 0:
                    self.get_logger().warn("Hollow cylinder filter resulted in zero points. Skipping further processing for this cloud.")
                    self.publish_empty_colored_clouds(self.target_frame, current_stamp)
                    # Still mark the first cloud as 'processed' for saving purposes if it was the first one
                    if not self.first_cloud_saved:
                         self.get_logger().warn("This was the first cloud, but it was empty after filtering. Will not save a file.")
                         self.first_cloud_saved = True # Prevent trying to save later clouds
                    return
            else:
                # Should not happen due to earlier checks, but good practice
                self.get_logger().warn("Cannot apply filter, points_transformed is empty.")
                self.publish_empty_colored_clouds(self.target_frame, current_stamp)
                return
        # --- END NEW FILTER SECTION ---

        # --- 5. Open3D Processing (using 'points_to_process' which are filtered points) ---
        pcd = o3d.geometry.PointCloud()
        # *** Use the filtered points ***
        pcd.points = o3d.utility.Vector3dVector(points_to_process)

        # Proceed with downsampling and segmentation as before, but on the filtered cloud
        voxel_size = 0.005
        try:
            if len(pcd.points) < 10:
                 self.get_logger().warn(f"Too few points ({len(pcd.points)}) after filtering/before downsampling. Skipping downsampling and RANSAC.")
                 pcd_downsampled = pcd # Use the (filtered) points directly
            else:
                 pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
                 self.get_logger().info(f"Downsampled filtered cloud to {len(pcd_downsampled.points)} points.")
        except Exception as e:
            self.get_logger().error(f"Error during downsampling (after filtering): {e}")
            self.publish_empty_colored_clouds(self.target_frame, current_stamp)
            return

        if not pcd_downsampled.has_points():
            self.get_logger().warn("Downsampling (after filtering) resulted in an empty point cloud.")
            self.publish_empty_colored_clouds(self.target_frame, current_stamp)
            return

        distance_threshold = 0.001
        mean_ground_z = None
        inlier_points = np.empty((0, 3), dtype=np.float32)
        # Start assuming all *filtered* downsampled points are outliers
        outlier_points = np.asarray(pcd_downsampled.points)

        if len(pcd_downsampled.points) >= 3:
            try:
                plane_model, inliers = pcd_downsampled.segment_plane(
                    distance_threshold=distance_threshold, ransac_n=3, num_iterations=100
                )
                # Separate clouds based on RANSAC result (using filtered, downsampled points)
                ground_cloud = pcd_downsampled.select_by_index(inliers)
                non_ground_cloud = pcd_downsampled.select_by_index(inliers, invert=True)
                inlier_points = np.asarray(ground_cloud.points)
                outlier_points = np.asarray(non_ground_cloud.points)
                self.get_logger().info(
                    f"Segmentation (on filtered cloud): {inlier_points.shape[0]} inliers, "
                    f"{outlier_points.shape[0]} outliers."
                )
                if inlier_points.shape[0] > 0:
                    mean_ground_z = np.mean(inlier_points[:, 2])
                    self.get_logger().info(f"Mean Z of ground plane points (filtered cloud): {mean_ground_z:.4f}")
                else:
                    self.get_logger().warn("No inlier points found in filtered cloud, cannot calculate mean ground Z.")
            except RuntimeError as e:
                self.get_logger().warn(f"RANSAC plane segmentation failed (on filtered cloud): {e}. Treating all filtered points as outliers.")
                # inliers = [] # Already handled by default initialization
                inlier_points = np.empty((0, 3), dtype=np.float32)
                # outlier_points remains all downsampled points
        else:
             self.get_logger().warn(f"Skipping RANSAC due to insufficient points ({len(pcd_downsampled.points)} < 3) in filtered cloud. Treating all as outliers.")

        # --- Calculate Projected Points (based on segmentation of filtered cloud) ---
        projected_points = np.empty((0, 3), dtype=np.float32)
        if outlier_points.shape[0] > 0 and mean_ground_z is not None:
            projected_points = outlier_points.copy()
            projected_points[:, 2] = mean_ground_z
            self.get_logger().debug(f"Created {projected_points.shape[0]} projected non-ground points at Z={mean_ground_z:.4f}.")
        elif outlier_points.shape[0] > 0 and mean_ground_z is None:
             self.get_logger().warn("Cannot project non-ground points because mean ground Z was not calculated (from filtered cloud).")
        # No else needed if outlier_points is empty

        # --- 6. MERGE AND SAVE FIRST *FILTERED* CLOUD ---
        if not self.first_cloud_saved:
            try:
                # Log message updated to reflect filtering
                self.get_logger().info("Processing the first *filtered* point cloud for merging and saving...")
                points_to_merge = []
                colors_to_merge = [] # Use float colors 0-1 for Open3D

                # Add ground points (Red) - derived from filtered cloud
                if inlier_points.shape[0] > 0:
                    points_to_merge.append(inlier_points)
                    colors_to_merge.append(np.full((inlier_points.shape[0], 3), [1.0, 0.0, 0.0])) # Red
                    self.get_logger().debug(f"Adding {inlier_points.shape[0]} ground points to merge (from filtered).")

                # Add non-ground points (Green) - derived from filtered cloud
                if outlier_points.shape[0] > 0:
                    points_to_merge.append(outlier_points)
                    colors_to_merge.append(np.full((outlier_points.shape[0], 3), [0.0, 1.0, 0.0])) # Green
                    self.get_logger().debug(f"Adding {outlier_points.shape[0]} non-ground points to merge (from filtered).")

                # Add projected non-ground points (Orange) - derived from filtered cloud
                if projected_points.shape[0] > 0:
                    points_to_merge.append(projected_points)
                    colors_to_merge.append(np.full((projected_points.shape[0], 3), [1.0, 0.65, 0.0])) # Orange
                    self.get_logger().debug(f"Adding {projected_points.shape[0]} projected points to merge (from filtered).")

                if points_to_merge: # Check if there is anything to merge
                    merged_points_array = np.vstack(points_to_merge)
                    merged_colors_array = np.vstack(colors_to_merge)

                    merged_pcd = o3d.geometry.PointCloud()
                    merged_pcd.points = o3d.utility.Vector3dVector(merged_points_array)
                    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors_array)

                    save_path = os.path.join(self.save_directory, self.save_filename)
                    o3d.io.write_point_cloud(save_path, merged_pcd)
                    # Log message updated
                    self.get_logger().info(f"Successfully saved merged *filtered* point cloud ({merged_points_array.shape[0]} points) to {save_path}")

                else:
                     # Log message updated
                    self.get_logger().warn("First cloud processed, but no points found after filtering/segmentation to merge and save.")

                self.first_cloud_saved = True

            except Exception as e:
                # Log message updated
                self.get_logger().error(f"Failed to merge or save the first *filtered* point cloud: {e}", exc_info=True)
                self.first_cloud_saved = True
        # --- *** END SAVE SECTION *** ---


        # --- 7. Create and Publish Individual Point Clouds (based on filtered/segmented results) ---
        output_header = Header()
        output_header.stamp = current_stamp
        output_header.frame_id = self.target_frame # All published clouds are in target_frame

        # Publish RED inliers (Ground)
        if inlier_points.shape[0] > 0:
            red_packed = self.pack_rgb(255, 0, 0)
            inlier_colors = np.full((inlier_points.shape[0], 1), red_packed, dtype=np.float32)
            inlier_data_combined = np.hstack((inlier_points, inlier_colors))
            inlier_data_list = inlier_data_combined.tolist()
            inlier_cloud_msg = pc2.create_cloud(output_header, self.xyzrgb_point_fields, inlier_data_list)
            self.inlier_pub.publish(inlier_cloud_msg)
            self.get_logger().debug(f"Published {inlier_points.shape[0]} ground points (red) from filtered cloud.")
        else:
            inlier_cloud_msg = pc2.create_cloud(output_header, self.xyzrgb_point_fields, [])
            self.inlier_pub.publish(inlier_cloud_msg)
            self.get_logger().debug(f"Published empty ground points cloud (filtered).")

        # Publish GREEN outliers (Non-Ground)
        if outlier_points.shape[0] > 0:
            green_packed = self.pack_rgb(0, 255, 0)
            outlier_colors = np.full((outlier_points.shape[0], 1), green_packed, dtype=np.float32)
            outlier_data_combined = np.hstack((outlier_points, outlier_colors))
            outlier_data_list = outlier_data_combined.tolist()
            outlier_cloud_msg = pc2.create_cloud(output_header, self.xyzrgb_point_fields, outlier_data_list)
            self.outlier_pub.publish(outlier_cloud_msg)
            self.get_logger().debug(f"Published {outlier_points.shape[0]} non-ground points (green) from filtered cloud.")
        else:
            outlier_cloud_msg = pc2.create_cloud(output_header, self.xyzrgb_point_fields, [])
            self.outlier_pub.publish(outlier_cloud_msg)
            self.get_logger().debug(f"Published empty non-ground points cloud (filtered).")

        # Publish ORANGE Projected Non-Ground Points
        if projected_points.shape[0] > 0:
            orange_packed = self.pack_rgb(255, 165, 0)
            projected_colors = np.full((projected_points.shape[0], 1), orange_packed, dtype=np.float32)
            projected_data_combined = np.hstack((projected_points, projected_colors))
            projected_data_list = projected_data_combined.tolist()
            projected_cloud_msg = pc2.create_cloud(output_header, self.xyzrgb_point_fields, projected_data_list)
            self.projected_pub.publish(projected_cloud_msg)
            self.get_logger().debug(f"Published {projected_points.shape[0]} projected non-ground points (orange) from filtered cloud.")
        else:
            projected_cloud_msg = pc2.create_cloud(output_header, self.xyzrgb_point_fields, [])
            self.projected_pub.publish(projected_cloud_msg)
            self.get_logger().debug(f"Published empty projected non-ground points cloud (filtered).")


    def publish_empty_colored_clouds(self, frame_id, stamp):
        """Publishes empty clouds on all output topics in the specified frame."""
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        empty_cloud_msg = pc2.create_cloud(header, self.xyzrgb_point_fields, [])
        self.inlier_pub.publish(empty_cloud_msg)
        self.outlier_pub.publish(empty_cloud_msg)
        self.projected_pub.publish(empty_cloud_msg)
        self.get_logger().debug(f"Published empty point clouds on all topics in {frame_id}.")


def main(args=None):
    rclpy.init(args=args)
    node = GroundPlaneSegmenterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        node.get_logger().fatal(f"Unhandled exception in main spin: {e}", exc_info=True)
    finally:
        if node:
             node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()
    print("Node shutdown complete.")


if __name__ == '__main__':
    main()