import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as ROSPoint
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import sys
import traceback # Import traceback

class CrackPathNode(Node):
    def __init__(self):
        super().__init__('crack_path_extractor')
        self.create_subscription(
            PointCloud2,
            '/camera2/points',
            self.pointcloud_callback,
            qos_profile_sensor_data
        )
        self.marker_pub = self.create_publisher(Marker, 'crack_path_marker', 10)
        self.crack_path_points = []
        self.get_logger().info("CrackPathNode initialized: waiting for point cloud data...")

    def pointcloud_callback(self, cloud_msg: PointCloud2):
        """Callback function to process incoming PointCloud2 messages."""
        try:
            # --- REVISED CONVERSION AGAIN ---
            try:
                # Attempt to read as numpy array
                raw_points_data = pc2.read_points_numpy(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
                self.get_logger().debug(f"Initial read_points_numpy dtype: {raw_points_data.dtype}, shape: {raw_points_data.shape}")

            except ValueError as e:
                # This might happen if fields aren't found etc.
                self.get_logger().error(f"Error reading points with read_points_numpy: {e}")
                self.get_logger().info(f"Available fields: {[f.name for f in cloud_msg.fields]}")
                 # Fallback or alternative reading method could go here if needed
                return
            except Exception as e:
                self.get_logger().error(f"Unexpected error during pc2.read_points_numpy: {e}")
                self.get_logger().error(traceback.format_exc())
                return

            # Check if data is empty
            if raw_points_data.size == 0:
                 self.get_logger().warn("Received empty point cloud or all NaNs after filtering.")
                 return

            # --- Check if it's structured or plain ---
            if raw_points_data.dtype.names:
                # It IS a structured array - extract fields 'x', 'y', 'z'
                self.get_logger().debug(f"Data is structured. Extracting fields: {raw_points_data.dtype.names}")
                # Verify required fields exist before accessing
                required_fields = {'x', 'y', 'z'}
                available_fields = set(raw_points_data.dtype.names)
                if not required_fields.issubset(available_fields):
                    self.get_logger().error(f"Structured array missing required fields. Found: {available_fields}, Need: {required_fields}")
                    return
                try:
                   points = np.stack([raw_points_data['x'], raw_points_data['y'], raw_points_data['z']], axis=-1)
                except KeyError as e:
                    # Should be caught by the check above, but as a safeguard
                    self.get_logger().error(f"KeyError accessing structured array field: {e}. Available fields: {available_fields}")
                    return

            elif raw_points_data.ndim == 2 and raw_points_data.shape[1] == 3:
                # It's NOT structured, but looks like an Nx3 array already
                self.get_logger().debug("Data is not structured, assuming it's already Nx3.")
                points = raw_points_data
            else:
                # It's neither structured nor Nx3 - unexpected format
                self.get_logger().error(f"Point data format unexpected. Not structured and not Nx3. Dtype: {raw_points_data.dtype}, Shape: {raw_points_data.shape}")
                return

            # Ensure float32 type for consistency downstream
            # This should work fine now whether 'points' came from stack or was raw_points_data
            points = points.astype(np.float32)
            # --- END REVISED CONVERSION ---


            # --- Standard checks and processing ---
            if points.ndim != 2 or points.shape[1] != 3:
                 # This check should technically be redundant now due to logic above, but safe to keep
                self.get_logger().error(f"Processed points have unexpected shape: {points.shape}. Expected Nx3.")
                return

            if points.shape[0] == 0:
                self.get_logger().warn("Point cloud resulted in 0 points after processing.")
                return

            self.get_logger().info(f"Processing point cloud with {points.shape[0]} points.")

            # Convert NumPy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # --- Rest of your processing code (Downsample, Segment, Cluster, Order, Publish) ---
            # (Keep the code from the previous correct version here)
            # Downsample
            voxel_size = 200000
            self.get_logger().info(f"Downsampling with voxel size: {voxel_size} meters")
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

            if not pcd_downsampled.has_points():
                self.get_logger().warn(f"Voxel downsampling resulted in zero points.")
                return
            pcd_to_process = pcd_downsampled
            self.get_logger().info(f"Downsampled cloud has {len(pcd_to_process.points)} points.")

            # Segment plane
            distance_threshold = 0.01
            num_iterations = 1000
            self.get_logger().info(f"Segmenting plane with distance threshold: {distance_threshold}, iterations: {num_iterations}")
            plane_model, inliers = pcd_to_process.segment_plane(distance_threshold=distance_threshold,
                                                               ransac_n=3, num_iterations=num_iterations)

            if not inliers:
                 self.get_logger().warn("RANSAC could not segment a plane. Processing all points as non-ground.")
                 non_ground_cloud = pcd_to_process
            else:
                self.get_logger().info(f"Plane segmentation found {len(inliers)} inliers.")
                non_ground_cloud = pcd_to_process.select_by_index(inliers, invert=True)

            if not non_ground_cloud.has_points():
                self.get_logger().warn("No points remaining after plane removal.")
                return

            remaining_points = np.asarray(non_ground_cloud.points)
            self.get_logger().info(f"Found {remaining_points.shape[0]} non-ground points.")

            # Cluster
            eps_distance = 0.02
            min_points = 10
            self.get_logger().info(f"Clustering with eps: {eps_distance}, min_points: {min_points}")
            labels = np.array(non_ground_cloud.cluster_dbscan(eps=eps_distance, min_points=min_points, print_progress=False))

            max_label = labels.max()
            if max_label < 0:
                self.get_logger().warn(f"DBSCAN found no clusters (only noise points labeled -1). Point count: {remaining_points.shape[0]}")
                return
            self.get_logger().info(f"DBSCAN found {max_label + 1} clusters.")

            # Largest cluster
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            if unique_labels.size == 0:
                 self.get_logger().warn("No valid clusters found after filtering noise.")
                 return
            largest_cluster_id = unique_labels[np.argmax(counts)]
            self.get_logger().info(f"Largest cluster is ID {largest_cluster_id} with {counts.max()} points.")

            crack_indices = np.where(labels == largest_cluster_id)[0]
            crack_cloud = non_ground_cloud.select_by_index(crack_indices)
            crack_points = np.asarray(crack_cloud.points)

            if crack_points.shape[0] < 2:
                self.get_logger().warn(f"Largest crack cluster (ID {largest_cluster_id}) has only {crack_points.shape[0]} points.")
                return

            # Path Ordering
            start_idx = np.argmin(crack_points[:, 0])
            path_order = [start_idx]
            used = {start_idx}
            current_idx = start_idx
            tree = o3d.geometry.KDTreeFlann(crack_cloud)
            while len(used) < crack_points.shape[0]:
                k_neighbors = min(10, crack_points.shape[0])
                [k, idxs, _] = tree.search_knn_vector_3d(crack_cloud.points[current_idx], k=k_neighbors)
                next_idx = -1
                for i in range(1, k):
                    neighbor_idx = idxs[i]
                    if neighbor_idx not in used:
                        next_idx = neighbor_idx
                        break
                if next_idx == -1:
                    self.get_logger().warn(f"Path ordering stopped prematurely.")
                    break
                path_order.append(next_idx)
                used.add(next_idx)
                current_idx = next_idx

            ordered_path = crack_points[path_order]
            self.crack_path_points = [(float(p[0]), float(p[1]), float(p[2])) for p in ordered_path]

            if not self.crack_path_points:
                 self.get_logger().warn("Path ordering resulted in an empty path.")
                 return
            self.get_logger().info(f"Crack path ({len(self.crack_path_points)} points) extracted.")

            # Publish Marker
            marker = Marker()
            marker.header.frame_id = cloud_msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "crack_path"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.color.a = 1.0
            marker.color.r = 1.0; marker.color.g = 0.8; marker.color.b = 0.0
            marker.points = [ROSPoint(x=p[0], y=p[1], z=p[2]) for p in self.crack_path_points]

            if not marker.points:
                 self.get_logger().error("Marker has no points, skipping publish.")
            else:
                 self.marker_pub.publish(marker)
                 self.get_logger().info(f"Published crack path marker with {len(marker.points)} points.")


        except Exception as e:
            self.get_logger().error(f"Unhandled error processing point cloud: {e}")
            self.get_logger().error(traceback.format_exc()) # Log the full traceback


# --- main function remains the same ---
def main(args=None):
    rclpy.init(args=args)
    node = CrackPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    except Exception as e:
        node.get_logger().error(f"Error during spin: {e}")
        node.get_logger().error(traceback.format_exc())
    finally:
        if rclpy.ok():
             node.destroy_node()
             rclpy.shutdown()

if __name__ == '__main__':
    main()