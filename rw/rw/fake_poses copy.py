#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
import numpy as np
from scipy.spatial import KDTree # For nearest neighbor search
import tf_transformations # For quaternion from euler
import math

class PathGeneratorNode(Node):
    def __init__(self):
        super().__init__('path_generator_publisher_node')

        # --- Parameters ---
        self.input_topic = '/projected_non_ground_points'
        self.output_topic = '/arm_path_poses' # Topic for the robot arm controller
        self.z_offset = 0.025
        # Fixed orientation for the end-effector (roll=pi, pitch=0, yaw=0) -> Tool pointing down
        # Adjust roll, pitch, yaw as needed for your robot's base frame and task
        self.target_roll = math.pi
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.target_orientation_quat = tf_transformations.quaternion_from_euler(
            self.target_roll, self.target_pitch, self.target_yaw
        )
        self.target_orientation = Quaternion(
            x=self.target_orientation_quat[0],
            y=self.target_orientation_quat[1],
            z=self.target_orientation_quat[2],
            w=0.0
        )
        self.output_frame_id = "base_link" # IMPORTANT: Frame ID the poses are relative to
        # --- End Parameters ---

        # Flag to prevent processing if the previous path generation is somehow still running (unlikely here)
        self._is_processing = False

        # Define QoS profile for the publisher to match potential subscribers like MoveIt
        pose_array_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Ensure delivery
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # Only keep the latest path
        )

        # Publisher for the PoseArray path
        self.pose_array_publisher = self.create_publisher(
            PoseArray,
            self.output_topic,
            pose_array_qos)

        # Subscriber for the input point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data # Use sensor data QoS for input
        )

        self.get_logger().info(f"PathGeneratorNode initialized.")
        self.get_logger().info(f"Subscribing to PointCloud2 on: '{self.input_topic}'")
        self.get_logger().info(f"Publishing PoseArray on: '{self.output_topic}' in frame '{self.output_frame_id}'")
        self.get_logger().info(f"Applying Z offset: {self.z_offset}")
        self.get_logger().info(f"Using fixed orientation (RPY): ({self.target_roll:.2f}, {self.target_pitch:.2f}, {self.target_yaw:.2f})")


    def pointcloud_callback(self, cloud_msg: PointCloud2):
        if self._is_processing:
            self.get_logger().warn("Still processing previous cloud, skipping new message.")
            return

        self._is_processing = True
        self.get_logger().info(f"Received PointCloud2 message on {self.input_topic}. Processing...")
        processing_successful = False
        try:
            # --- 1. Read Points ---
            points_generator = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
            projected_points_list = [[x, y, z] for x, y, z in points_generator]

            if not projected_points_list:
                 self.get_logger().warn("Received empty point cloud (or all NaNs). Cannot generate path.")
                 self._is_processing = False
                 return # Don't publish anything

            projected_points_np = np.array(projected_points_list, dtype=np.float32)
            self.get_logger().info(f"Extracted {projected_points_np.shape[0]} points.")

            finite_mask = np.all(np.isfinite(projected_points_np), axis=1)
            projected_points_np = projected_points_np[finite_mask]

            if projected_points_np.shape[0] < 2:
                self.get_logger().warn(f"Need at least 2 points to generate a path, found {projected_points_np.shape[0]}. Skipping.")
                self._is_processing = False
                return # Don't publish anything

            # --- 2. Generate Path using Nearest Neighbors ---
            path_indices = []
            self.get_logger().info(f"Starting path generation for {len(projected_points_np)} points...")
            num_points = len(projected_points_np)

            distances_from_origin = np.linalg.norm(projected_points_np, axis=1)
            start_index = np.argmax(distances_from_origin)
            self.get_logger().info(f"Starting path at point index {start_index}")

            kdtree = KDTree(projected_points_np)
            visited = np.zeros(num_points, dtype=bool)
            path_indices = []

            current_index = start_index
            path_indices.append(current_index)
            visited[current_index] = True

            for i in range(num_points - 1):
                distances, indices = kdtree.query(projected_points_np[current_index], k=num_points, distance_upper_bound=np.inf)
                next_index = -1
                for neighbor_idx in indices:
                    if neighbor_idx >= 0 and neighbor_idx < num_points:
                        if neighbor_idx != current_index and not visited[neighbor_idx]:
                            next_index = neighbor_idx
                            break
                if next_index != -1:
                    path_indices.append(next_index)
                    visited[next_index] = True
                    current_index = next_index
                else:
                    self.get_logger().warn(f"Path generation stopped early. Found path for {len(path_indices)}/{num_points} points.")
                    break

            self.get_logger().info(f"Path generation complete. Path sequence length: {len(path_indices)}")
            if len(path_indices) != num_points:
                 self.get_logger().warn("Full path covering all points not found.")

            # --- 3. Reorder Points and Modify Z ---
            if not path_indices:
                self.get_logger().error("Path generation resulted in empty path_indices. Cannot proceed.")
                self._is_processing = False
                return

            sorted_points = projected_points_np[path_indices]
            self.get_logger().info(f"Applying Z offset ({self.z_offset}) to {sorted_points.shape[0]} sorted points.")
            sorted_points[:, 2] += self.z_offset

            # --- 4. Convert to PoseArray and Publish ---
            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = self.get_clock().now().to_msg()
            # IMPORTANT: Use the frame the points are actually in. This should match the 'target_frame'
            # from the previous node and the frame the robot planner expects.
            pose_array_msg.header.frame_id = self.output_frame_id

            for point in sorted_points:
                pose = Pose()
                pose.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                pose.orientation = self.target_orientation # Use the pre-calculated fixed orientation
                pose_array_msg.poses.append(pose)

            self.pose_array_publisher.publish(pose_array_msg)
            self.get_logger().info(f"Published PoseArray with {len(pose_array_msg.poses)} poses on {self.output_topic}")
            processing_successful = True

        except Exception as e:
            self.get_logger().error(f"Error during point cloud processing or publishing: {e}", exc_info=True)

        finally:
            self._is_processing = False # Release the lock

def main(args=None):
    rclpy.init(args=args)
    node = PathGeneratorNode()
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
    print("PathGeneratorNode shutdown complete.")

if __name__ == '__main__':
    main()