#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Pose, PointStamped
from xarm_msgs.srv import PlanSingleStraight, PlanExec
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 # For reading point cloud data
import numpy as np
from scipy.spatial import KDTree # For nearest neighbor search
import tf2_ros
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs # For easy point transformation
import sys
import time
import math
import traceback # Import traceback module for logging exceptions

# --- Configuration ---
POINT_CLOUD_TOPIC = "/projected_non_ground_points" #"/projected_non_ground_points" # Topic to subscribe to
PLANNING_FRAME = 'link_base'                # Target frame for planning (usually robot base)
WAIT_FOR_TRANSFORM_SEC = 5.0               # Max time to wait for TF transform
POINT_CLOUD_PROCESS_TIMEOUT_SEC = 20.0     # Max time to wait for a point cloud message
MIN_POINTS_FOR_PATH = 2                    # Minimum number of points required to generate a path
INTER_POSE_DELAY_SEC = 1.5                 # Delay between reaching poses
PLAN_TIMEOUT_SEC = 15.0                    # Timeout for planning service call
EXEC_TIMEOUT_SEC = 60.0                    # Timeout for execution service call (adjust based on move speed/distance)

# Service names provided by xarm_planner_node
STRAIGHT_PLAN_SERVICE = '/xarm_straight_plan'
EXEC_PLAN_SERVICE = '/xarm_exec_plan'
# --- End Configuration ---

class PointCloudPlannerClient(Node):

    def __init__(self):
        super().__init__('pointcloud_planner_client')
        self.get_logger().info("Initializing PointCloud Planner Client...")

        # --- Service Clients ---
        self.straight_plan_client = self.create_client(PlanSingleStraight, STRAIGHT_PLAN_SERVICE)
        self.exec_plan_client = self.create_client(PlanExec, EXEC_PLAN_SERVICE)
        self._check_services_ready() # Check availability early

        # --- TF Listener ---
        self.tf_buffer = Buffer()
        # If using SingleThreadedExecutor for the node, TF lookups must happen when spinning.
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info("TF listener initialized.")

        # --- Point Cloud Subscriber ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Or BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            POINT_CLOUD_TOPIC,
            self.point_cloud_callback,
            qos_profile
        )
        self.get_logger().info(f"Subscribed to {POINT_CLOUD_TOPIC}")

        # --- State Variables ---
        self._point_cloud_data = None
        self._point_cloud_frame = None
        self._processing_triggered = False
        self._path_generated = False
        self._target_poses = []

        # --- Fixed Orientation ---
        # Roll=pi, Pitch=0, Yaw=0 -> Quaternion x=1,y=0,z=0,w=0
        self.fixed_orientation = Pose().orientation
        self.fixed_orientation.x = 1.0
        self.fixed_orientation.y = 0.0
        self.fixed_orientation.z = 0.0
        self.fixed_orientation.w = 0.0
        # Note: This orientation points the Z-axis of the tool frame along the negative X-axis of the base frame.
        # Verify this is the desired orientation for your task.

        self.get_logger().info("Node initialized. Waiting for PointCloud data...")

    def _check_services_ready(self):
        """Checks if required services are available and exits if not."""
        services_ok = True
        if not self.straight_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {STRAIGHT_PLAN_SERVICE} not available.')
            services_ok = False
        if not self.exec_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {EXEC_PLAN_SERVICE} not available.')
            services_ok = False

        if not services_ok:
            self.get_logger().fatal("Essential planning services not available. Shutting down.")
            # Use sys.exit directly for immediate shutdown
            sys.exit(1)
        self.get_logger().info("Planning service clients ready.")

    # --- Point Cloud Callback ---
    def point_cloud_callback(self, msg: PointCloud2):
        """Stores the first valid PointCloud2 message received."""
        if self._point_cloud_data is not None or self._processing_triggered:
            return

        self.get_logger().info(f"Received PointCloud2 message with {msg.height * msg.width} points from frame '{msg.header.frame_id}'.")
        try:
            field_names = [field.name for field in msg.fields]
            if not all(f in field_names for f in ['x', 'y', 'z']):
                 self.get_logger().error(f"PointCloud is missing required fields 'x', 'y', or 'z'. Found fields: {field_names}")
                 return

            # Correctly parse point cloud data
            gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points_list_structured = list(gen)

            if not points_list_structured:
                self.get_logger().warn("Received point cloud is empty or contains only NaNs. Waiting for valid data.")
                return

            # Convert list of structured points/tuples into a list of [x, y, z] lists
            points_xyz = [[p[0], p[1], p[2]] for p in points_list_structured]

            # Now create the NumPy array
            self._point_cloud_data = np.array(points_xyz, dtype=np.float32)

            self._point_cloud_frame = msg.header.frame_id
            self._processing_triggered = True
            self.get_logger().info(f"Successfully parsed {len(self._point_cloud_data)} valid points. Triggering path generation.")

        except Exception as e:
            tb_str = traceback.format_exc() # Get traceback string
            self.get_logger().error(f"Failed to read points from PointCloud2 message: {e}\n{tb_str}") # Log message + traceback
            self._point_cloud_data = None
            self._point_cloud_frame = None
            self._processing_triggered = False


    # --- Coordinate Transformation ---
    def transform_points(self, points_np: np.ndarray, source_frame: str, target_frame: str) -> np.ndarray | None:
        """Transforms a NumPy array of points from source_frame to target_frame."""
        if points_np is None or len(points_np) == 0:
            self.get_logger().error("Cannot transform points: input array is empty or None.")
            return None
        if source_frame == target_frame:
             self.get_logger().info(f"Source and target frames ('{source_frame}') are the same. No transformation needed.")
             return points_np

        transformed_points = []
        try:
            # Use time=rclpy.time.Time(seconds=0) for latest available transform
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(seconds=0),
                rclpy.duration.Duration(seconds=WAIT_FOR_TRANSFORM_SEC)
            )
            self.get_logger().info(f"Successfully looked up transform from '{source_frame}' to '{target_frame}'.")

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"Could not get transform from '{source_frame}' to '{target_frame}': {e}\n{tb_str}")
            return None

        # Apply transform to each point
        for point in points_np:
            ps = PointStamped()
            ps.header.stamp = self.get_clock().now().to_msg() # Stamp recommended
            ps.header.frame_id = source_frame
            ps.point.x = float(point[0])
            ps.point.y = float(point[1])
            ps.point.z = float(point[2])
            try:
                transformed_ps = tf2_geometry_msgs.do_transform_point(ps, transform_stamped)
                transformed_points.append([
                    transformed_ps.point.x,
                    transformed_ps.point.y,
                    transformed_ps.point.z
                ])
            except Exception as e:
                tb_str = traceback.format_exc()
                self.get_logger().error(f"Failed to transform point {point} after getting transform: {e}\n{tb_str}")
                return None # Abort if any point fails

        if not transformed_points:
             self.get_logger().error("Transformation resulted in an empty list of points.")
             return None

        return np.array(transformed_points, dtype=np.float32)


    # --- Path Generation ---
    def generate_path_from_points(self, points_np: np.ndarray) -> bool:
        """Generates a path order using the nearest neighbor approach and stores poses."""
        if points_np is None or len(points_np) < MIN_POINTS_FOR_PATH:
            self.get_logger().warn(f"Not enough points ({len(points_np) if points_np is not None else 0}) for path generation (minimum {MIN_POINTS_FOR_PATH}). Cannot generate path.")
            return False

        num_points = len(points_np)
        self.get_logger().info(f"Generating path for {num_points} points using nearest neighbor...")

        # Find starting point (farthest from origin)
        if points_np.ndim == 1: # Should not happen with check above, but safe
            points_np = points_np.reshape(1, -1)
        distances_from_origin = np.linalg.norm(points_np, axis=1)
        start_index = np.argmax(distances_from_origin)
        self.get_logger().info(f"Starting path at point index {start_index} (coord: {points_np[start_index]}) - farthest from origin.")

        # Build path using KDTree
        try:
            kdtree = KDTree(points_np)
        except ValueError as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"Failed to create KDTree: {e}\n{tb_str}")
            return False

        visited = np.zeros(num_points, dtype=bool)
        path_indices = []
        current_index = start_index
        path_indices.append(current_index)
        visited[current_index] = True

        for _ in range(num_points - 1): # N-1 steps to connect N points
            try:
                 # Query neighbors, k=num_points ensures we check all others if needed
                 distances, indices = kdtree.query(points_np[current_index], k=num_points)
            except Exception as e:
                 tb_str = traceback.format_exc()
                 self.get_logger().error(f"KDTree query failed from index {current_index}: {e}\n{tb_str}")
                 return False # Cannot continue

            next_index = -1
            found_unvisited = False
            # Iterate neighbors (skip index 0, which is the point itself)
            for neighbor_idx in indices[1:]:
                 if not visited[neighbor_idx]:
                     next_index = neighbor_idx
                     found_unvisited = True
                     break # Found nearest unvisited

            if found_unvisited:
                path_indices.append(next_index)
                visited[next_index] = True
                current_index = next_index
            else:
                # Stop if no unvisited neighbor found
                if np.all(visited):
                     self.get_logger().info("All points visited. Path generation complete.")
                else:
                     self.get_logger().warn(f"Could not find an unvisited neighbor from point index {current_index}. Path might be incomplete. Visited: {np.where(visited)[0]}")
                break

        self.get_logger().info(f"Path generation finished. Sequence (indices): {path_indices}")

        # Convert ordered points to poses
        ordered_points = points_np[path_indices]
        self._target_poses = [self.create_pose_from_point(pt) for pt in ordered_points]

        if not self._target_poses:
             self.get_logger().error("Failed to create any target poses from the ordered points.")
             return False

        self.get_logger().info(f"Generated {len(self._target_poses)} target poses.")
        self._path_generated = True
        return True


    # --- Pose Creation ---
    def create_pose_from_point(self, point_xyz: np.ndarray) -> Pose:
        """Creates a geometry_msgs/Pose from an XYZ point using the fixed orientation."""
        pose = Pose()
        pose.position.x = float(point_xyz[0])
        pose.position.y = float(point_xyz[1])
        pose.position.z = float(point_xyz[2])
        pose.orientation = self.fixed_orientation # Use the pre-defined orientation
        return pose

    # --- Planning and Execution ---
    def plan_and_execute_straight(self, target_pose: Pose, pose_index: int, total_poses: int) -> bool:
        """Plans a straight-line Cartesian path and executes it. Returns True on success."""
        pos = target_pose.position
        # Log target pose with index, handle -1 for return move logging
        log_index_str = f"{pose_index + 1}/{total_poses}" if pose_index >= 0 else "Return"
        self.get_logger().info(f"--- Planning Pose {log_index_str} ---")
        self.get_logger().info(f"Target: P(x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f})")

        # --- 1. Plan the Straight Path ---
        plan_req = PlanSingleStraight.Request()
        plan_req.target = target_pose
        plan_future = self.straight_plan_client.call_async(plan_req)

        try:
            # Spin until future is complete using the node's default executor
            rclpy.spin_until_future_complete(self, plan_future, timeout_sec=PLAN_TIMEOUT_SEC)
        except Exception as e:
             tb_str = traceback.format_exc()
             self.get_logger().error(f'Exception while waiting for planning service response: {e}\n{tb_str}')
             return False

        if plan_future.result() is None:
            self.get_logger().error(f'Planning service call for {STRAIGHT_PLAN_SERVICE} timed out after {PLAN_TIMEOUT_SEC} sec.')
            return False
        if not plan_future.done() or plan_future.exception() is not None:
            # Log the exception if available
            exc = plan_future.exception()
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else "No exception details."
            self.get_logger().error(f'Exception during planning service call: {exc}\n{tb_str}')
            return False

        plan_res = plan_future.result()
        # === FIX: Removed access to non-existent '.message' field ===
        if not plan_res.success:
            self.get_logger().error(f"Cartesian planning failed! Service response success={plan_res.success}")
            return False
        # ===========================================================

        self.get_logger().info("Cartesian planning successful.")

        # --- 2. Execute the Plan ---
        exec_req = PlanExec.Request()
        exec_req.wait = True # Wait for execution to finish before returning
        exec_future = self.exec_plan_client.call_async(exec_req)

        try:
            rclpy.spin_until_future_complete(self, exec_future, timeout_sec=EXEC_TIMEOUT_SEC)
        except Exception as e:
             tb_str = traceback.format_exc()
             self.get_logger().error(f'Exception while waiting for execution service response: {e}\n{tb_str}')
             return False

        if exec_future.result() is None:
            self.get_logger().error(f'Execution service call for {EXEC_PLAN_SERVICE} timed out after {EXEC_TIMEOUT_SEC} sec.')
            return False
        if not exec_future.done() or exec_future.exception() is not None:
            # Log the exception if available
            exc = exec_future.exception()
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else "No exception details."
            self.get_logger().error(f'Exception during execution service call: {exc}\n{tb_str}')
            return False

        exec_res = exec_future.result()
        # === FIX: Removed access to non-existent '.message' field ===
        if not exec_res.success:
            self.get_logger().error(f"Plan execution failed! Service response success={exec_res.success}")
            return False
        # ===========================================================

        self.get_logger().info(f"Pose {log_index_str} execution successful.")
        return True

    # --- Main Execution Logic ---
    def run(self):
        """Main loop: waits for data, processes, generates path, and executes."""
        # 1. Wait for point cloud data via callback
        self.get_logger().info(f"Waiting up to {POINT_CLOUD_PROCESS_TIMEOUT_SEC} seconds for a valid PointCloud on '{POINT_CLOUD_TOPIC}'...")
        start_wait_time = self.get_clock().now()
        while rclpy.ok() and not self._processing_triggered:
            rclpy.spin_once(self, timeout_sec=0.1) # Check for callbacks/timers
            elapsed_time = (self.get_clock().now() - start_wait_time).nanoseconds / 1e9
            if elapsed_time > POINT_CLOUD_PROCESS_TIMEOUT_SEC:
                self.get_logger().error(f"Timeout: No valid point cloud data received within {POINT_CLOUD_PROCESS_TIMEOUT_SEC} seconds.")
                return # Exit run method

        if not rclpy.ok():
             self.get_logger().info("RCLPY shutdown requested during wait for point cloud.")
             return

        # At this point, _processing_triggered should be True
        self.get_logger().info("Point cloud received. Proceeding with processing...")

        # 2. Transform Points
        self.get_logger().info(f"Transforming points from '{self._point_cloud_frame}' to '{PLANNING_FRAME}'...")
        transformed_points_np = self.transform_points(self._point_cloud_data, self._point_cloud_frame, PLANNING_FRAME)

        if transformed_points_np is None:
            self.get_logger().error("Failed to transform points. Cannot proceed.")
            return # Exit run method

        self.get_logger().info(f"Successfully transformed {len(transformed_points_np)} points.")

        # 3. Generate Path
        if not self.generate_path_from_points(transformed_points_np):
            self.get_logger().error("Failed to generate path from points. Cannot proceed.")
            return # Exit run method

        # 4. Execute Path Sequence
        if not self._path_generated or not self._target_poses:
             self.get_logger().error("Path generation failed or produced no poses. Cannot execute.")
             return

        self.get_logger().info(f"Starting execution of {len(self._target_poses)} poses...")
        total_poses = len(self._target_poses)
        start_pose = None # Store the first pose to return to it optionally

        execution_successful = True # Track overall success
        for i, pose in enumerate(self._target_poses):
            if i == 0:
                start_pose = pose # Save the first pose

            success = self.plan_and_execute_straight(pose, i, total_poses)
            if not success:
                self.get_logger().error(f"Failed to reach pose {i+1}. Aborting sequence.")
                execution_successful = False
                break # Stop executing the sequence

            if i < total_poses - 1: # Don't pause after the last pose or if aborted
                self.get_logger().info(f"Pausing for {INTER_POSE_DELAY_SEC} seconds...")
                time.sleep(INTER_POSE_DELAY_SEC)

        # Optional: Return to the first pose only if the main sequence was successful
        if execution_successful and start_pose and total_poses > 1:
             self.get_logger().info("--- Returning to the start pose of the sequence ---")
             # Use index -1 for logging clarity when returning
             success = self.plan_and_execute_straight(start_pose, -1, total_poses)
             if success:
                  self.get_logger().info("Successfully returned to the start pose.")
             else:
                  # Don't mark overall execution as failed just because return failed
                  self.get_logger().warn("Failed to return to the start pose.")

        if execution_successful:
             self.get_logger().info("Point cloud planning and execution sequence finished successfully.")
        else:
             self.get_logger().error("Point cloud planning and execution sequence failed.")


def main(args=None):
    rclpy.init(args=args)

    planner_client_node = None
    try:
        planner_client_node = PointCloudPlannerClient()
        # The run method now encapsulates the main logic
        planner_client_node.run()

    except KeyboardInterrupt:
        if planner_client_node:
            planner_client_node.get_logger().info("KeyboardInterrupt received, shutting down.")
    except SystemExit as e: # Catch explicit sys.exit() calls
        if planner_client_node:
             planner_client_node.get_logger().info(f"SystemExit called ({e}), shutting down.")
        else:
             print(f"SystemExit called ({e}) during initialization, shutting down.")
    except Exception as e:
        if planner_client_node:
            tb_str = traceback.format_exc()
            planner_client_node.get_logger().fatal(f"An unhandled exception occurred in main: {e}\n{tb_str}")
        else:
            print(f"An unhandled exception occurred during node initialization: {e}", file=sys.stderr)
            traceback.print_exc() # Print traceback if logger isn't available
    finally:
        # Shutdown sequence
        if planner_client_node:
            planner_client_node.get_logger().info("Shutting down node.")
            # Ensure node is destroyed before shutdown
            planner_client_node.destroy_node()
        # Always attempt shutdown, checking if it's necessary
        if rclpy.ok():
            rclpy.shutdown()
        print("RCLPY shutdown sequence complete.")


if __name__ == '__main__':
    main()