#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from xarm_msgs.srv import PlanPose, PlanSingleStraight, PlanExec
import sys
import time
import math
import numpy as np
# from threading import Lock # Not strictly needed for this sequential logic

# Service names
# POSE_PLAN_SERVICE = '/xarm_pose_plan' # Not used in this version
# JOINT_PLAN_SERVICE = '/xarm_joint_plan' # Not used in this version
STRAIGHT_PLAN_SERVICE = '/xarm_straight_plan'
EXEC_PLAN_SERVICE = '/xarm_exec_plan'

# Input PointCloud Topic
PROJECTED_POINTS_TOPIC = '/projected_non_ground_points'

# --- Realistic Service Timeouts ---
# Allow sufficient time for planning and execution
PLANNING_TIMEOUT = 2.0 # seconds (Increased slightly for potentially complex plans)
EXECUTION_TIMEOUT = 10.0 # seconds (Increased significantly to allow for slower moves)

class CartesianPlannerClient(Node):
    """
    ROS 2 Node to control an xArm robot based on incoming PointCloud2 data.

    Subscribes to projected points, calculates target flange poses, and commands
    the robot to move sequentially through them: first to the farthest point,
    then iteratively to the nearest remaining point. It processes one point cloud
    at a time, ignoring new messages while busy. Includes an initial homing move.
    """

    def __init__(self):
        """Initializes the node, service clients, subscriber, and state variables."""
        super().__init__('xarm6_sequence_cartesian_planner_client')
        self.get_logger().info("Initializing Sequence Cartesian Planner Client...")

        # --- Tool Offset Configuration ---
        # This is the vector FROM the tool tip TO the flange mount point,
        # expressed in the BASE frame when the tool is pointing straight down (-Z base).
        # Since we want flange_pose = tip_pose - offset_vector, the offset vector
        # needs careful consideration based on the tool's default orientation.
        # Assuming the provided offset is correct for the desired fixed orientation:
        self.tip_to_flange_offset_base = np.array([
             0.038302994,  # X offset in base frame
             0.0,          # Y offset in base frame
            -0.090695661  # Z offset in base frame (Tip is below flange)
        ], dtype=np.float64)
        self.get_logger().info(f"Using Tool Tip Offset (subtracted from target tip pose): {self.tip_to_flange_offset_base}")

        # --- Fixed Orientation for the Tool Tip ---
        # Quaternion for pointing straight down (e.g., along negative Z of base)
        # Corresponds to Roll=180deg (about X), Pitch=0, Yaw=0
        self.target_orientation = Pose().orientation
        self.target_orientation.x = 1.0
        self.target_orientation.y = 0.0
        self.target_orientation.z = 0.0
        self.target_orientation.w = 0.0
        self.get_logger().info(f"Using Fixed Target Orientation: O({self.target_orientation.x:.1f}, {self.target_orientation.y:.1f}, {self.target_orientation.z:.1f}, {self.target_orientation.w:.1f})")

        # --- Service Clients ---
        self.straight_plan_client = self.create_client(PlanSingleStraight, STRAIGHT_PLAN_SERVICE)
        self.exec_plan_client = self.create_client(PlanExec, EXEC_PLAN_SERVICE)

        # Wait for services
        if not self.straight_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {STRAIGHT_PLAN_SERVICE} not available.')
            rclpy.shutdown(); sys.exit(1)
        if not self.exec_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {EXEC_PLAN_SERVICE} not available.')
            rclpy.shutdown(); sys.exit(1)
        self.get_logger().info("Service clients created and ready.")

        # --- State Variables ---
        self.is_busy = False # Flag to indicate if processing a sequence
        self.current_flange_position = None # Stores the last known flange position [x, y, z] as numpy array
        # self.state_lock = Lock() # Mutex lock, not strictly needed if callback runs sequentially in one thread

        # --- Subscriber ---
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            PROJECTED_POINTS_TOPIC,
            self.projected_points_callback,
            qos_profile_sensor_data # Reliable, keep last history depth 1
        )
        self.get_logger().info(f"Subscribed to {PROJECTED_POINTS_TOPIC}")
        self.get_logger().info(f"Using Planning Timeout: {PLANNING_TIMEOUT}s, Execution Timeout: {EXECUTION_TIMEOUT}s")


    def _execute_move(self, target_flange_pose: Pose):
        """
        Plans and executes a single straight-line move to the target flange pose.

        Uses standard spin_until_future_complete with timeouts.
        Updates self.current_flange_position ONLY on successful execution.

        Args:
            target_flange_pose: The desired geometry_msgs.msg.Pose for the robot flange.

        Returns:
            True if both planning and execution were successful, False otherwise.
        """
        self.get_logger().info(f"Attempting move to flange pose: P({target_flange_pose.position.x:.3f}, {target_flange_pose.position.y:.3f}, {target_flange_pose.position.z:.3f}) O({target_flange_pose.orientation.x:.1f}, {target_flange_pose.orientation.y:.1f}, {target_flange_pose.orientation.z:.1f}, {target_flange_pose.orientation.w:.1f})")

        # --- 1. Plan the Straight Move ---
        plan_req = PlanSingleStraight.Request()
        plan_req.target = target_flange_pose
        plan_future = self.straight_plan_client.call_async(plan_req)
        self.get_logger().debug("Calling Straight Plan service...")

        try:
            # Wait for the planning result with timeout
            rclpy.spin_until_future_complete(self, plan_future, timeout_sec=PLANNING_TIMEOUT)

            if not plan_future.done():
                self.get_logger().error(f"Planning service call timed out after {PLANNING_TIMEOUT}s.")
                plan_future.cancel() # Attempt to cancel the future
                return False

            plan_res = plan_future.result()
            if plan_res is None: # Check if result is None (can happen on exceptions during call)
                 self.get_logger().error(f"Planning service call failed to return result. Exception: {plan_future.exception()}")
                 return False

            if not plan_res.success:
                self.get_logger().error(f"Cartesian planning failed! Service message: '{plan_res.message}'")
                return False

            self.get_logger().info("Planning successful.")

        except Exception as e:
             self.get_logger().error(f"Exception during planning service call: {e}", exc_info=True)
             # Ensure future is cancelled if exception happens mid-wait
             if plan_future and not plan_future.done():
                 plan_future.cancel()
             return False

        # --- 2. Execute the Planned Move ---
        exec_req = PlanExec.Request()
        exec_req.wait = True # IMPORTANT: Set to True to block until motion finishes
        exec_future = self.exec_plan_client.call_async(exec_req)
        self.get_logger().debug("Calling Execution service (wait=True)...")

        try:
            # Wait for the execution result with timeout
            rclpy.spin_until_future_complete(self, exec_future, timeout_sec=EXECUTION_TIMEOUT)

            if not exec_future.done():
                self.get_logger().error(f"Execution service call timed out after {EXECUTION_TIMEOUT}s.")
                exec_future.cancel()
                # CRITICAL: If execution times out, the arm might still be moving or in an unknown state.
                # Invalidate the current position knowledge.
                self.current_flange_position = None
                self.get_logger().error("Current flange position is now UNKNOWN due to execution timeout!")
                # Consider adding recovery logic here (e.g., attempt to stop arm, go home)
                return False

            exec_res = exec_future.result()
            if exec_res is None:
                 self.get_logger().error(f"Execution service call failed to return result. Exception: {exec_future.exception()}")
                 # Position is likely unknown if the service call itself failed
                 self.current_flange_position = None
                 return False

            if not exec_res.success:
                self.get_logger().error(f"Plan execution failed! Service message: '{exec_res.message}'")
                # Position might be inaccurate if execution failed mid-move or failed to start.
                self.current_flange_position = None
                self.get_logger().error("Current flange position is now UNKNOWN due to execution failure!")
                return False

            # --- Success ---
            self.get_logger().info("Execution successful.")
            # Update current position only after confirmed successful execution
            self.current_flange_position = np.array([
                target_flange_pose.position.x,
                target_flange_pose.position.y,
                target_flange_pose.position.z
            ], dtype=np.float64)
            self.get_logger().debug(f"Updated current flange position: {self.current_flange_position}")
            return True

        except Exception as e:
             self.get_logger().error(f"Exception during execution service call: {e}", exc_info=True)
             if exec_future and not exec_future.done():
                 exec_future.cancel()
             # Position is unknown after an unexpected exception during execution call
             self.current_flange_position = None
             self.get_logger().error("Current flange position is now UNKNOWN due to exception during execution!")
             return False


    def projected_points_callback(self, msg: PointCloud2):
        """
        Callback function for projected point cloud messages.

        Processes the points sequentially: moves to the farthest point first,
        then iteratively moves to the nearest remaining point. Ignores messages
        if already processing a sequence or if the robot's position is unknown.
        """
        # --- Check State ---
        if self.is_busy:
            self.get_logger().warn("Arm is busy processing previous sequence, skipping new point cloud.", throttle_duration_sec=5.0)
            return

        if self.current_flange_position is None:
            self.get_logger().error("Current flange position is unknown. Cannot start sequence. Please restart or re-home. Skipping message.")
            # Set is_busy to False just in case, although it should be false if position is None unless error occurred mid-sequence
            self.is_busy = False
            return

        self.get_logger().info(f"Received new point cloud message (Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}). Processing sequence...")
        self.is_busy = True # Set busy flag

        try:
            # --- Read Points from PointCloud2 ---
            # Use sensor_msgs_py for efficient reading
            points_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            # Convert generator to a list of lists
            tip_positions_list = [[x, y, z] for x, y, z in points_generator]

            if not tip_positions_list:
                 self.get_logger().warn("Received empty or invalid projected point cloud. Sequence finished (no points).")
                 # Nothing to do, sequence is finished for this empty cloud
                 self.is_busy = False # Reset busy flag before returning
                 return

            # Convert list of lists to a NumPy array for vectorized operations
            tip_positions = np.array(tip_positions_list, dtype=np.float64) # Use float64 for precision
            num_points = tip_positions.shape[0]
            self.get_logger().info(f"Successfully read {num_points} projected points.")

            # --- Calculate all Target Flange Positions ---
            # Target flange position = Target tip position - Offset vector
            target_flange_positions = tip_positions - self.tip_to_flange_offset_base
            self.get_logger().debug(f"Calculated {num_points} target flange positions.")

            # --- Initialize Sequence ---
            # Make a copy of the current known flange position
            # This 'current_pos' variable will track the robot's *intended* position *during* the sequence planning
            current_planning_pos = np.copy(self.current_flange_position)
            unvisited_indices = set(range(num_points)) # Set of indices of points yet to be visited

            # --- 1. Find and Move to Farthest Point First ---
            self.get_logger().info("Finding farthest point from current position...")
            distances = np.linalg.norm(target_flange_positions - current_planning_pos, axis=1)

            if len(distances) == 0: # Should not happen if num_points > 0, but safety check
                self.get_logger().error("No valid distances calculated to target points.")
                # Abort sequence for this cloud
                self.is_busy = False # Reset busy flag
                return

            farthest_idx = np.argmax(distances)
            farthest_flange_pos = target_flange_positions[farthest_idx]

            # Construct the target pose message
            target_pose = Pose()
            target_pose.position.x = float(farthest_flange_pos[0])
            target_pose.position.y = float(farthest_flange_pos[1])
            target_pose.position.z = float(farthest_flange_pos[2])
            target_pose.orientation = self.target_orientation # Use the pre-defined fixed orientation

            self.get_logger().info(f"Moving to farthest point #{farthest_idx} at {farthest_flange_pos} first.")
            success = self._execute_move(target_pose) # This function handles planning, execution, and updates self.current_flange_position on success

            if not success:
                self.get_logger().error(f"Failed to move to farthest point #{farthest_idx}. Aborting sequence for this point cloud.")
                # self.current_flange_position might be None now if _execute_move failed badly.
                # The finally block will reset self.is_busy.
                return # Exit the callback

            # If move was successful, update planning position and mark as visited
            current_planning_pos = np.copy(self.current_flange_position) # Re-sync planning pos with actual state
            unvisited_indices.remove(farthest_idx)
            self.get_logger().debug(f"Successfully moved to point #{farthest_idx}. Remaining points: {len(unvisited_indices)}")


            # --- 2. Iterate through Nearest Remaining Points ---
            self.get_logger().info("Starting nearest neighbor sequence for remaining points...")
            point_counter = 1 # Start counter after the first (farthest) point
            while unvisited_indices:
                point_counter += 1
                # Safety check: ensure position is still known (could become None if a move failed)
                if self.current_flange_position is None:
                    self.get_logger().error("Lost known robot position during sequence. Aborting remaining moves.")
                    break # Exit the while loop

                # Update the reference position for distance calculation to the actual last known position
                current_planning_pos = np.copy(self.current_flange_position)
                self.get_logger().debug(f"Finding nearest point from: {current_planning_pos}. Unvisited count: {len(unvisited_indices)}")

                # Get the indices and positions of only the unvisited points
                unvisited_list = list(unvisited_indices)
                if not unvisited_list: break # Should be caught by while condition, but safety first

                candidate_flange_positions = target_flange_positions[unvisited_list]

                # Calculate distances from the current position to all *remaining* points
                distances_to_unvisited = np.linalg.norm(candidate_flange_positions - current_planning_pos, axis=1)

                if len(distances_to_unvisited) == 0:
                    self.get_logger().warn("No distances calculated to remaining unvisited points. Breaking loop.")
                    break

                # Find the index of the nearest point *within the candidate list*
                nearest_local_idx = np.argmin(distances_to_unvisited)
                # Map this local index back to the *original* index from the full point cloud
                nearest_original_idx = unvisited_list[nearest_local_idx]

                # Get the position of the nearest point
                nearest_flange_pos = target_flange_positions[nearest_original_idx]

                # Construct the target pose message
                target_pose = Pose()
                target_pose.position.x = float(nearest_flange_pos[0])
                target_pose.position.y = float(nearest_flange_pos[1])
                target_pose.position.z = float(nearest_flange_pos[2])
                target_pose.orientation = self.target_orientation # Use the pre-defined fixed orientation

                self.get_logger().info(f"Moving to nearest remaining point #{nearest_original_idx} (Move {point_counter}/{num_points}) at {nearest_flange_pos}.")
                success = self._execute_move(target_pose) # Execute the move

                if not success:
                    self.get_logger().error(f"Failed to move to point #{nearest_original_idx}. Aborting sequence for this point cloud.")
                    # self.current_flange_position might be None now.
                    # Exit the while loop, the finally block handles is_busy.
                    break

                # If successful, mark this point as visited
                # current_planning_pos is updated implicitly because self.current_flange_position was updated inside _execute_move
                unvisited_indices.remove(nearest_original_idx)
                # **** TYPO CORRECTED HERE ****
                self.get_logger().debug(f"Successfully moved to point #{nearest_original_idx}. Remaining points: {len(unvisited_indices)}")

            # End of while loop
            if not unvisited_indices:
                 self.get_logger().info("Successfully visited all points in the sequence.")
            else:
                 self.get_logger().warn("Sequence terminated before visiting all points (likely due to a move failure).")

        except Exception as e:
            self.get_logger().error(f"Unexpected error during point cloud sequence processing: {e}", exc_info=True)
            # Position might be compromised depending on where the error occurred
            # Consider setting self.current_flange_position = None here if the error source is unclear

        finally:
            # CRITICAL: Ensure the busy flag is cleared regardless of success or failure
            self.is_busy = False
            self.get_logger().info("Finished processing point cloud. Ready for next message.")


def main(args=None):
    """Main function to initialize ROS, create the node, perform initial homing, and spin."""
    rclpy.init(args=args)
    planner_client_node = None # Initialize to None for robust cleanup

    try:
        planner_client_node = CartesianPlannerClient()

        # --- IMPORTANT: Initial Homing/Positioning ---
        # Move to a known starting pose before accepting point clouds
        planner_client_node.get_logger().info("Attempting initial move to a defined ready pose...")
        # Define your desired starting pose (adjust coordinates as needed)
        ready_pose = Pose()
        ready_pose.position.x = 0.300 # Example X
        ready_pose.position.y = 0.0   # Example Y
        ready_pose.position.z = 0.350 # Example Z (safe height)
        ready_pose.orientation = planner_client_node.target_orientation # Use the same fixed orientation

        # Prevent callback from triggering during this initial move
        planner_client_node.is_busy = True
        time.sleep(1.0) # Small delay to ensure services are fully ready after node creation

        # Use the internal move function which handles planning, execution, and updates position state
        initial_move_success = planner_client_node._execute_move(ready_pose)

        if not initial_move_success:
             # _execute_move already logs errors and sets position to None if execution fails
             planner_client_node.get_logger().fatal(
                 "Failed to move to initial ready pose! Robot position is unknown. "
                 "Cannot safely proceed. Shutting down."
             )
             # No need to set position to None here, _execute_move should have done it
             planner_client_node.destroy_node()
             rclpy.shutdown()
             sys.exit(1) # Exit with error code
        else:
             # Position is now known and set by _execute_move
             planner_client_node.get_logger().info(
                 f"Successfully moved to ready pose. Current flange position confirmed: {planner_client_node.current_flange_position}"
             )
             planner_client_node.is_busy = False # Allow callbacks now that arm is ready

    except Exception as e:
        if planner_client_node:
            planner_client_node.get_logger().fatal(f"Exception during initialization or initial homing: {e}", exc_info=True)
        else:
            print(f"Exception during node creation: {e}", file=sys.stderr) # Logger might not be ready
        if planner_client_node:
             planner_client_node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()
        sys.exit(1) # Exit with error code

    # --- Start Processing Callbacks ---
    try:
        planner_client_node.get_logger().info("Initialization complete. Cartesian planner client spinning, waiting for projected points...")
        rclpy.spin(planner_client_node)

    except KeyboardInterrupt:
        planner_client_node.get_logger().info("KeyboardInterrupt received, shutting down.")
    except Exception as e:
        # Log any exceptions that occur during spin
        planner_client_node.get_logger().error(f"An unexpected exception occurred during spin: {e}", exc_info=True)
    finally:
        # Cleanup resources
        planner_client_node.get_logger().info("Shutting down node.")
        if planner_client_node:
             planner_client_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        planner_client_node.get_logger().info("ROS shutdown complete.")

if __name__ == '__main__':
    main()