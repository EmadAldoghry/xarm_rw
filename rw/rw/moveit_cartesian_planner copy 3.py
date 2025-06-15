#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray # Import PoseArray
from xarm_msgs.srv import PlanPose, PlanSingleStraight, PlanExec
import sys
import time
import math
import threading # Import threading for locking

# Service names provided by xarm_planner_node (adjust if namespace/prefix is used)
POSE_PLAN_SERVICE = '/xarm_pose_plan'
JOINT_PLAN_SERVICE = '/xarm_joint_plan' # Not used here, but good to know
STRAIGHT_PLAN_SERVICE = '/xarm_straight_plan'
EXEC_PLAN_SERVICE = '/xarm_exec_plan'

# Topic name for receiving pose arrays
POSE_ARRAY_TOPIC = '/arm_path_poses'

class CartesianPlannerClient(Node):

    def __init__(self):
        super().__init__('xarm6_cartesian_planner_client')
        self.get_logger().info("Initializing Cartesian Planner Client...")

        # --- Service Clients ---
        self.pose_plan_client = self.create_client(PlanPose, POSE_PLAN_SERVICE) # Keep for potential future use
        self.straight_plan_client = self.create_client(PlanSingleStraight, STRAIGHT_PLAN_SERVICE)
        self.exec_plan_client = self.create_client(PlanExec, EXEC_PLAN_SERVICE)

        # Wait for essential services
        if not self.straight_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {STRAIGHT_PLAN_SERVICE} not available.')
            rclpy.shutdown()
            sys.exit(1)
        if not self.exec_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {EXEC_PLAN_SERVICE} not available.')
            rclpy.shutdown()
            sys.exit(1)
        self.get_logger().info("Service clients created and ready.")

        # --- Subscriber ---
        # Create subscriber for the PoseArray topic
        self.pose_array_sub = self.create_subscription(
            PoseArray,
            POSE_ARRAY_TOPIC,
            self.pose_array_callback,
            10) # QoS profile depth 10
        self.get_logger().info(f"Subscribed to {POSE_ARRAY_TOPIC}")

        # --- State Management ---
        # Use a lock to prevent processing multiple pose arrays concurrently
        self._is_executing = False
        self._lock = threading.Lock()


    def pose_array_callback(self, msg: PoseArray):
        """
        Callback function triggered when a PoseArray message is received.
        Plans and executes a straight-line path for each pose in the array sequentially.
        """
        with self._lock: # Acquire lock
            if self._is_executing:
                self.get_logger().warn("Already executing a path sequence. Ignoring new PoseArray message.")
                return # Exit if already busy

            if not msg.poses:
                self.get_logger().warn("Received empty PoseArray. No action taken.")
                return

            self.get_logger().info(f"Received PoseArray with {len(msg.poses)} poses. Starting execution sequence...")
            self._is_executing = True # Set flag indicating busy

        # --- Execute sequence based on received poses ---
        # We release the lock here so the node can potentially still process other callbacks
        # if needed, but the _is_executing flag prevents re-entry into this specific logic.
        try:
            total_poses = len(msg.poses)
            for i, target_pose in enumerate(msg.poses):
                self.get_logger().info(f"--- Processing Pose {i + 1}/{total_poses} ---")

                # Validate the pose (basic check for NaN/Inf if necessary, though usually handled by sender)
                # Example basic validation (can be expanded)
                if not all(math.isfinite(getattr(target_pose.position, p)) for p in ['x', 'y', 'z']) or \
                   not all(math.isfinite(getattr(target_pose.orientation, o)) for o in ['x', 'y', 'z', 'w']):
                    self.get_logger().error(f"Pose {i+1} contains invalid (NaN/Inf) values. Skipping pose.")
                    continue # Skip this invalid pose

                # Plan and Execute for the current pose
                success = self.plan_and_execute_straight(target_pose, wait_for_execution=True)

                if success:
                    self.get_logger().info(f"Successfully reached Pose {i + 1}")
                    # Optional short pause between points
                    time.sleep(0.5)
                else:
                    self.get_logger().error(f"Failed to plan or execute path to Pose {i + 1}. Aborting sequence for this PoseArray.")
                    # Stop processing the rest of the poses in this array on failure
                    break # Exit the loop

            self.get_logger().info("Finished processing PoseArray sequence.")

        except Exception as e:
             self.get_logger().error(f"An exception occurred during PoseArray execution: {e}")
        finally:
            # Ensure the lock is released and flag is reset even if errors occur
            with self._lock:
                self._is_executing = False # Reset flag

    def plan_and_execute_straight(self, target_pose: Pose, wait_for_execution=True):
        """
        Plans a straight-line Cartesian path to the target pose and executes it.
        (Function remains largely the same as before)
        """
        # Validate service clients are ready (optional robustness check)
        if not self.straight_plan_client.service_is_ready() or not self.exec_plan_client.service_is_ready():
             self.get_logger().error("Planning/Execution services not ready. Cannot proceed.")
             return False

        self.get_logger().info(f"Planning straight path to: Position(x={target_pose.position.x:.4f}, y={target_pose.position.y:.4f}, z={target_pose.position.z:.4f}), Orientation(x={target_pose.orientation.x:.4f}, y={target_pose.orientation.y:.4f}, z={target_pose.orientation.z:.4f}, w={target_pose.orientation.w:.4f})")

        # --- 1. Plan the Straight Path ---
        plan_req = PlanSingleStraight.Request()
        plan_req.target = target_pose

        plan_future = self.straight_plan_client.call_async(plan_req)
        rclpy.spin_until_future_complete(self, plan_future, timeout_sec=10.0) # Add timeout

        if plan_future.result() is None:
            if plan_future.cancelled():
                self.get_logger().error(f'Planning service call ({STRAIGHT_PLAN_SERVICE}) was cancelled.')
            elif plan_future.exception() is not None:
                self.get_logger().error(f'Exception during planning service call ({STRAIGHT_PLAN_SERVICE}): {plan_future.exception()}')
            else: # Timeout occurred
                self.get_logger().error(f'Planning service call ({STRAIGHT_PLAN_SERVICE}) timed out.')
            return False

        plan_res = plan_future.result()
        if not plan_res.success:
            self.get_logger().error(f"Cartesian planning failed! Planner returned success=false. (Code: {plan_res.code if hasattr(plan_res, 'code') else 'N/A'})") # Check if xarm_msgs provides error codes
            return False

        self.get_logger().info("Cartesian planning successful.")

        # --- 2. Execute the Plan ---
        exec_req = PlanExec.Request()
        exec_req.wait = wait_for_execution # If true, service call blocks until execution finishes

        exec_future = self.exec_plan_client.call_async(exec_req)
        rclpy.spin_until_future_complete(self, exec_future, timeout_sec=30.0 if wait_for_execution else 5.0) # Longer timeout if waiting

        if exec_future.result() is None:
            if exec_future.cancelled():
                 self.get_logger().error(f'Execution service call ({EXEC_PLAN_SERVICE}) was cancelled.')
            elif exec_future.exception() is not None:
                self.get_logger().error(f'Exception during execution service call ({EXEC_PLAN_SERVICE}): {exec_future.exception()}')
            else: # Timeout occurred
                self.get_logger().error(f'Execution service call ({EXEC_PLAN_SERVICE}) timed out.')
            # Even on timeout, the arm might still be moving if wait=False.
            # If wait=True, timeout likely means execution failed or got stuck.
            return False

        exec_res = exec_future.result()
        if not exec_res.success:
            self.get_logger().error(f"Plan execution failed! Executor returned success=false. (Code: {exec_res.code if hasattr(exec_res, 'code') else 'N/A'})")
            return False

        self.get_logger().info("Plan execution successful.")
        return True

def main(args=None):
    rclpy.init(args=args)

    planner_client_node = CartesianPlannerClient()

    try:
        # Node setup is done in __init__. Now we just spin to keep it alive.
        planner_client_node.get_logger().info("Cartesian planner client node is running and waiting for PoseArray messages...")
        rclpy.spin(planner_client_node) # Keep node alive to receive messages

    except KeyboardInterrupt:
        planner_client_node.get_logger().info("Keyboard interrupt received.")
    except Exception as e:
        planner_client_node.get_logger().error(f"An unexpected error occurred in main: {e}")
    finally:
        # Cleanup
        planner_client_node.get_logger().info("Shutting down node.")
        planner_client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()