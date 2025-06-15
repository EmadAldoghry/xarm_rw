#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from xarm_msgs.srv import PlanPose, PlanSingleStraight, PlanExec
import sys
import time
import math

# Service names provided by xarm_planner_node (adjust if namespace/prefix is used)
POSE_PLAN_SERVICE = '/xarm_pose_plan'
JOINT_PLAN_SERVICE = '/xarm_joint_plan' # Not used here, but good to know
STRAIGHT_PLAN_SERVICE = '/xarm_straight_plan'
EXEC_PLAN_SERVICE = '/xarm_exec_plan'

class CartesianPlannerClient(Node):

    def __init__(self):
        super().__init__('xarm6_cartesian_planner_client')
        self.get_logger().info("Initializing Cartesian Planner Client...")

        # Create Service Clients
        self.pose_plan_client = self.create_client(PlanPose, POSE_PLAN_SERVICE)
        self.straight_plan_client = self.create_client(PlanSingleStraight, STRAIGHT_PLAN_SERVICE)
        self.exec_plan_client = self.create_client(PlanExec, EXEC_PLAN_SERVICE)

        # Wait for services to be available
        if not self.straight_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {STRAIGHT_PLAN_SERVICE} not available.')
            # Consider raising an exception or shutting down
            rclpy.shutdown()
            sys.exit(1)
        if not self.exec_plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {EXEC_PLAN_SERVICE} not available.')
            rclpy.shutdown()
            sys.exit(1)
            
        self.get_logger().info("Service clients created and ready.")

    def plan_and_execute_straight(self, target_pose, wait_for_execution=True):
        """
        Plans a straight-line Cartesian path to the target pose and executes it.
        """
        self.get_logger().info(f"Planning straight path to: {target_pose}")
        
        # --- 1. Plan the Straight Path ---
        plan_req = PlanSingleStraight.Request()
        plan_req.target = target_pose

        plan_future = self.straight_plan_client.call_async(plan_req)
        
        # Wait for the planning result
        rclpy.spin_until_future_complete(self, plan_future)

        if plan_future.result() is None:
            self.get_logger().error(f'Exception while calling service {STRAIGHT_PLAN_SERVICE}: {plan_future.exception()}')
            return False
            
        plan_res = plan_future.result()
        if not plan_res.success:
            self.get_logger().error("Cartesian planning failed!")
            return False
            
        self.get_logger().info("Cartesian planning successful.")

        # --- 2. Execute the Plan ---
        exec_req = PlanExec.Request()
        exec_req.wait = wait_for_execution # If true, service call blocks until execution finishes

        exec_future = self.exec_plan_client.call_async(exec_req)
        
        # Wait for the execution result
        rclpy.spin_until_future_complete(self, exec_future)

        if exec_future.result() is None:
            self.get_logger().error(f'Exception while calling service {EXEC_PLAN_SERVICE}: {exec_future.exception()}')
            return False
        
        exec_res = exec_future.result()
        if not exec_res.success:
            self.get_logger().error("Plan execution failed!")
            return False

        self.get_logger().info("Plan execution successful.")
        return True

def main(args=None):
    rclpy.init(args=args)
    
    planner_client_node = CartesianPlannerClient()

    try:
        # Define target poses (adjust these values based on your workspace)
        # Remember: These are absolute poses in the planning frame (usually 'link_base')
        
        # Example Pose 1 (adjust coordinates as needed)
        
        # Example Pose 2 (move slightly up and to the right)
        pose2 = Pose()
        pose2.position.x = 0.413969
        pose2.position.y = -0.0558126
        pose2.position.z = -0.0625005
        # Same orientation
        pose2.orientation.x = q_x
        pose2.orientation.y = q_y
        pose2.orientation.z = q_z
        pose2.orientation.w = q_w
        
        # Example Pose 3 (move back to the center, a bit higher)
        pose3 = Pose()
        pose3.position.x = 0.361638
        pose3.position.y = -0.0488051
        pose3.position.z = -0.0625005
        # Same orientation
        pose3.orientation.x = q_x
        pose3.orientation.y = q_y
        pose3.orientation.z = q_z
        pose3.orientation.w = q_w

        # --- Execute sequence ---
        planner_client_node.get_logger().info("Starting Cartesian planning sequence...")
        time.sleep(1.0) # Give MoveIt/Planner time to fully initialize after launch

        if planner_client_node.plan_and_execute_straight(pose1):
            planner_client_node.get_logger().info("Reached Pose 1")
            time.sleep(1.0)
        else:
             planner_client_node.get_logger().error("Failed to reach Pose 1")
             raise RuntimeError("Planning/Execution Failed")

        if planner_client_node.plan_and_execute_straight(pose2):
            planner_client_node.get_logger().info("Reached Pose 2")
            time.sleep(1.0)
        else:
             planner_client_node.get_logger().error("Failed to reach Pose 2")
             raise RuntimeError("Planning/Execution Failed")

        if planner_client_node.plan_and_execute_straight(pose3):
            planner_client_node.get_logger().info("Reached Pose 3")
            time.sleep(1.0)
        else:
             planner_client_node.get_logger().error("Failed to reach Pose 3")
             raise RuntimeError("Planning/Execution Failed")
             
        if planner_client_node.plan_and_execute_straight(pose1): # Go back to pose 1
            planner_client_node.get_logger().info("Returned to Pose 1")
            time.sleep(1.0)
        else:
             planner_client_node.get_logger().error("Failed to return to Pose 1")
             raise RuntimeError("Planning/Execution Failed")

        planner_client_node.get_logger().info("Cartesian planning sequence finished.")

    except Exception as e:
        planner_client_node.get_logger().error(f"An exception occurred during planning/execution: {e}")
    finally:
        # Shutdown
        planner_client_node.get_logger().info("Shutting down node.")
        planner_client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()