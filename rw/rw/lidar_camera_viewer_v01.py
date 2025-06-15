#!/usr/bin/env python3

"""
Created/Modified files during execution:
   lidar_3d_viewer.py
"""

import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import open3d as o3d

import threading
import time


class Open3DViewer:
    """
    Manages an Open3D Visualizer in a separate thread.
    You can call update_points(np_points) to replace
    the current cloud with new data in real time.
    """

    def __init__(self, window_name="LiDAR 3D View"):
        self.window_name = window_name

        # The geometry to visualize: a single PointCloud
        self.cloud = o3d.geometry.PointCloud()

        # For thread-safe updates
        self._lock = threading.Lock()
        self._new_data = False
        self._new_points = o3d.utility.Vector3dVector([])

        # Launch the background thread that runs the Open3D visualizer
        self._vis_thread = threading.Thread(target=self._vis_loop)
        self._vis_thread.daemon = True
        self._vis_thread.start()

    def _vis_loop(self):
        """
        The Open3D visualization loop. Runs in a dedicated thread.
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=self.window_name, width=1280, height=720)
        self.vis.add_geometry(self.cloud)

        while True:
            # If there's new data, lock and update the geometry
            with self._lock:
                if self._new_data:
                    self.cloud.points = self._new_points
                    self.vis.update_geometry(self.cloud)
                    self._new_data = False

            self.vis.poll_events()
            self.vis.update_renderer()

            # Sleep briefly so we don't max out the CPU
            time.sleep(0.01)

    def update_points(self, np_points):
        """
        Replaces the current point cloud with the provided (N x 3) NumPy array.
        """
        with self._lock:
            self._new_points = o3d.utility.Vector3dVector(np_points)
            self._new_data = True


class Lidar3DViewerNode(Node):
    """
    ROS 2 node that subscribes to a LiDAR PointCloud2 topic,
    then passes the point data to our Open3DViewer for 3D visualization.
    """

    def __init__(self):
        super().__init__('lidar_3d_viewer_node')

        # Parameter for LiDAR topic name
        self.declare_parameter('lidar_topic', '/scan/points')
        self.lidar_topic = (
            self.get_parameter('lidar_topic')
                .get_parameter_value()
                .string_value
        )

        # Subscribe to the PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            10
        )

        # Create an Open3DViewer instance to display points in real time
        self.viewer = Open3DViewer("LiDAR 3D View")

        self.get_logger().info(f"Subscribed to LiDAR topic: {self.lidar_topic}")
        self.get_logger().info("Open3D 3D Viewer running in background thread.")

    def lidar_callback(self, msg: PointCloud2):
        """
        Converts incoming PointCloud2 messages into a NumPy array
        and passes them to the Open3DViewer.
        """
        points_list = []

        # Extract (x,y,z) fields, skipping NaN
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            # Check for non-finite or huge values
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue
            if abs(x) > 1e4 or abs(y) > 1e4 or abs(z) > 1e4:
                continue

            points_list.append([x, y, z])

        # Convert list to a NumPy array
        if points_list:
            np_points = np.array(points_list, dtype=np.float64)
        else:
            np_points = np.zeros((0, 3), dtype=np.float64)

        # Pass the points to the Open3D viewer
        self.viewer.update_points(np_points)


def main(args=None):
    rclpy.init(args=args)
    node = Lidar3DViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()