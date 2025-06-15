#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
from transforms3d.euler import euler2quat
import math

class ImuCovarianceFixer(Node):
    def __init__(self):
        super().__init__('imu_covariance_fixer')

        # Create subscription and publisher
        self.subscription = self.create_subscription(
            Imu,
            '/oak/imu/data',
            self.imu_callback,
            10
        )

        self.publisher = self.create_publisher(
            Imu,
            '/oak/imu/data_fixed',
            10
        )

        # Covariance matrices
        self.orientation_cov = np.diag([0.01, 0.01, 0.01]).flatten()
        self.angular_vel_cov = np.diag([0.01, 0.01, 0.01]).flatten()
        self.linear_acc_cov = np.diag([0.01, 0.01, 0.01]).flatten()

        # Orientation tracking variables
        self.last_time = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Bias estimation variables
        self.accel_bias = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.gyro_bias = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.bias_samples = 200  # Number of samples for initial calibration
        self.samples_collected = 0
        self.is_calibrating = True

        # Buffers for initial samples
        self.accel_buffer = {'x': [], 'y': [], 'z': []}
        self.gyro_buffer = {'x': [], 'y': [], 'z': []}

        # Complementary filter parameters
        self.alpha = 0.98  # Weight for gyroscope data (0.98 = 98% gyro, 2% accel)

        # Logging
        self.get_logger().info('IMU Covariance Fixer node initialized')
        self.get_logger().info(f'Collecting {self.bias_samples} samples for calibration...')

    def calibrate_bias(self, msg):
        """Calibrate IMU biases using initial samples."""
        if self.samples_collected < self.bias_samples:
            # Collect acceleration samples
            self.accel_buffer['x'].append(msg.linear_acceleration.x)
            self.accel_buffer['y'].append(msg.linear_acceleration.y)
            self.accel_buffer['z'].append(msg.linear_acceleration.z)

            # Collect gyroscope samples
            self.gyro_buffer['x'].append(msg.angular_velocity.x)
            self.gyro_buffer['y'].append(msg.angular_velocity.y)
            self.gyro_buffer['z'].append(msg.angular_velocity.z)

            self.samples_collected += 1
            return True

        elif self.is_calibrating:
            # Compute acceleration bias
            self.accel_bias['x'] = sum(self.accel_buffer['x']) / self.bias_samples
            self.accel_bias['y'] = sum(self.accel_buffer['y']) / self.bias_samples
            self.accel_bias['z'] = sum(self.accel_buffer['z']) / self.bias_samples - 9.81  # Remove gravity

            # Compute gyroscope bias
            self.gyro_bias['x'] = sum(self.gyro_buffer['x']) / self.bias_samples
            self.gyro_bias['y'] = sum(self.gyro_buffer['y']) / self.bias_samples
            self.gyro_bias['z'] = sum(self.gyro_buffer['z']) / self.bias_samples

            self.is_calibrating = False
            self.get_logger().info('IMU calibration completed')
            self.get_logger().info(f'Accel bias: x={self.accel_bias["x"]:.3f}, y={self.accel_bias["y"]:.3f}, z={self.accel_bias["z"]:.3f}')
            self.get_logger().info(f'Gyro bias: x={self.gyro_bias["x"]:.3f}, y={self.gyro_bias["y"]:.3f}, z={self.gyro_bias["z"]:.3f}')

        return False

    def compute_orientation_from_accel(self, ax, ay, az):
        """Compute roll and pitch from accelerometer data."""
        roll = math.atan2(ay, math.sqrt(ax*ax + az*az))
        pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
        return roll, pitch

    def imu_callback(self, msg):
        # Check if still calibrating
        if self.calibrate_bias(msg):
            return

        # Create new IMU message
        fixed_msg = Imu()
        fixed_msg.header = msg.header

        # Remove bias from acceleration
        ax = msg.linear_acceleration.x - self.accel_bias['x']
        ay = msg.linear_acceleration.y - self.accel_bias['y']
        az = msg.linear_acceleration.z - self.accel_bias['z']

        # Remove bias from angular velocity
        wx = msg.angular_velocity.x - self.gyro_bias['x']
        wy = msg.angular_velocity.y - self.gyro_bias['y']
        wz = msg.angular_velocity.z - self.gyro_bias['z']

        # Compute roll and pitch from accelerometer
        accel_roll, accel_pitch = self.compute_orientation_from_accel(ax, ay, az)

        # Update yaw using gyro integration
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is not None:
            dt = current_time - self.last_time
            self.yaw += wz * dt
            # Complementary filter: combine gyro and accel data
            self.roll = self.alpha * (self.roll + wx * dt) + (1 - self.alpha) * accel_roll
            self.pitch = self.alpha * (self.pitch + wy * dt) + (1 - self.alpha) * accel_pitch
        self.last_time = current_time

        # Convert to quaternion
        quat = euler2quat(self.roll, self.pitch, self.yaw, 'sxyz')

        # Set orientation
        fixed_msg.orientation.x = quat[1]
        fixed_msg.orientation.y = quat[2]
        fixed_msg.orientation.z = quat[3]
        fixed_msg.orientation.w = quat[0]

        # Set angular velocity
        fixed_msg.angular_velocity.x = wx
        fixed_msg.angular_velocity.y = wy
        fixed_msg.angular_velocity.z = wz

        # Set linear acceleration
        fixed_msg.linear_acceleration.x = ax
        fixed_msg.linear_acceleration.y = ay
        fixed_msg.linear_acceleration.z = az

        # Set covariances
        fixed_msg.orientation_covariance = self.orientation_cov.tolist()
        fixed_msg.angular_velocity_covariance = self.angular_vel_cov.tolist()
        fixed_msg.linear_acceleration_covariance = self.linear_acc_cov.tolist()

        # Publish the fixed message
        self.publisher.publish(fixed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuCovarianceFixer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()