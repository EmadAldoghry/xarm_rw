#!/usr/bin/env python3
"""
Combined launch file for RWBot with xarm

This file merges:
  - The xarm MoveIt configuration (from your first launcher), which builds the
    MoveIt configuration (using MoveItConfigsBuilder) and includes the
    xarm_moveit_config launch.
  - The RWBot simulation and supporting nodes (from your second launcher) that
    launch Gazebo (with ros_gz_sim), spawn the robot, run topic bridges, state
    publisher, EKF, and optionally RViz.

Ensure that the necessary packages (rw, xarm_moveit_config, ros_gz_sim, etc.)
are available in your environment.
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# Import MoveIt configuration builder and control utility from your libraries
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file


def setup_moveit(context, *args, **kwargs):
    # Retrieve MoveIt-related launch configurations from the context
    dof = LaunchConfiguration('dof').perform(context)
    robot_type = LaunchConfiguration('robot_type').perform(context)
    prefix = LaunchConfiguration('prefix').perform(context)
    hw_ns = LaunchConfiguration('hw_ns').perform(context)
    limited = LaunchConfiguration('limited').perform(context)
    attach_to = LaunchConfiguration('attach_to').perform(context)
    attach_xyz = LaunchConfiguration('attach_xyz').perform(context)
    attach_rpy = LaunchConfiguration('attach_rpy').perform(context)
    add_gripper = LaunchConfiguration('add_gripper').perform(context)
    add_vacuum_gripper = LaunchConfiguration('add_vacuum_gripper').perform(context)
    add_bio_gripper = LaunchConfiguration('add_bio_gripper').perform(context)
    ros_namespace = LaunchConfiguration('ros_namespace').perform(context)

    # Get the package share directory for 'rw'
    pkg_path = get_package_share_directory('rw')

    # Define file paths (make sure these files exist in your package)
    urdf_file = os.path.join(pkg_path, 'model', 'rwbot_with_xarm.urdf.xacro')
    srdf_file = os.path.join(pkg_path, 'srdf', 'rwbot_with_xarm.srdf.xacro')
    controllers_file = os.path.join(pkg_path, 'config', 'controllers.yaml')
    joint_limits_file = os.path.join(pkg_path, 'config', 'joint_limits.yaml')
    kinematics_file = os.path.join(pkg_path, 'config', 'kinematics.yaml')
    pipeline_filedir = os.path.join(pkg_path, 'config')

    # Setup ros2_control parameters for simulation
    ros2_control_plugin = 'gz_ros2_control/GazeboSimSystem'
    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(pkg_path, 'config', 'ros2_controllers.yaml'),
        prefix=prefix,
        add_gripper=(add_gripper.lower() == 'true'),
        add_bio_gripper=(add_bio_gripper.lower() == 'true'),
        ros_namespace=ros_namespace,
        update_rate=1000,
        use_sim_time=True,
        robot_type=robot_type
    )

    # Build the MoveIt configuration using the builder
    moveit_config = (
        MoveItConfigsBuilder(
            context=context,
            dof=dof,
            robot_type=robot_type,
            prefix=prefix,
            hw_ns=hw_ns,
            limited=limited,
            attach_to=attach_to,
            attach_xyz=attach_xyz,
            attach_rpy=attach_rpy,
            ros2_control_plugin=ros2_control_plugin,
            ros2_control_params=ros2_control_params,
            add_gripper=add_gripper,
            add_vacuum_gripper=add_vacuum_gripper,
            add_bio_gripper=add_bio_gripper,
        )
        .robot_description(file_path=urdf_file)
        .robot_description_semantic(file_path=srdf_file)
        .robot_description_kinematics(file_path=kinematics_file)
        .joint_limits(file_path=joint_limits_file)
        .trajectory_execution(file_path=controllers_file)
        .planning_pipelines(config_folder=pipeline_filedir)
        .to_moveit_configs()
    )
    moveit_config_dump = yaml.dump(moveit_config.to_dict())

    # Include the xarm MoveIt common launch (from the xarm_moveit_config package)
    xarm_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_common2.launch.py'])
        ),
        launch_arguments={
            'prefix': prefix,
            'attach_to': attach_to,
            'attach_xyz': attach_xyz,
            'attach_rpy': attach_rpy,
            'show_rviz': 'false',
            'use_sim_time': 'true',
            'moveit_config_dump': moveit_config_dump,
            'rviz_config': PathJoinSubstitution([FindPackageShare('rw'), 'rviz', 'moveit.rviz'])
        }.items(),
    )

    return [xarm_moveit_common_launch]


def generate_launch_description():
    ld = LaunchDescription()

    # ----- Launch arguments for RWBot simulation (from the second launcher) -----
    ld.add_action(DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz'))
    ld.add_action(DeclareLaunchArgument('rviz_config', default_value='moveit.rviz', description='RViz config file'))
    ld.add_action(DeclareLaunchArgument('world', default_value='cracked_road.sdf', description='Gazebo world file to load'))
    ld.add_action(DeclareLaunchArgument('model', default_value='rwbot_with_xarm.urdf.xacro', description='URDF/Xacro file for the robot'))
    ld.add_action(DeclareLaunchArgument('x', default_value='-13', description='x coordinate of spawned robot'))
    ld.add_action(DeclareLaunchArgument('y', default_value='0.0', description='y coordinate of spawned robot'))
    ld.add_action(DeclareLaunchArgument('z', default_value='0.65', description='z coordinate of spawned robot'))
    ld.add_action(DeclareLaunchArgument('yaw', default_value='0', description='yaw angle of spawned robot'))
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='True', description='Flag to enable use_sim_time'))
    ld.add_action(DeclareLaunchArgument('ros_namespace', default_value='', description='ROS namespace'))

    # ----- Launch arguments for xarm MoveIt configuration (from the first launcher) -----
    ld.add_action(DeclareLaunchArgument('dof', default_value='6', description='Degrees of freedom'))
    ld.add_action(DeclareLaunchArgument('robot_type', default_value='xarm', description='Robot type'))
    ld.add_action(DeclareLaunchArgument('prefix', default_value='', description='Robot prefix'))
    ld.add_action(DeclareLaunchArgument('hw_ns', default_value='xarm', description='Hardware namespace'))
    ld.add_action(DeclareLaunchArgument('limited', default_value='true', description='Limited configuration flag'))
    ld.add_action(DeclareLaunchArgument('attach_to', default_value='base_link', description='Attachment frame'))
    ld.add_action(DeclareLaunchArgument('attach_xyz', default_value='"0 0 0.0401"', description='Attachment xyz offset'))
    ld.add_action(DeclareLaunchArgument('attach_rpy', default_value='"0 0 3.14159265"', description='Attachment rpy offset'))
    ld.add_action(DeclareLaunchArgument('add_gripper', default_value='false', description='Add gripper'))
    ld.add_action(DeclareLaunchArgument('add_vacuum_gripper', default_value='false', description='Add vacuum gripper'))
    ld.add_action(DeclareLaunchArgument('add_bio_gripper', default_value='false', description='Add bio gripper'))

    # ----- RWBot simulation nodes (from the second launcher) -----
    pkg_rw = get_package_share_directory('rw')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Define the URDF file path using the 'model' launch argument
    urdf_file_path = PathJoinSubstitution([pkg_rw, 'model', LaunchConfiguration('model')])

    # Include the Gazebo simulation launch from ros_gz_sim
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [
                PathJoinSubstitution([pkg_rw, 'worlds', LaunchConfiguration('world')]),
                TextSubstitution(text=' -r -v -v1')
            ],
            'on_exit_shutdown': 'true'
        }.items()
    )
    ld.add_action(gazebo_launch)

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_rw, 'rviz', LaunchConfiguration('rviz_config')])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    ld.add_action(rviz_node)

    # Spawn the URDF model using the ros_gz_sim create service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "rw",
            "-topic", "robot_description",
            "-x", LaunchConfiguration('x'),
            "-y", LaunchConfiguration('y'),
            "-z", LaunchConfiguration('z'),
            "-Y", LaunchConfiguration('yaw')
        ],
        output="screen",
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    ld.add_action(spawn_urdf_node)

    # Bridge node for topics between Gazebo and ROS
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            "/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        ],
        output="screen",
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    ld.add_action(gz_bridge_node)

    # Bridge node for the camera image topic
    gz_image_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image"],
        output="screen",
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'camera.image.compressed.jpeg_quality': 75
        }]
    )
    ld.add_action(gz_image_bridge_node)

    # Relay node to republish camera info
    relay_camera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_camera_info',
        output='screen',
        arguments=['camera/camera_info', 'camera/image/camera_info'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    ld.add_action(relay_camera_info_node)

    # robot_state_publisher node (using xacro to process the URDF)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_file_path]),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        remappings=[('tf', 'tf'), ('tf_static', 'tf_static')]
    )
    ld.add_action(robot_state_publisher_node)

    # EKF node for state estimation
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_rw, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )
    ld.add_action(ekf_node)

    # ----- Include the xarm MoveIt configuration (from the first launcher) -----
    ld.add_action(OpaqueFunction(function=setup_moveit))

    return ld


if __name__ == '__main__':
    generate_launch_description()
