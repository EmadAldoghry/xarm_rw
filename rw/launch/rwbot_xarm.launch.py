#!/usr/bin/env python3
import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, Command, PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):
    # ==============
    # 1) Arguments
    # ==============
    show_rviz     = LaunchConfiguration('show_rviz').perform(context)   # "true"/"false"
    rviz_config   = LaunchConfiguration('rviz_config').perform(context)
    dof           = LaunchConfiguration('dof').perform(context)         # e.g. "6"
    robot_type    = LaunchConfiguration('robot_type').perform(context)  # "xarm"
    prefix        = LaunchConfiguration('prefix').perform(context)      # e.g. ""
    ros_namespace = LaunchConfiguration('ros_namespace').perform(context)

    # ==============
    # 2) Paths
    # ==============
    # Package containing your RWBot + XArm Xacro and world
    pkg_rw = get_package_share_directory('rw')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    # Adjust if your Xacro is in a different location
    xacro_file = PathJoinSubstitution([pkg_rw, 'model', 'rwbot_with_xarm.urdf.xacro'])
    world_file = PathJoinSubstitution([pkg_rw, 'worlds', 'empty.world'])

    # ==============
    # 3) Generate URDF from Xacro
    # ==============
    # We pass dof, robot_type, prefix as needed. If your Xacro doesn't accept them, remove.
    robot_description_content = Command([
        'xacro ', xacro_file,
        ' ', f'dof:={dof}',
        ' ', f'robot_type:={robot_type}',
        ' ', f'prefix:={prefix}'
    ])
    # Used by the robot_state_publisher
    robot_description = {'robot_description': robot_description_content}

    # ==============
    # 4) Robot State Publisher
    # ==============
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            robot_description
        ],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )

    # ==============
    # 5) Gazebo / GZ Launch
    # ==============
    # If using Ignition/GZ. For Gazebo Classic, replace with 'gazebo_ros' approach
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={
            'gz_args': f'-r -v 3 {world_file.perform(context)}'
        }.items()
    )

    # ==============
    # 6) Spawn Robot in Gazebo
    # ==============
    spawn_entity_node = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'rwbot_with_xarm'
        ],
        parameters=[{'use_sim_time': True}]
    )

    # ==============
    # 7) Bridge sensor topics + cmd_vel
    # ==============
    # Using ros_gz_bridge. If your URDF uses <sensor type=...> with Ign topics, map them here
    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge_node',
        output='screen',
        arguments=[
            # clock
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # cmd_vel
            "/cmd_vel@geometry_msgs/msg/Twist[gz.msgs.Twist",
            # odom
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            # joint states
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",

            # LIDAR
            "scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",

            # Camera info
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # Depth image
            "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",

            # IMU
            "imu@sensor_msgs/msg/Imu[gz.msgs.IMU",

            # NavSat
            "/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat",
        ],
        parameters=[{'use_sim_time': True}]
    )

    # If your URDF publishes a color camera topic at "/camera/image", we can convert it to sensor_msgs/Image:
    gz_image_bridge_node = Node(
        package='ros_gz_image',
        executable='image_bridge',
        output='screen',
        arguments=["/camera/image"],
        parameters=[{'use_sim_time': True, 'camera.image.compressed.jpeg_quality': 75}]
    )

    # If your camera_info is published at /camera/camera_info, but you need /camera/image/camera_info, use a relay:
    relay_camera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_camera_info',
        output='screen',
        arguments=['camera/camera_info', 'camera/image/camera_info'],
        parameters=[{'use_sim_time': True}]
    )

    # ==============
    # 8) xArm Controllers
    # ==============
    # Make sure your ros2_controllers.yaml has: 
    #  joint_state_broadcaster, xarm6_traj_controller, etc.
    xarm_type = f'{robot_type}{dof}'  # e.g. xarm6
    controllers = [
        'joint_state_broadcaster',
        f'{prefix}{xarm_type}_traj_controller'
    ]
    controller_spawner_nodes = []
    for ctrl in controllers:
        controller_spawner_nodes.append(
            Node(
                package='controller_manager',
                executable='spawner',
                name=f'spawner_{ctrl}',
                arguments=[ctrl, '--controller-manager', f'{ros_namespace}/controller_manager'],
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        )

    # ==============
    # 9) Optional: RViZ
    # ==============
    # If user doesn't supply a custom .rviz file, pick one from xarm or your package
    if rviz_config == "":
        # Example: xarm_moveit_config's RViz
        rviz_config_final = PathJoinSubstitution([
            FindPackageShare('xarm_moveit_config'), 'rviz', 'moveit.rviz'
        ])
    else:
        rviz_config_final = rviz_config

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        condition=IfCondition(LaunchConfiguration('show_rviz')),
        arguments=['-d', rviz_config_final],
        parameters=[{'use_sim_time': True}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        name='rviz2'
    )

    # ==============
    # 10) Return all actions
    # ==============
    return [
        gazebo_launch,
        robot_state_publisher_node,
        spawn_entity_node,

        gz_bridge_node,
        gz_image_bridge_node,
        relay_camera_info_node,

        *controller_spawner_nodes,
        rviz_node
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('show_rviz',   default_value='false',
                              description="If 'true', launch RViz."),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description="Path to an RViz config file."),
        DeclareLaunchArgument('dof',         default_value='6',
                              description="Number of xArm joints (6 or 7)."),
        DeclareLaunchArgument('robot_type',  default_value='xarm',
                              description="Robot type (e.g. 'xarm')."),
        DeclareLaunchArgument('prefix',      default_value='',
                              description="Prefix for joint names, e.g. 'xarm_'."),
        DeclareLaunchArgument('ros_namespace', default_value='',
                              description="Namespace for controller manager, if any."),
        OpaqueFunction(function=launch_setup),
    ])
