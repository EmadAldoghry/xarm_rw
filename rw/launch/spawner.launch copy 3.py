# FILE: ~/ros2_ws/src/YOUR_PACKAGE_NAME/launch/YOUR_MAIN_LAUNCH_FILE.launch.py

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription


# For the sub-launch includes
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

# uf_ros_lib utilities (as in your original code)
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file


def launch_setup(context, *args, **kwargs):
    # ----------------------------------------------------------------
    # 1) Prepare xArm MoveIt config, ros2_control params, etc.
    # ----------------------------------------------------------------
    dof = LaunchConfiguration('dof', default=6)
    robot_type = LaunchConfiguration('robot_type', default='xarm')
    prefix = LaunchConfiguration('prefix', default='') # Ensure this prefix matches how the arm is named in MoveIt/Controllers
    hw_ns = LaunchConfiguration('hw_ns', default='xarm')
    limited = LaunchConfiguration('limited', default=True)
    attach_to = LaunchConfiguration('attach_to', default='base_link') # Frame where the xArm is attached on the rwbot
    attach_xyz = LaunchConfiguration('attach_xyz', default='"-0.039 0.0 0.091"') # Relative position of xArm base to attach_to frame
    attach_rpy = LaunchConfiguration('attach_rpy', default='"0 0 0"') # Relative orientation of xArm base to attach_to frame

    add_gripper = LaunchConfiguration('add_gripper', default=False)
    add_vacuum_gripper = LaunchConfiguration('add_vacuum_gripper', default=False)
    add_bio_gripper = LaunchConfiguration('add_bio_gripper', default=False)

    ros_namespace = LaunchConfiguration('ros_namespace', default='').perform(context)
    ros2_control_plugin = 'gz_ros2_control/GazeboSimSystem' # Use 'gazebo_ros2_control/GazeboSystem' for older Gazebo versions if needed

    # Generate a temporary ros2_control params file
    # Ensure the controllers defined in 'ros2_controllers.yaml' match the 'prefix' and 'robot_type' + 'dof'
    ros2_control_params_config_file = os.path.join(get_package_share_directory('rw'), 'config', 'ros2_controllers.yaml')
    ros2_control_params = generate_ros2_control_params_temp_file(
        ros2_control_params_config_file,
        prefix=prefix.perform(context),
        add_gripper=add_gripper.perform(context) in ('True', 'true'),
        add_bio_gripper=add_bio_gripper.perform(context) in ('True', 'true'),
        ros_namespace=ros_namespace,
        update_rate=1000, # Gazebo update rate
        use_sim_time=True,
        robot_type=robot_type.perform(context)
    )

    # Paths to your URDF/SRDF and config YAMLs (Ensure these files exist and are correct)
    pkg_path = os.path.join(get_package_share_directory('rw'))
    # --- IMPORTANT: Ensure xacro files correctly handle the 'prefix' argument if needed ---
    urdf_file = os.path.join(pkg_path, 'model', 'rwbot_with_xarm.urdf.xacro')
    srdf_file = os.path.join(pkg_path, 'srdf', 'rwbot_with_xarm.srdf.xacro')
    controllers_file = os.path.join(pkg_path, 'config', 'controllers.yaml') # Used by trajectory_execution manager in MoveIt
    joint_limits_file = os.path.join(pkg_path, 'config', 'joint_limits.yaml') # Optional, can be part of URDF or SRDF
    kinematics_file = os.path.join(pkg_path, 'config', 'kinematics.yaml') # Defines IK solver, etc.
    pipeline_filedir = os.path.join(pkg_path, 'config') # Directory containing ompl_planning.yaml etc.

    # --- Build MoveIt configs dynamically ---
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
            ros2_control_params=ros2_control_params, # Pass path to generated temp file
            add_gripper=add_gripper,
            add_vacuum_gripper=add_vacuum_gripper,
            add_bio_gripper=add_bio_gripper,
        )
        # Explicitly set paths - ensure MoveItConfigsBuilder uses these correctly
        .robot_description(file_path=urdf_file)
        .robot_description_semantic(file_path=srdf_file)
        .robot_description_kinematics(file_path=kinematics_file)
        .joint_limits(file_path=joint_limits_file)
        .trajectory_execution(file_path=controllers_file)
        # ****************************************************************
        # --- CORRECTED planning_pipelines call ---
        # Pass the *directory* containing the pipeline config YAML files
        # using the correct argument name 'config_folder'.
        # Leave 'pipelines=None' (default) to let the builder auto-detect
        # *_planning.yaml files (like ompl_planning.yaml) in that folder.
        .planning_pipelines(config_folder=pipeline_filedir)
        # ****************************************************************
        .to_moveit_configs()
    )
    # Dump the generated config to pass to sub-launches
    moveit_config_dump = yaml.dump(moveit_config.to_dict())

    # ----------------------------------------------------------------
    # 2) Include sub-launch: _robot_on_rwbot_gz.launch.py
    #    This starts Gazebo, spawns the ROBOT (rwbot+xarm), loads controllers via controller_manager
    # ----------------------------------------------------------------
    robot_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('rw'), 'launch', '_robot_on_rwbot_gz.launch.py'])
        ),
        launch_arguments={
            'dof': dof,
            'robot_type': robot_type,
            'prefix': prefix,
            'add_gripper': add_gripper,
            'add_bio_gripper': add_bio_gripper,
            'ros_namespace': ros_namespace,
            'moveit_config_dump': moveit_config_dump, # Pass generated config
            'show_rviz': 'false', # Usually MoveIt common launch handles RViz
            # Pass any other args needed by _robot_on_rwbot_gz.launch.py
        }.items(),
    )

    # ----------------------------------------------------------------
    # 3) Include sub-launch: _robot_moveit_common2.launch.py
    #    This starts MoveIt nodes (move_group). RViz can be optionally started here.
    # ----------------------------------------------------------------
    rviz_config_file = PathJoinSubstitution([FindPackageShare('rw'), 'rviz', 'moveit.rviz'])
    robot_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            # Assuming this common launch file exists and works with the config dump
            PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_common2.launch.py'])
        ),
        launch_arguments={
            # Pass necessary arguments - ensure these match what _robot_moveit_common2 expects
            'prefix': prefix,
            'dof': dof,
            'robot_type': robot_type,
            'add_gripper': add_gripper,
            # Arguments related to physical attachment (might not be needed by common launch?)
            'attach_to': attach_to,
            'attach_xyz': attach_xyz,
            'attach_rpy': attach_rpy,
            # Control RViz display
            'show_rviz': 'true', # Set to true to show RViz from here
            'rviz_config': rviz_config_file,
            # Pass simulation time and generated config
            'use_sim_time': 'true',
            'moveit_config_dump': moveit_config_dump,
        }.items(),
    )

    # ****************************************************************
    # 4) Launch the xarm_planner_node
    #    Provides the services your Python node needs
    # ****************************************************************
    # Prepare parameters for the planner node by extracting from moveit_config
    # The planner node primarily needs the robot description parts to initialize MoveGroupInterface internally
    move_group_interface_params = {}
    move_group_interface_params.update(moveit_config.robot_description)
    move_group_interface_params.update(moveit_config.robot_description_semantic)
    move_group_interface_params.update(moveit_config.robot_description_kinematics)

    xarm_planner_node = Node(
        name='xarm_planner_node', # Standard name, can be namespaced if needed
        package='xarm_planner',
        executable='xarm_planner_node',
        output='screen',
        parameters=[
            move_group_interface_params, # Pass robot descriptions
            {
                # Parameters to identify the target arm group within MoveIt
                'robot_type': robot_type,
                'dof': dof,
                'prefix': prefix # Crucial if you used a prefix in MoveIt/SRDF
            },
            # Ensure simulation time is used by the planner node too
            {'use_sim_time': True},
            # Add any specific planner parameters here if needed (e.g., velocity/accel scaling, though defaults are set in C++)
        ],
        # Add remappings if your service names are non-standard or namespaced
        remappings=[
            ('xarm_straight_plan', '/my_ns/xarm_straight_plan'),
            ('xarm_exec_plan', '/my_ns/xarm_exec_plan'),
            # Add other services if needed (pose_plan, joint_plan)
        ]
    )

    # --- Return list of actions ---
    # Decide which RViz to show (from Gazebo launch or MoveIt common launch)
    # If both are set to true, you might get two RViz windows.

    nodes_to_launch = [
        robot_gazebo_launch,
        robot_moveit_common_launch,
        xarm_planner_node,
    ]

    return nodes_to_launch


# Main launch description function
def generate_launch_description():
    # Declare top-level launch arguments allowing override from command line
    declared_arguments = []
    declared_arguments.append(DeclareLaunchArgument(
        'use_sim_time', default_value='true', # Use 'true', not 'True' for command line consistency
        description='Use simulation (Gazebo) clock.'))
    declared_arguments.append(DeclareLaunchArgument(
        'dof', default_value='6', description='Degrees of freedom of the xArm (e.g., 5, 6, 7)'))
    declared_arguments.append(DeclareLaunchArgument(
        'robot_type', default_value='xarm', description='Type of robot (xarm, lite, uf850)'))
    declared_arguments.append(DeclareLaunchArgument(
        'prefix', default_value='', description='Prefix for joints and links (e.g., "L_")'))
    declared_arguments.append(DeclareLaunchArgument(
        'add_gripper', default_value='false', description='Add the standard xArm gripper'))
    declared_arguments.append(DeclareLaunchArgument(
        'add_bio_gripper', default_value='false', description='Add the Bio gripper'))
    declared_arguments.append(DeclareLaunchArgument(
        'ros_namespace', default_value='', description='Namespace for ROS topics/services, if any'))
    declared_arguments.append(DeclareLaunchArgument(
        'limited', default_value='true', description='Use limited range joint limits'))
    declared_arguments.append(DeclareLaunchArgument(
        'attach_to', default_value='base_link', description='Link on rwbot to attach xArm base to'))
    declared_arguments.append(DeclareLaunchArgument(
        'attach_xyz', default_value='"-0.039 0.0 0.091"', description='XYZ offset of xArm base relative to attach_to link'))
    declared_arguments.append(DeclareLaunchArgument(
        'attach_rpy', default_value='"0 0 0"', description='RPY offset of xArm base relative to attach_to link'))
     # Add other arguments you might want to control from the command line

    # Nodes that run regardless of the sub-launches (e.g., bridging, localization)
    # Ensure these are necessary and not duplicated in sub-launches

    # EKF Node (Localization)
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(get_package_share_directory('rw'), 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        # Ensure remappings match topics published by Gazebo/Bridge
        remappings=[('odometry/filtered', 'odom')] # Example remapping
    )

    # Gazebo Bridge Node
    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge', # Give it a name
        arguments=[
            # Clock
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # Base Control/Feedback
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist", # Bridge TO Gazebo for Twist
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",    # Bridge FROM Gazebo for Odometry
            # Arm Joint States (Ensure topic name matches Gazebo publisher)
            # Check Gazebo topics (gz topic -l) if '/joint_states' doesn't work
            # It might publish to something like /world/<world_name>/model/<model_name>/joint_state
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model", # Check if gz.msgs.Model is correct type
             # Sensors (Update topics based on your URDF/SDF sensor definitions)
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/camera2/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo", # If you have camera2
            "/camera2/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera2/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            # "/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked", # Often redundant with /scan
            "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU", # Common IMU topic suffix is /data
        ],
        output="screen",
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        # Remappings if needed (e.g., if Gazebo topic names differ)
        # remappings=[
        #      ('/imu/data', '/imu_gz'),
        # ]
    )

    # Relay node (Often not needed if TF setup is correct)
    # relay_camera_info_node = Node(
    #     package='topic_tools',
    #     executable='relay',
    #     name='relay_camera_info',
    #     output='screen',
    #     arguments=['camera/camera_info', 'camera/image/camera_info'],
    #     parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    # )


    # Create the final launch description
    ld = LaunchDescription(declared_arguments)

    # Add the standalone nodes
    ld.add_action(ekf_node)
    ld.add_action(gz_bridge_node)
    # ld.add_action(relay_camera_info_node) # Add if needed

    # Add the OpaqueFunction to execute the main setup logic
    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld