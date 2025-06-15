# spawner_merged.launch.py (or rename to spawner.launch.py)

import os
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    RegisterEventHandler,
    LogInfo
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart

# uf_ros_lib utilities
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file


def launch_setup(context, *args, **kwargs):
    # ----------------------------------------------------------------
    # 1) Gather Launch Arguments (as LaunchConfiguration objects mostly)
    # ----------------------------------------------------------------
    dof = LaunchConfiguration('dof')
    robot_type = LaunchConfiguration('robot_type')
    prefix = LaunchConfiguration('prefix')
    hw_ns = LaunchConfiguration('hw_ns')
    limited = LaunchConfiguration('limited')
    attach_to = LaunchConfiguration('attach_to')
    attach_xyz = LaunchConfiguration('attach_xyz')
    attach_rpy = LaunchConfiguration('attach_rpy')
    add_gripper = LaunchConfiguration('add_gripper')
    add_vacuum_gripper = LaunchConfiguration('add_vacuum_gripper')
    add_bio_gripper = LaunchConfiguration('add_bio_gripper')
    ros_namespace_config = LaunchConfiguration('ros_namespace') # Keep as config for evaluation later
    use_sim_time = LaunchConfiguration('use_sim_time') # Keep as LaunchConfiguration
    show_rviz = LaunchConfiguration('show_rviz')
    rviz_config_arg = LaunchConfiguration('rviz_config')

    # Evaluate arguments needed for immediate logic (like path generation, controller names)
    prefix_str = prefix.perform(context)
    add_gripper_bool = add_gripper.perform(context).lower() == 'true'
    add_bio_gripper_bool = add_bio_gripper.perform(context).lower() == 'true'
    robot_type_str = robot_type.perform(context)
    ros_namespace_str = ros_namespace_config.perform(context)

    # ----------------------------------------------------------------
    # 2) Prepare xArm MoveIt config, ros2_control params, etc.
    # ----------------------------------------------------------------
    ros2_control_plugin = 'gz_ros2_control/GazeboSimSystem'

    # Generate a temporary ros2_control params file path
    ros2_control_params_path = generate_ros2_control_params_temp_file(
        os.path.join(get_package_share_directory('rw'), 'config', 'ros2_controllers.yaml'),
        prefix=prefix_str,
        add_gripper=add_gripper_bool,
        add_bio_gripper=add_bio_gripper_bool,
        ros_namespace=ros_namespace_str,
        update_rate=1000,
        use_sim_time=True, # Consistent with use_sim_time arg default
        robot_type=robot_type_str
    )

    # Paths to URDF/SRDF and config YAMLs
    pkg_path = get_package_share_directory('rw')
    urdf_file = os.path.join(pkg_path, 'model', 'rwbot_with_xarm.urdf.xacro')
    srdf_file = os.path.join(pkg_path, 'srdf', 'rwbot_with_xarm.srdf.xacro')
    controllers_file = os.path.join(pkg_path, 'config', 'controllers.yaml') # Used by MoveItConfigsBuilder
    joint_limits_file = os.path.join(pkg_path, 'config', 'joint_limits.yaml')
    kinematics_file = os.path.join(pkg_path, 'config', 'kinematics.yaml')
    pipeline_filedir = os.path.join(pkg_path, 'config') # Directory for planning pipelines

    # Build MoveIt configs using the builder
    moveit_configs = (
        MoveItConfigsBuilder("rwbot_with_xarm") # Robot name - ensure consistency if needed elsewhere
        .robot_description(file_path=urdf_file,
                           mappings={ # Pass mappings needed by xacro
                               "prefix": prefix,
                               "hw_ns": hw_ns,
                               "limited": limited,
                               "attach_to": attach_to,
                               "attach_xyz": attach_xyz,
                               "attach_rpy": attach_rpy,
                               "add_gripper": add_gripper,
                               "add_vacuum_gripper": add_vacuum_gripper,
                               "add_bio_gripper": add_bio_gripper,
                               "dof": dof,
                               "robot_type": robot_type,
                               "ros2_control_plugin": ros2_control_plugin,
                               "ros2_control_params_path": ros2_control_params_path,
                               # Add any other xacro args your URDF needs
                           })
        .robot_description_semantic(file_path=srdf_file,
                                    mappings={ # Pass mappings needed by SRDF xacro
                                        "prefix": prefix,
                                        "dof": dof,
                                        "robot_type": robot_type,
                                        "add_gripper": add_gripper,
                                        "add_bio_gripper": add_bio_gripper,
                                        # Add any other xacro args your SRDF needs
                                    })
        .robot_description_kinematics(file_path=kinematics_file)
        .joint_limits(file_path=joint_limits_file)
        .trajectory_execution(file_path=controllers_file)
        # Adjust planning pipelines based on your actual config files (e.g., ompl_planning.yaml)
        .planning_pipelines(
             default_planning_pipeline="ompl", # Or your default
             pipelines=["ompl", "stomp", "pilz_industrial_motion_planner"] # Adjust list based on your available configs
        )
        .to_moveit_configs()
    )

    # Get the dictionary for parameter passing to nodes
    moveit_config_dict = moveit_configs.to_dict()
    # Get the YAML dump string for the include launch file
    moveit_config_dump = yaml.dump(moveit_config_dict)

    # ----------------------------------------------------------------
    # 3) Define Core Simulation and Robot Nodes
    # ----------------------------------------------------------------

    # Robot State Publisher
    rsp_params = {
        'robot_description': moveit_config_dict['robot_description'], # Assign string value to key
        'use_sim_time': use_sim_time
    }
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[rsp_params], # Pass the single dictionary
        remappings=[
            ('/tf', PathJoinSubstitution([ros_namespace_config, 'tf'])),
            ('/tf_static', PathJoinSubstitution([ros_namespace_config, 'tf_static'])),
        ]
    )

    # Gazebo Launch
    # *** Use the EXACT extension found in your install space (.world or .wrold) ***
    # *** Revert to the string formatting method from the working file ***
    # --- Make SURE this extension matches your actual file ---
    world_file_extension = '.wrold' # <--- VERIFY THIS FILENAME EXTENSION IN YOUR INSTALLATION
    # --- If your file is actually .world, change the line above ---

    xarm_gazebo_world = PathJoinSubstitution([FindPackageShare('rw'), 'worlds', f'cracked_road_02{world_file_extension}'])

    # Evaluate the path immediately and create a single string argument
    gz_args_string = ' -r -v 3 {}'.format(xarm_gazebo_world.perform(context))
    LogInfo(msg=f"Gazebo arguments: '{gz_args_string}'").log(context=context) # Log the generated arguments for verification

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={
            'gz_args': gz_args_string, # Pass the pre-formatted string
        }.items(),
    )


    # Gazebo Spawn Entity Node
    gazebo_spawn_entity_node = Node(
        package="ros_gz_sim",
        executable="create",
        name='spawn_rwbot_with_xarm',
        output='screen',
        arguments=[
            '-topic', PathJoinSubstitution([ros_namespace_config, 'robot_description']), # Use namespaced topic if ns is set
            '-name', 'rwbot_with_xarm',
            "-x", '-0.88', "-y", '-2.1', "-z", '0.31', "-Y", '-1.52'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # RViz Node (Conditional)
    rviz_config_file_subst = PathJoinSubstitution([FindPackageShare('rw'), 'rviz', 'moveit.rviz'])
    rviz_params = {}
    rviz_params.update(moveit_config_dict) # moveit_config_dict contains resolved values
    rviz_params['use_sim_time'] = use_sim_time

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_arg if rviz_config_arg.perform(context) else rviz_config_file_subst],
        parameters=[rviz_params], # Pass single dictionary
        condition=IfCondition(show_rviz),
        remappings=[
            ('/tf', PathJoinSubstitution([ros_namespace_config, 'tf'])),
            ('/tf_static', PathJoinSubstitution([ros_namespace_config, 'tf_static'])),
            ('/joint_states', PathJoinSubstitution([ros_namespace_config,'joint_states'])),
            ('/display_planned_path', PathJoinSubstitution([ros_namespace_config,'display_planned_path'])),
            ('/monitored_planning_scene', PathJoinSubstitution([ros_namespace_config,'monitored_planning_scene'])),
            # Add more remappings if needed for namespaced MoveIt topics
        ]
    )

    # Load Controllers
    xarm_type_resolved = '{}{}'.format(robot_type_str, dof.perform(context) if robot_type_str in ('xarm', 'lite') else '')
    controllers_list = [
        'joint_state_broadcaster',
        f'{prefix_str}{xarm_type_resolved}_traj_controller'
    ]
    if robot_type_str != 'lite':
        if add_gripper_bool:
            controllers_list.append(f'{prefix_str}{robot_type_str}_gripper_traj_controller')
        elif add_bio_gripper_bool:
            controllers_list.append(f'{prefix_str}bio_gripper_traj_controller')

    controller_spawner_nodes = []
    controller_manager_name = PathJoinSubstitution([ros_namespace_config, 'controller_manager'])
    for controller in controllers_list:
        controller_spawner_nodes.append(Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            arguments=[ controller, '--controller-manager', controller_manager_name ],
            parameters=[{'use_sim_time': use_sim_time}],
        ))

    # ----------------------------------------------------------------
    # 4) Include MoveIt Common Launch (_robot_moveit_common2.launch.py)
    # ----------------------------------------------------------------
    robot_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_common2.launch.py'])
        ),
        # Pass all potentially relevant args the common launch might need
        launch_arguments={
            'prefix': prefix, 'hw_ns': hw_ns, 'limited': limited, 'dof': dof, 'robot_type': robot_type,
            'add_gripper': add_gripper, 'add_bio_gripper': add_bio_gripper, 'add_vacuum_gripper': add_vacuum_gripper, # Added vacuum just in case
            'attach_to': attach_to, 'attach_xyz': attach_xyz, 'attach_rpy': attach_rpy,
            'show_rviz': 'false', 'use_sim_time': use_sim_time,
            'moveit_config_dump': moveit_config_dump,
            # Pass the namespace if the common launch file needs it
            'ros_namespace': ros_namespace_config,
        }.items(),
    )

    # ----------------------------------------------------------------
    # 5) Add xArm Planner Node
    # ----------------------------------------------------------------
    planner_params_list = [
        # Wrap each parameter value in its own dictionary
        {'robot_description': moveit_config_dict.get('robot_description', '')},
        {'robot_description_semantic': moveit_config_dict.get('robot_description_semantic', '')},
        {'robot_description_kinematics': moveit_config_dict.get('robot_description_kinematics', {})},
        {'planning_pipelines': moveit_config_dict.get('planning_pipelines', {})},
         # Specific parameters for the planner itself (already a dictionary)
        {'robot_type': robot_type, 'dof': dof, 'prefix': prefix, 'use_sim_time': use_sim_time},
    ]

    xarm_planner_node = Node(
        package='xarm_planner',
        executable='xarm_planner_node',
        name='xarm_planner_node',
        namespace=ros_namespace_config,
        output='screen',
        parameters=planner_params_list, # Pass the list of dictionaries
        remappings=[ # Adjust remappings as needed
             ('/move_group/status', 'move_group/status'),
             ('/attached_collision_object', 'attached_collision_object'),
             ('/planning_scene', 'planning_scene'),
             # Add others as needed based on planner's subscriptions/clients
         ]
    )

    # ----------------------------------------------------------------
    # 6) Define Startup Sequence using Event Handlers
    # ----------------------------------------------------------------
    actions_to_return = [
        LogInfo(msg="Starting merged robot launch sequence..."),
        robot_state_publisher_node,
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=robot_state_publisher_node,
                on_start=[
                    LogInfo(msg="Robot State Publisher started. Launching Gazebo simulation..."),
                    gazebo_launch
                ]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessStart(
                 target_action=robot_state_publisher_node,
                 on_start=[
                    LogInfo(msg="Attempting to spawn robot in Gazebo... (Waiting for robot_description topic)"),
                    gazebo_spawn_entity_node # Spawner waits for the /robot_description topic
                 ]
             )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gazebo_spawn_entity_node,
                on_exit=[
                    LogInfo(msg="Robot spawned successfully. Starting controllers..."),
                    *controller_spawner_nodes,
                    LogInfo(msg="Starting MoveIt common components..."),
                    robot_moveit_common_launch,
                    LogInfo(msg="Starting xArm planner node..."),
                    xarm_planner_node,
                    LogInfo(msg="Checking if RViz should be launched..."),
                    rviz2_node # RViz node handles the IfCondition internally
                ]
            )
        ),
    ]
    return actions_to_return


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock?'),
        DeclareLaunchArgument('show_rviz', default_value='true', description='Launch RViz?'),
        DeclareLaunchArgument('rviz_config', default_value='', description='Optional full path to custom RViz config file'),
        DeclareLaunchArgument('dof', default_value='6', choices=['6', '7'], description='Degrees of freedom'),
        DeclareLaunchArgument('robot_type', default_value='xarm', choices=['xarm', 'lite', 'uf850'], description='Type of robot'),
        DeclareLaunchArgument('prefix', default_value='', description='Prefix for TF/topics'),
        DeclareLaunchArgument('hw_ns', default_value='xarm', description='Namespace for hardware interface'),
        DeclareLaunchArgument('limited', default_value='true', choices=['true', 'false'], description='Use limited joint ranges?'),
        DeclareLaunchArgument('attach_to', default_value='base_link', description='Link to attach arm to'),
        DeclareLaunchArgument('attach_xyz', default_value='"-0.039 0.0 0.091"', description='XYZ offset for attachment'),
        DeclareLaunchArgument('attach_rpy', default_value='"0 0 0"', description='RPY offset for attachment'),
        DeclareLaunchArgument('add_gripper', default_value='false', choices=['true', 'false'], description='Add standard gripper?'),
        DeclareLaunchArgument('add_vacuum_gripper', default_value='false', choices=['true', 'false'], description='Add vacuum gripper?'),
        DeclareLaunchArgument('add_bio_gripper', default_value='false', choices=['true', 'false'], description='Add bio gripper?'),
        DeclareLaunchArgument('ros_namespace', default_value='', description='Apply namespace to ROS nodes/topics'),
    ]

    pkg_rw = get_package_share_directory('rw')
    ekf_node = Node(
        package='robot_localization', executable='ekf_node', name='ekf_filter_node',
        namespace=LaunchConfiguration('ros_namespace'), output='screen',
        parameters=[
            os.path.join(pkg_rw, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    gz_bridge_node = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='parameter_bridge',
        namespace=LaunchConfiguration('ros_namespace'),
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/camera2/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera2/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        output="screen",
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    ld = LaunchDescription()
    for arg in declared_arguments:
        ld.add_action(arg)
    ld.add_action(gz_bridge_node)
    ld.add_action(ekf_node)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld