import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace


def generate_launch_description():

    # ------------------------- SET PARAMETERS -------------------------
    # args that can be set from the command line or a default will be used
    
    # background_g_launch_arg = DeclareLaunchArgument(
    #     "background_g", default_value=TextSubstitution(text="255")
    # )


    # -------------------------INCLUDE ANOTHER LAUNCH FILE-------------------------
    
    # launch_include = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('demo_nodes_cpp'),
    #             'launch/topics/talker_listener.launch.py'))
    # )
    # #include another launch file in the chatter_ns namespace
    # launch_include_with_namespace = GroupAction(
    #     actions=[
    #         # push_ros_namespace to set namespace of included nodes
    #         PushRosNamespace('chatter_ns'),
    #         IncludeLaunchDescription(
    #             PythonLaunchDescriptionSource(
    #                 os.path.join(
    #                     get_package_share_directory('demo_nodes_cpp'),
    #                     'launch/topics/talker_listener.launch.py'))
    #         ),
    #     ]
    # )

    # --------------------------- START NODES ---------------------------

    img_puber_node = Node(
            package='planner_in_go2',
            executable='img_puber',
            name='ImgPuberNode'
        )

    goal_tf_node = Node(
            package='planner_in_go2',
            executable='goal_tf',
            name='GoalTfNode'
        )
    
    ml_planner_node = Node(
            package='planner_in_go2',
            executable='planner',
            name='PlannerNode'
        )

    ego_state_node = Node(
            package='planner_in_go2',
            executable='ego_state',
            name='EgoStateNode'
        )
    
    rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/zhangshenghong_lx/Nutstore Files/planner_in_go2/rviz/planner.rviz'],
        )
    
    # turtlesim_node_with_parameters = Node(
    #         package='turtlesim',
    #         namespace='turtlesim2',
    #         executable='turtlesim_node',
    #         name='sim',
    #         parameters=[{
    #             "background_r": LaunchConfiguration('background_r'),
    #             "background_g": LaunchConfiguration('background_g'),
    #             "background_b": LaunchConfiguration('background_b'),
    #         }],
    #         remappings=[
    #             ('/input/pose', '/turtlesim1/turtle1/pose'),
    #             ('/output/cmd_vel', '/turtlesim2/turtle1/cmd_vel'),
    #         ]
    #     )

    # --------------------------- RETURN LAUNCH OBJECTS ---------------------------
    return LaunchDescription([
        # background_g_launch_arg,
        # launch_include_with_namespace,
        img_puber_node,
        goal_tf_node,
        ml_planner_node,
        # tracker_node,
        ego_state_node,
        rviz_node,
    ])