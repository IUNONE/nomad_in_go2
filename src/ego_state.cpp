/*
    Node: ego_state_node
    function: 1. receive high level ego state from "rt/sportmodestate" (real-time) or "lf/sportmodestate" (low-frequence)
              2. pub find pos in map and brocast tf
              map <--> base, base <--> odom
*/
#include "unitree_go/msg/sport_mode_state.hpp"

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include <string>

const std::string MAP_FRAME = "planner/map";
const std::string GOAL_FRAME = "planner/goal_xy";
const std::string EGO_FRAME = "planner/ego";
const std::string BASE_FRAME = "planner/base";

class EgoState : public rclcpp::Node
{
    public:
        EgoState() : Node("ego_state_node")
        {   
            // param = this->declare_parameter<std::string>("turtlename", "turtle");
            native_state_suber = this->create_subscription<unitree_go::msg::SportModeState>(
                "lf/sportmodestate",    // or  "sportmodestate" (500 Hz)
                20, 
                std::bind(&EgoState::state_callback, this, std::placeholders::_1)
            );

            // pub history path for past
            std_ego_state_puber = this->create_publisher<geometry_msgs::msg::PoseStamped>(
                "/MLplanner/ego_state", 
                20
            );

            // pub history path for past 
            history_path_puber = this->create_publisher<nav_msgs::msg::Path>(
                "/MLplanner/history_path", 
                2
            );

            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

            history_path.header.frame_id = BASE_FRAME;
            current_pose.header.frame_id = BASE_FRAME;
        };

    private:
        rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr native_state_suber;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr history_path_puber;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr std_ego_state_puber;

        nav_msgs::msg::Odometry current_pose;
        nav_msgs::msg::Path history_path;


        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        void state_callback(const unitree_go::msg::SportModeState::SharedPtr msg)
        {   
            RCLCPP_INFO(this->get_logger(), "Receieve a state at %d.%d", msg->stamp.sec, msg->stamp.nanosec);
            rclcpp::Time now = this->now();


            geometry_msgs::msg::TransformStamped t;
            // t.header.stamp.sec = msg->stamp.sec;
            // t.header.stamp.nanosec = msg->stamp.nanosec;
            t.header.stamp = now;
            RCLCPP_INFO(this->get_logger(), "Use system timestamp at %d.%d", t.header.stamp.sec, t.header.stamp.nanosec);
            t.header.frame_id = BASE_FRAME;
            t.child_frame_id = EGO_FRAME;
            t.transform.translation.x = msg->position[0];
            t.transform.translation.y = msg->position[1];
            t.transform.translation.z = 0; // msg->position[2];
            t.transform.rotation.w = msg->imu_state.quaternion[0];
            t.transform.rotation.x = msg->imu_state.quaternion[1];
            t.transform.rotation.y = msg->imu_state.quaternion[2];
            t.transform.rotation.z = msg->imu_state.quaternion[3];
            tf_broadcaster_->sendTransform(t);

            // history path
            geometry_msgs::msg::PoseStamped tmp_pose;
            history_path.header.stamp = now;
            // history_path.header.stamp.sec = msg->stamp.sec;
            // history_path.header.stamp.nanosec = msg->stamp.nanosec;
            if (history_path.poses.size() > 100)
                history_path.poses.erase(history_path.poses.begin());
            // tmp_pose.header.stamp.sec = msg->stamp.sec;
            // tmp_pose.header.stamp.nanosec = msg->stamp.nanosec;
            tmp_pose.header.stamp = now;
            tmp_pose.header.frame_id = BASE_FRAME;
            tmp_pose.pose.position.x = msg->position[0];
            tmp_pose.pose.position.y = msg->position[1];
            tmp_pose.pose.position.z = 0;
            tmp_pose.pose.orientation.w = msg->imu_state.quaternion[0];
            tmp_pose.pose.orientation.x = msg->imu_state.quaternion[1];
            tmp_pose.pose.orientation.y = msg->imu_state.quaternion[2];
            tmp_pose.pose.orientation.z = msg->imu_state.quaternion[3];
            history_path.poses.push_back(tmp_pose);
            std_ego_state_puber->publish(tmp_pose);
            history_path_puber->publish(history_path);
        }


};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<EgoState>()); //Run ROS2 node

    rclcpp::shutdown();
    return 0;
}
