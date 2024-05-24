/*
    Node: follow_traj_node
    function: 1. receive waypoints from planner; ( /planner/waypoints )
              2. convert to request message; 
                - message encode  ( class SportClient )
                - trajectory interpolation 
                - velocity generation
              3. publish request message;  ( unitree_api::msg::Request req to "/api/sport/request" )
*/

#include <unistd.h>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"

#include "unitree_api/msg/request.hpp"
#include "unitree_api/msg/response.hpp"
#include "common/ros2_sport_client.h"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"

using std::placeholders::_1;

class FollowTraj : public rclcpp::Node
{
    public:
        FollowTraj() : Node("follow_traj_node")
        {
            //--------------------------------sub-----------------------------------------------
            // sub waypoints from MLplanner
            planner_suber = this->create_subscription<nav_msgs::msg::Path>(
                "/MLplanner/pred_waypoints", 
                10, 
                std::bind(&FollowTraj::planner_callback, this, _1)
            );
            
            res_suber = this->create_subscription<unitree_api::msg::Response>(
                "/api/sport/response", 
                10, 
                std::bind(&FollowTraj::res_callback, this, _1)
            );

            //--------------------------------pub-----------------------------------------------
            // pub to control unit
            req_puber = this->create_publisher<unitree_api::msg::Request>(
                "/api/sport/request", 
                10
            );


            visu_traj_puber = this->create_publisher<nav_msgs::msg::Path>(
                "/MLplanner/track_waypoints", 
                10
            );

            //--------------------------------param-----------------------------------------------
            // this->declare_parameter<int>("my_parameter", 42);
            
        };
    private:
        rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr planner_suber;
        rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber;
        rclcpp::Subscription<unitree_api::msg::Response>::SharedPtr res_suber;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr visu_traj_puber;

        // rclcpp::TimerBase::SharedPtr timer;
    
        unitree_api::msg::Request req;
        SportClient get_req;
        
        const float MAX_V = 2.5;
        const int N_USE_WAYPOINTS = 4;
        const float CONTROL_STEP = 0.02;
        const float PLANNER_INTERVAL = 0.1;

        nav_msgs::msg::Path waypoint_track_msg;
        // const float kP = 1.0;
        // const float kD = 0.1;
        
        std::vector<std::pair<float, float>> waypoints;

        void res_callback(const unitree_api::msg::Response::SharedPtr msg)
        {
            RCLCPP_INFO(this->get_logger(), "Receieve a response with code %d", msg->header.status.code);
        }

        void planner_callback(const nav_msgs::msg::Path::SharedPtr msg)
        {   
            RCLCPP_INFO(this->get_logger(), "Receieve a path");
            waypoints.push_back(std::make_pair(0.0, 0.0));
            
            for (const auto& waypoint : msg->poses)
            {
                float x = waypoint.pose.position.x;
                float y = waypoint.pose.position.y;
                waypoints.push_back(std::make_pair(x, y));
            }


            std::vector<PathPoint> waypoints_to_track = simple_controller(waypoints, N_USE_WAYPOINTS);

            get_req.TrajectoryFollow(req, waypoints_to_track);
            req_puber->publish(req);
            
            waypoint_track_msg.header.stamp = msg->header.stamp;
            waypoint_track_msg.header.frame_id = "planner/base";
            visu_traj_puber->publish(waypoint_track_msg);

            auto go_to = waypoints_to_track.back();

            RCLCPP_INFO(this->get_logger(), "Send request to control unit, towards ( %f, %f, %f)", go_to.x, go_to.y, go_to.yaw);
            waypoints.clear();
            waypoint_track_msg.poses.clear();
        }

        std::vector<PathPoint> simple_controller(const std::vector<std::pair<float, float>>& waypoints, int num_to_track)
        {
            std::vector<PathPoint> waypoints_to_track;
            float yaw_ = 0;
            
            for (int i = 0; i <= num_to_track; ++i)  // only track first 0-0.2s (use 2 waypoints), and interpolate to 20 points
            {   
                float x0 = waypoints[i].first;
                float y0 = waypoints[i].second;
                float t0 = i * PLANNER_INTERVAL;

                float x1 = waypoints[i + 1].first;
                float y1 = waypoints[i + 1].second;
                float t1 = (i + 1) * PLANNER_INTERVAL;

                float yaw = atan2(y1 - y0, x1 - x0); // -pi ~ pi

                // assume in uniform motion
                float vx = (x1-x0) / PLANNER_INTERVAL;
                float vy = (y1-y0) / PLANNER_INTERVAL;
                float vyaw = (yaw - yaw_) / PLANNER_INTERVAL;

                vx = std::max(-MAX_V, std::min(MAX_V, vx));
                vy = std::max(-MAX_V, std::min(MAX_V, vy));
                vyaw = std::max(-MAX_V, std::min(MAX_V, vyaw));

                for (float t = t0 + CONTROL_STEP; t <= t1; t += CONTROL_STEP)
                {   
                    // interpolate
                    float x_interpolated = x0 + (x1 - x0) * (t - t0) / PLANNER_INTERVAL;
                    float y_interpolated = y0 + (y1 - y0) * (t - t0) / PLANNER_INTERVAL;
                    float yaw_interpolated  = yaw_ + (yaw - yaw_) * (t - t0) / PLANNER_INTERVAL;

                    // simple PD controller dx/dt
                    // float vx = kP * (x1 - x_interpolated) + kD * ((x1 - x0) / (t1 - t0));
                    // float vy = kP * (y1 - y_interpolated) + kD * ((y1 - y0) / (t1 - t0));
                    // float vyaw = kP * (yaw1 - yaw_interpolated) + kD * ((yaw1 - yaw0) / (t1 - t0));

                    PathPoint path_point_tmp;
                    path_point_tmp.timeFromStart = t;
                    path_point_tmp.x = x_interpolated;
                    path_point_tmp.y = y_interpolated;
                    path_point_tmp.yaw = yaw_interpolated;
                    path_point_tmp.vx = vx;
                    path_point_tmp.vy = vy;
                    path_point_tmp.vyaw = vyaw;
                    waypoints_to_track.push_back(path_point_tmp);
                    
                    // for visualize
                    geometry_msgs::msg::PoseStamped pose_stamped;
                    pose_stamped.header.frame_id = "planner/ego_odom";
                    pose_stamped.pose.position.x = x_interpolated;
                    pose_stamped.pose.position.y = y_interpolated;
                    waypoint_track_msg.poses.push_back(pose_stamped);
                }
                yaw_ = yaw;
            }

            return waypoints_to_track;
        }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<FollowTraj>()); //Run ROS2 node

    rclcpp::shutdown();
    return 0;
}