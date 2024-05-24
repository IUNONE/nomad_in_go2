/*
    NOTICE: NOT USED NOW !!
    
    Node: joy_stick_node
    function: 1. fetch data from joystick, how?
              2. publish data to unitree_api to control go1 to move : ( https://support.unitree.com/home/zh/developer/sports_services )

*/
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg//wireless_controller.hpp"
using std::placeholders::_1;

class JoyStickerControl : public rclcpp::Node
{
    public:
        JoyStickerControl() : Node("custom_joystick_node")
        {   
            //-------------------------------sub------------------------
            // wireless topic from which unit? ( it seems that dds )
            joy_suber = this->create_subscription<unitree_go::msg::WirelessController>(
                "/wirelesscontroller", 
                10, 
                std::bind(&wireless_controller_suber::joystick_topic_callback, this, _1)
            );

            //-------------------------------pub-------------------------
            // pub cmd for moving
            req_puber = this->create_publisher<unitree_api::msg::Request>(
                "/api/sport/request", 
                10
            ); 

            //-------------------------------init------------------------
            
            get_req.SwitchJoystick(req_on, False);
            req_puber->publish(req_on);    
            RCLCPP_INFO(this->get_logger(), "turn OFF native joy-stick, use self-define one!");
        }


    private:

        rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr suber;
        
        rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber;

        unitree_api::msg::Request req_on;
        unitree_api::msg::Request req_move;
        unitree_api::msg::Request req_stop;

        SportClient get_req;
        
        float MAX_V_FORWARD = 5.0;
        float MAX_V_BACKWARD = 2.5;
        float vx = 0; 
        float vy = 0; 
        float vyaw = 0;

        bool stop = false;

        void joystick_topic_callback(unitree_go::msg::WirelessController::SharedPtr msg)
        {
            // map to vx, vy, vyaw
            if (msg->ly > 0)
            {
                vx = _map_to()(msg->ly, 0.0, 1.0, 0.0, MAX_V_FORWARD); // move forward
            }
            else
            {
                vx = _map_to()(msg->ly, -1.0, 0.0, -MAX_V_BACKWARD, 0.0); // move backward
            }

            if (msg->lx > 0)
            {
                vy = -_map_to()(msg->lx, 0.0, 1.0, 0.0, MAX_V_BACKWARD); // move right
            }
            else
            {
                vy = -_map_to()(msg->lx, -1.0, 0.0, -MAX_V_BACKWARD, 0.0); // move left
            }

            if (msg->rx > 0)
            {
                vyaw = -_map_to()(msg->rx, 0.0, 1.0, 0.0, MAX_V_BACKWARD); // turn right
            }
            else
            {
                vyaw = -map_to()(msg->rx, -1.0, 0.0, -MAX_V_BACKWARD, 0.0); // turn left
            }


            // How to set to stopmove
            stop = msg->keys
        
            // TODO: pub to control to go2
            if (!stop) 
            {
                get_req.Move(req_move, vx, vy, vyaw);   // get req message
                req_puber->publish(req_move);
            }
            else 
            {
                get_req.StopMove(req_stop);
                req_puber->publish(req_stop);
            }
            
        }

        float _map_to()(float value, float inMin, float inMax, float outMin, float outMax) 
        {
            return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
        }

}


int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<JoyStickerControl>()); 

  rclcpp::shutdown();
  return 0;
}