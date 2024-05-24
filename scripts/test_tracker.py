import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import numpy as np

MAP_FRAME = 'planner/map'
GOAL_FRAME = 'planner/goal_xy'
EGO_FRAME = 'planner/ego_odom'

class TestTracker(Node):
    def __init__(self):
        super().__init__('Test_tracker_node')

        # ---------------------------- ROS -----------------------------------------------

        self.waypoint_puber = self.create_publisher(Path, '/MLplanner/pred_waypoints', 1)
        self.timer = self.create_timer(0.2, self.pub_waypoints) # 5Hz

        # ---------------------------- test waypoints -----------------------------------------------
        # 40 waypoints with x, y, yaw
        t = np.linspace(0, 2*np.pi, 50)
        A, B = 1, 0.5
        self.x = A * np.sin(t)
        self.y = B * np.sin(2*t)

        dx_dt = A * np.cos(t)
        dy_dt = 2 * B * np.cos(2*t)
        self.yaw = np.arctan2(dy_dt, dx_dt)

        self.current_index = 0
        
    def pub_waypoints(self):
        
        # TODO: tramform waypoints to ego frame
        
        path_msg = Path()
        path_msg.header.frame_id = EGO_FRAME
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # pub 5 waypoints
        N = 5
        i = self.current_index
        x = self.x[i:i+N]
        y = self.y[i:i+N]
        yaw = self.yaw[i:i+N]

        if len(x) < N:
            x = np.concatenate((x, self.x[0:(N - len(x))]))
            y = np.concatenate((y, self.y[0:(N - len(y))]))
            yaw = np.concatenate((yaw, self.yaw[0:(N - len(yaw))]))

        waypoints = np.column_stack((x, y))

        init = np.array([x[0], y[0]])


        waypoints = self.to_local_coords(waypoints, init, yaw[0])

        for i in range(N-1):
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = waypoints[i+1][0]
            pose_stamped.pose.position.y = waypoints[i+1][1]
            path_msg.poses.append(pose_stamped)
        self.waypoint_puber.publish(path_msg)
        
        self.current_index += 2
        self.current_index %= 50

        self.get_logger().info(f'Pub waypoitns at Index: {self.current_index}')
    
    def to_local_coords(self, positions, curr_pos, curr_yaw):
        # positions: [N, 2]
        # curr_pos: [2]
        # curr_yaw: scalar
        rotmat =  np.array(
            [
                [np.cos(curr_yaw), -np.sin(curr_yaw), 0.0],
                [np.sin(curr_yaw), np.cos(curr_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        if positions.shape[-1] == 2:
            rotmat = rotmat[:2, :2]
        else:
            print("the positions shape is error: {positions.shape}")
            raise ValueError
    
        return (positions - curr_pos).dot(rotmat) 

def main(args=None):
    rclpy.init(args=args)

    planner = TestTracker()
    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()