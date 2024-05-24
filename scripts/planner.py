'''
    Node: ml_planner_node
    function:   1. receive img from img_puber_node (img_puber.py) : '/MLplanner/front_cam_img'
                2. listen goal pos via tf from EGO_FRAME to GOAL_FRAME 
                3. forward model to predict ( use lightning )
                4. pub waypoints
'''
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker 

from cv_bridge import CvBridge
from torchvision import transforms
import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


import numpy as np
import torch
from collections import deque
import time
import sys
import copy
from threading import Lock
import threading

from model.nomad.nomad import NoMaD
from utils.transformation import  (
    to_base_frame, 
    yaw_from_quaternion, 
    array_to_path_xy,
    array_to_markers_points,
)

MAP_FRAME = 'planner/map'
GOAL_FRAME = 'planner/goal_xy'
EGO_FRAME = "planner/ego"
BASE_FRAME = "planner/base"

class MLPlanner(Node):
    def __init__(self):
        super().__init__('MLplanner_node')

        # ---------------------------- ROS -----------------------------------------------

        self.image_suber = self.create_subscription(
            Image, 
            '/MLplanner/front_cam_img', 
            self.img_callback,
            5,
        )
        self.odom_suber = self.create_subscription(
            PoseStamped,
            '/MLplanner/ego_state',
            self.odom_callback,
            10,
        )
        self.image_suber # prevent unused warning

        self.waypoint_puber= self.create_publisher(Path, '/MLplanner/pred_waypoints', 1)
        self.waypoint_vis_puber = self.create_publisher(Marker, '/MLplanner/waypoint_vis', 1)
        # ---------------------------- model -----------------------------------------------
        # input img
        self.cur_img_timestamp = None
        self.img_cache = deque(maxlen=21)
        self.bridge = CvBridge()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((96, 54)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # input goal coord: from tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.tf_static_listener = StaticTransformListener(self)
        self.goal_pos = torch.tensor([5.0, 5.0]).view(1, 2)
        # load model
        self.ckpt_path = './src/model/ckpt/nomad_200epch_10log.ckpt'
        self.model = NoMaD.load_from_checkpoint(self.ckpt_path)
        self.model.eval()
        self.context_size = 20
        # output
        self.pred_dist = None
        self.pred_waypoints_msg = Path()
        self.deploy_unnorm = {'min': 0, 'max': 1.5}
        self.get_new_img = False

        self.t = (0., 0., 0.) # [x, y, yaw]
        
        # --------------------------- thread -----------------------------------------------
        
    # img receiver : msg: sensor_msgs.msg.Image
    def img_callback(self, msg):
        # with self.lock:
        self.cur_img_timestamp = msg.header.stamp
        obs_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')   # TODO: check encoding
        obs_img = self.transform(obs_img)
        self.img_cache.append(obs_img)
        # self.get_logger().info(f'Receive IMG at timestamp : {msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9}')

        self.get_new_img = True

    def odom_callback(self, msg):
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        yaw = yaw_from_quaternion(q)
        self.t = ( msg.pose.position.x, msg.pose.position.y, yaw)
        # self.get_logger().info(f'Update ODOM at timestamp : {msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9}')

    def plan_waypoints(self):
        if ((len(self.img_cache) == (self.context_size+1))and self.get_new_img):
            # self._get_goal_pos_tf()
            # with self.lock:
            start_time = time.time()
            obs_images = torch.cat(list(self.img_cache), dim=0).unsqueeze(0)
            with torch.no_grad():
                # batch_obs_images: [B, C*(context+1), H, W] , batch_goal_pos: [B, 1, 2]
                goal_pos = self.goal_pos.cuda()
                obs_images = obs_images.cuda()
                noise_pred, self.pred_dist, _ =  self.model(obs_images, goal_pos)
                noise_pred = noise_pred.detach().cpu().numpy()
        
                # unnorm & cumsum : [B, len_traj_pred, 2]
                noise_pred = (noise_pred + 1) / 2 * (self.deploy_unnorm['max'] - self.deploy_unnorm['min']) + self.deploy_unnorm['min']
                pred_waypoints = np.cumsum(noise_pred, axis=1)
            
            # to base frame, np : [n, 2]
            pred_waypoints = to_base_frame(pred_waypoints.squeeze(0), self.t)

            # pub Path
            self.pred_waypoints_msg = array_to_path_xy(pred_waypoints, BASE_FRAME)
            self.pred_waypoints_msg.header.stamp = self.cur_img_timestamp
            self.waypoint_puber.publish(self.pred_waypoints_msg)

            # pub Marker to visualize
            marker = array_to_markers_points(pred_waypoints, BASE_FRAME)
            self.waypoint_vis_puber.publish(marker)

            end_time = time.time()
            self.get_logger().info(f'Prediction Done! Infer Time : {end_time - start_time} seconds')
            self.get_new_img = False

    def _get_goal_pos_tf(self):
        
        time = rclpy.time.Time.from_msg(self.cur_img_timestamp)
        t = self.tf_buffer.lookup_transform(
            GOAL_FRAME, # to_frame_rel
            BASE_FRAME, # from_frame_rel
            time,
        )
        self.goal_pos = torch.tensor([t.transform.translation.x, t.transform.translation.y]).view(1, 2)
        self.get_logger().info(f'Toward Goal Position: {self.goal_pos}')

def main(args=None):
    rclpy.init(args=args)

    planner = MLPlanner()
    # img_callback_thread = threading.Thread(target=rclpy.spin, args=(planner,), daemon=True)
    # img_callback_thread.start()

    try:
        while rclpy.ok():
            planner.plan_waypoints()
            rclpy.spin_once(planner)
    except KeyboardInterrupt:
        pass

    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main(sys.argv)