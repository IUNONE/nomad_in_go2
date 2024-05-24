'''
    There are 2 camera
    - FoV 120°, 720p(1280 * 720, 16: 9) / 180p(320 * 180) ( 25fps )
    - 1080P 30fps / 720p 60fps

    Node: img_puber_node
    function: 1. fetch video stream from camera via Gsteamer in cv2
              2. publish img with timestamp
'''

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import time # unit ns
from datetime import datetime   # unit s
from rclpy.time import Time 
from nav_msgs.msg import Path

# check the interface name : enx207bd2404ea5
GSTREAM_CONFIG = "udpsrc address=230.1.1.1 port=1720 multicast-iface=enx207bd2404ea5 ! \
application/x-rtp, media=video, encoding-name=H264 ! \
rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! \
video/x-raw,width=1280,height=720,format=BGR ! \
appsink drop=1"

class ImgPuber(Node):
    def __init__(self):
        super().__init__('img_puber_node')
        
        # capture video stream tools
        self.gstreamer_str = GSTREAM_CONFIG
        self.cap = cv2.VideoCapture(self.gstreamer_str, cv2.CAP_GSTREAMER)
        self.bridge = CvBridge()        
        
        self.timer = self.create_timer(0.1, self.publish_image)  # 10 Hz

        self.img_puber = self.create_publisher(
            Image, 
            '/MLplanner/front_cam_img', 
            5,
        )

        self.timestamp_suber = self.create_subscription(
            Path,
            '/MLplanner/history_path',
            self.timestamp_callback,
            10,
        )

        self.timestamp = None
    
    def publish_image(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret: # success read frame
                try:
                    image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    if self.timestamp is not None:
                        image_msg.header.stamp = self.timestamp
                    else:
                        self.get_logger().info(f'Open odom node to sync timestamp!')
                    
                    self.img_puber.publish(image_msg)
                    self.get_logger().info(f'Pub image at timestamp {image_msg.header.stamp.sec}')

                except CvBridgeError as e:
                    self.get_logger().error(f'{e}')
        else:
            self.get_logger().warning('Video stream capture tool is not ready!')

    def timestamp_callback(self, msg):
        self.timestamp = msg.header.stamp


def main(args=None):
    rclpy.init(args=args)

    img_puber = ImgPuber()
    rclpy.spin(img_puber)
    
    img_puber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)