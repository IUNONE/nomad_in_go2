'''
    Node: goal_tf_node
    function: 1. visualize the map where user can determine the goal pos via "2D Nav Tool" in rviz
              2. broadcast the goal & map & ego odom to tf
'''
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker


MAP_FRAME = 'planner/map'
GOAL_FRAME = 'planner/goal_xy'
EGO_FRAME = 'planner/ego_odom'
BASE_FRAME = 'planner/base'

class GoalTF(Node):
    def __init__(self):
        super().__init__('goal_tf_node')
        
        #------------------------ param manager
        # self.map_frame = self.declare_parameter(
        #   'map-frame-name', 'planner/map').get_parameter_value().string_value

        #------------------------ topic
        
        # Subscribe to the map topic
        # self.map_suber = self.create_subscription(
        #     Image,  # OccupancyGrid
        #     '/MLplanner/map',
        #     self.map_callback,
        #     1,
        # )
        # Subscribe to "2D Nav Goal" tool in rviz
        # self.goal_suber = self.create_subscription(
        #     PoseStamped,
        #     'move_base_simple/goal',
        #     self.goal_callback,
        #     1,
        # )

        # prevent unused variable warning
        # self.goal_suber
        # self.map_suber

        # Publish the goal position marker for visuilization in rviz
        self.marker_publisher = self.create_publisher(Marker, '/MLplanner/goal_marker', 20)
        
        # TF broadcaster
        # self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        #------------------------ variables

        self.map = None
        self.goal_pos_in_map = (5.0, 5.0)

        self.timer = self.create_timer(0.05, self.init)
        self.clock = self.get_clock()

        self.init()
    # def map_callback(self, msg):

    #     # grey scale map?
    #     self.map = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        
    #     self.get_logger().info('Received map data')
    def init(self):
        # check msg type?
        self.get_logger().info(f'Received goal pos: {self.goal_pos_in_map}')
        # TODO: check the msg frame_id ; pos should be in = MAP_FRAME
        self._visualize_pos_marker()
        # pub goal tf
        transform = self._get_transform_stamped(BASE_FRAME, GOAL_FRAME)
        self.tf_broadcaster.sendTransform(transform)        
    
    # def goal_callback(self, msg):
    #     # check msg type?
    #     self.goal_pos_in_map = (msg.pose.position.x, msg.pose.position.y)
    #     self.get_logger().info(f'Received goal pos: {self.goal_pos_in_map}')
    #     # TODO: check the msg frame_id ; pos should be in = MAP_FRAME
    #     self._visualize_pos_marker(msg.pose)
    #     # pub goal tf
    #     transform = self.get_transform_stamped(BASE_FRAME, GOAL_FRAME, msg)
    #     self.tf_static_broadcaster.sendTransform(transform)

    def _visualize_pos_marker(self):
        marker = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.goal_pos_in_map[0]
        marker.pose.position.y = self.goal_pos_in_map[1]
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_publisher.publish(marker)

    def _get_transform_stamped(self, parent_frame, child_frame):
        transform = TransformStamped()
        transform.header.stamp = self.clock.now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = self.goal_pos_in_map[0]
        transform.transform.translation.y = self.goal_pos_in_map[1]
        return transform

def main(args=None):
    rclpy.init(args=args)
    node = GoalTF()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
