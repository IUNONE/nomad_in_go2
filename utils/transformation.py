# https://docs.ros.org/en/foxy/Tutorials/Intermediate/Tf2/Quaternion-Fundamentals.html
import numpy as np

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

def quaternion_multiply(q0, q1):
    """
    Multiplies two quaternions.

    Input
    :param q0: A 4 element array containing the first quaternion (q01, q11, q21, q31)
    :param q1: A 4 element array containing the second quaternion (q02, q12, q22, q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    # Extract the values from q0
    w0 = q0[0]
    x0 = q0[1]
    y0 = q0[2]
    z0 = q0[3]

    # Extract the values from q1
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]

    # Computer the product of the two quaternions, term by term
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion

def quaternion_from_yaw():
    pass

def yaw_from_quaternion(quaternion):
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw

def array_trans_to_base():
    # transform [3, n] (x, y, yaw) to base frame
    pass

def array_to_path_xy(waypoints: np.ndarray, frame: str):
    '''
        convert [n, 2] (x, y) to Path
    ''' 

    # [1, n, 2] --> [n, 2]
    if waypoints.shape[0] == 1:
        waypoints = waypoints.squeeze(0)

    path = Path()
    path.header.frame_id = frame
    for i in range(waypoints.shape[0]):
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = float(waypoints[i][0])
        pose_stamped.pose.position.y = float(waypoints[i][1])
        pose_stamped.header.frame_id = frame
        # pose_stamped.header.stamp = self.cur_img_timestamp 
        path.poses.append(pose_stamped)
    
    return path

def array_to_markers_points(waypoints: np.ndarray, frame: str):
    # convert [n, 2] (x, y) to Marker (points)
    
    marker_points = Marker()
    marker_points.header.frame_id = frame
    marker_points.type = Marker.POINTS
    marker_points.action = Marker.ADD
    marker_points.scale.x = 0.01
    marker_points.scale.y = 0.01
    marker_points.scale.z = 0.01
    marker_points.color.a = 1.0
    marker_points.color.r = 0.0
    marker_points.color.g = 1.0
    marker_points.color.b = 0.0

    for i in range(waypoints.shape[0]):
        point = Point()
        point.x = float(waypoints[i][0])
        point.y = float(waypoints[i][1])
        marker_points.points.append(point)

    return marker_points

def get_rotmat_from_yaw(yaw: float) -> np.ndarray:
    # scalar --> [3, 3] rotation matrix
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def to_base_frame(positions: np.ndarray, trans: tuple) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert [2, n]
        trans (tuple): (x ,y ,yaw)
        curr_pos (np.ndarray): current position [2]
        curr_yaw (float): current yaw [1]
        the yaw direction is the 
    Returns:
        np.ndarray: positions in local coordinates
    """

    curr_pos = np.array([trans[0], trans[1]])
    rotmat = get_rotmat_from_yaw(-trans[2])
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return curr_pos + positions.dot(rotmat)