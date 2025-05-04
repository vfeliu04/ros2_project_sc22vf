# Imports
import threading
import signal
import sys
import time
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import nav2_msgs.action


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    # Compute half-angle trig and combine to get quaternion
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return [qx, qy, qz, qw]


class DetectAndApproachBlue(Node):
    def __init__(self):
        super().__init__('detect_and_approach_blue_planner')
        # Image bridge and filtering thresholds
        self.bridge = CvBridge()
        self.noise_area_thresh = 500
        self.depth_estimate = 1.0

        # cmd_vel publisher to drive the robot manually
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Nav2 action client for waypoint navigation
        self.nav_client = ActionClient(self, nav2_msgs.action.NavigateToPose, 'navigate_to_pose')

        # Listener for transforms, and static broadcaster to set initial pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_bcast = tf2_ros.StaticTransformBroadcaster(self)

        # Camera and OpenCV windows setup
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

        # Predefined waypoints: (x, y, z, expected_color or none)
        self.waypoints = [
            (-0.433, -3.72, 0.00143, None),
            (3.19, -6.83, 0.00247, 'green'),
            (-5.71, -1.12, -0.00143, 'red'),
            (-4.16, -8.66, 0.00342, 'blue'),
        ]
        
        self.current_wp = 0
        self.detect_mode = False
        self.expected_color = None

        # Declare frames for parameters
        self.declare_parameter('camera_frame', 'camera_frame')
        self.declare_parameter('map_frame', 'map')
        self.get_logger().info('Node up: broadcasting static map --> odometry, then waypoints+detect')

        
        # Broadcast initial static transform 
        self.broadcast_initial_tf(0.00753, 0.00377, yaw=0.0)

        # Start waypoint sequence after a short delay
        self._delayed_start_timer = self.create_timer(1.0, self._delayed_start)

    def broadcast_initial_tf(self, x: float, y: float, yaw: float):
        # Init transformations 
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'map'
        tf.child_frame_id = 'odom'

        # Set translation
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = 0.0

        # Set rotation from euler yaw
        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw


    def _delayed_start(self):
        # Cancel timer, start sending waypoints
        self._delayed_start_timer.cancel()
        self.send_next_waypoint()

    def send_next_waypoint(self):
        """Send next waypoint as a NavigateToPose action goal"""
        if self.current_wp >= len(self.waypoints):
            self.get_logger().info('Mission Success!!')
            return

        x, y, z, _ = self.waypoints[self.current_wp]
        pose = PoseStamped()
        pose.header.frame_id = self.get_parameter('map_frame').value
        pose.header.stamp = self.get_clock().now().to_msg()

        # Fill in the target position
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        # Face original position
        qx, qy, qz, qw = quaternion_from_euler(0, 0, 0)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        # Wrap in Nav2 action goal and transmit to console
        goal = nav2_msgs.action.NavigateToPose.Goal()
        goal.pose = pose
        self.get_logger().info(
            f'---> Sending waypoint #{self.current_wp+1}: '
            f'x={x:.2f}, y={y:.2f}'
        )
        
        # Wait for Nav2 server and then send goal
        self.nav_client.wait_for_server()
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(self._on_waypoint_response)
        self.current_wp += 1

    def _on_waypoint_response(self, future):
        """Called when Nav2 accepts or rejects goal"""
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Waypoint goal was rejected')
            return
        gh.get_result_async().add_done_callback(self._on_waypoint_done)

    def _on_waypoint_done(self, future):
        """Called when Nav2 reports the goal has been reached or not"""
        goal_result = future.result()
        status_code = goal_result.status
        idx = self.current_wp - 1
        expected = self.waypoints[idx][3]
        self.get_logger().info(
            f'---> Reached waypoint #{idx+1} (status {status_code})'
        )

        # If a colored block is expected then switch to detect mode
        if expected is None:
            self.send_next_waypoint()
        else:
            self.expected_color = expected
            self.detect_mode = True
            self.get_logger().info(
                f'---> Looking for a {expected} block…'
            )

    def image_callback(self, msg: Image):
        """Process incoming camera frames and detect colored blocks"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # Show the raw camera feed
        cv2.imshow('Camera Feed', frame)
        if not self.detect_mode:
            # Not in detect mode: just update window
            cv2.waitKey(1)
            return

        # Convert to HSV for color masking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        col = self.expected_color

        # Set HSV thresholds based on expected color
        if col == 'green':
            lo, hi = np.array([40, 50, 50]), np.array([80, 255, 255])
            mask, draw = cv2.inRange(hsv, lo, hi), (0, 255, 0)
        elif col == 'red':
            lo1, hi1 = np.array([0, 100, 100]), np.array([10, 255, 255])
            lo2, hi2 = np.array([170, 100, 100]), np.array([180, 255, 255])
            m1 = cv2.inRange(hsv, lo1, hi1)
            m2 = cv2.inRange(hsv, lo2, hi2)
            mask, draw = cv2.bitwise_or(m1, m2), (0, 0, 255)
        elif col == 'blue':
            lo, hi = np.array([100, 100, 100]), np.array([140, 255, 255])
            mask, draw = cv2.inRange(hsv, lo, hi), (255, 0, 0)
        else:
            # Else if there is an unknown color then skip
            cv2.waitKey(1)
            return

        # Find contours in the mask
        det = frame.copy()
        cnts, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        best, br = 0, None
        for c in cnts:
            a = cv2.contourArea(c)
            if a > self.noise_area_thresh and a > best:
                best = a
                br = cv2.boundingRect(c)

        if br:
            # Draw bounding box around block
            x, y, w, h = br
            cv2.rectangle(det, (x, y), (x+w, y+h), draw, 2)
            cv2.putText(
                det,
                f'{col}: {best:.0f}',
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                draw,
                2
            )
            cv2.imshow('Detection', det)
            cv2.waitKey(1)

            # Stop rotating and announce detection of colored block
            self.cmd_pub.publish(Twist())
            self.get_logger().info(f'---> Detected {col} block!')

            # Reset detect mode and move to the next waypoint
            self.detect_mode = False
            time.sleep(0.5)
            self.send_next_waypoint()
        else:
            # Else if there is no block yet then keep spinning in place
            twist = Twist()
            twist.angular.z = 0.2
            self.cmd_pub.publish(twist)
            cv2.imshow('Detection', det)
            cv2.waitKey(1)


def main(args=None):
    # Initialize ROS client library and node
    rclpy.init(args=args)
    node = DetectAndApproachBlue()

    # Handle shut down properly
    def on_shutdown(sig, frame):
        node.get_logger().info('Shutting down')
        node.cmd_pub.publish(Twist())
        cv2.destroyAllWindows()
        rclpy.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, on_shutdown)

    # Communicate and listen in a separate thread so OpenCV can run windows in main thread
    thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    thread.start()

    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # Cleanup on exit
    node.get_logger().info('Exiting')
    node.cmd_pub.publish(Twist())
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
