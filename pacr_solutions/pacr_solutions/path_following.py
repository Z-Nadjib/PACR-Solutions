"""ROS2 Node for Path Following with Behavior-Based Local Planning.

This node integrates the Pure Pursuit controller with a local reactive
planner to achieve safe navigation in dynamic environments. It uses
TF to track concurrent robots and provides a service to the executive
node for high-level command generation.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Duration, Time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point as GeoPoint

from pacr_interfaces.srv import GetCmd
from pacr_interfaces.msg import Velocity2D

from pacr_simulation.geometry2d import Pose, Point, Vector, Transform
from pacr_simulation.map_utils import GridMap
from pacr_solutions.controller import PathFollower
from pacr_solutions.local_planner import LocalPlanner, Scenario


# ─────────────────────────────────────────────
# Path Following Node
# ─────────────────────────────────────────────

class PathFollowingNode(Node):
    """ROS2 Node for path following with local collision avoidance.

    Orchestrates the interplay between global path tracking and 
    local behaviors.
    """

    def __init__(self):
        super().__init__('path_following')

        # ── Parameters ──
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('safety_radius', 2.5)
        self.declare_parameter('v_boost_factor', 1.5)
        self.declare_parameter('static_frame_id', 'world')
        self.declare_parameter('other_frame_id', 'concurrent_robot/base_link')

        v_max = self.get_parameter('v_max').get_parameter_value().double_value
        w_max = self.get_parameter('w_max').get_parameter_value().double_value
        lookahead = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        safety_radius = self.get_parameter('safety_radius').get_parameter_value().double_value
        v_boost_factor = self.get_parameter('v_boost_factor').get_parameter_value().double_value
        self.static_frame_id = self.get_parameter('static_frame_id').get_parameter_value().string_value
        self.other_frame_id = self.get_parameter('other_frame_id').get_parameter_value().string_value

        # ── Pure Pursuit Controller ──
        self.controller = PathFollower(
            v_max=v_max,
            w_max=w_max,
            lookahead_distance=lookahead
        )

        # ── Behavior-Based Local Planner ──
        self.local_planner = LocalPlanner(
            safety_radius=safety_radius,
            v_boost_factor=v_boost_factor,
            v_max=v_max,
            w_max=w_max
        )

        # ── TF Infrastructure ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Map Awareness ──
        self.grid_map = None
        latching_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, 'costmap', self._costmap_cb,
            qos_profile=latching_qos)

        # ── Visualization ──
        self.marker_pub = self.create_publisher(
            MarkerArray, 'local_planner/markers', 1)

        # ── Service Interface ──
        self.srv = self.create_service(GetCmd, 'get_cmd', self.get_cmd_cb)

        self.get_logger().info(
            f'PathFollowingNode initialized with local planner '
            f'(safety_radius={safety_radius}, boost={v_boost_factor}x).')

    # ─────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────

    def _costmap_cb(self, msg: OccupancyGrid):
        """Buffer the incoming costmap for spatial reasoning."""
        self.grid_map = GridMap.from_msg(msg)
        self.get_logger().info('Costmap received for local planner.')

    def _get_other_pose(self) -> Pose:
        """Retrieve the concurrent robot's pose from TF."""
        transform = self.tf_buffer.lookup_transform(
            self.static_frame_id,
            self.other_frame_id,
            Time(),
        )
        return Pose.from_transform(Transform.from_msg(transform.transform))

    def get_cmd_cb(self, request, response):
        """
        Main control service callback.
        
        Evaluates local collision threats before falling back to
        the global Pure Pursuit controller.
        """
        robot_pose = Pose.from_msg(request.pose)
        goal_pose = Pose.from_msg(request.goal)
        path = [Point.from_msg(pt) for pt in request.path]

        # ── Update World State ──
        other_pose = None
        try:
            other_pose = self._get_other_pose()
        except TransformException as e:
            self.get_logger().debug(
                f'Could not get concurrent robot pose: {e}')

        # ── Local Planner Logic ──
        scenario = Scenario.NONE
        override_cmd = None

        if other_pose is not None:
            # Update Direction Estimators
            self.local_planner.update(
                robot_pose.position, other_pose.position)

            # Check for collision scenarios
            scenario, override_cmd = self.local_planner.evaluate(
                robot_pose, other_pose.position,
                request.path, self.grid_map)

            # Visualize current state
            self._publish_markers(robot_pose, other_pose, scenario)

        # ── Command Selection ──
        if override_cmd is not None:
            # Local planner override is active
            v, w = override_cmd
            # Ensure override respects velocity limits
            v = max(-self.controller.v_max * self.local_planner.v_boost_factor,
                    min(self.controller.v_max * self.local_planner.v_boost_factor, v))
            w = max(-self.controller.w_max,
                    min(self.controller.w_max, w))

            response.command = Velocity2D(linear=v, angular=w)
            response.finished = False

            self.get_logger().info(
                f'[LocalPlanner] Scenario {scenario.name}: '
                f'v={v:.2f}, w={w:.2f}')
        else:
            # Normal Pure Pursuit path tracking
            try:
                v, w, is_finished = self.controller.get_command(
                    robot_pose, path, goal_pose)

                response.command = Velocity2D(linear=v, angular=w)
                response.finished = is_finished

            except Exception as e:
                self.get_logger().error(f'Error computing command: {e}')
                response.command = Velocity2D(linear=0.0, angular=0.0)
                response.finished = True

        return response

    # ─────────────────────────────────────
    # RViz Visualization
    # ─────────────────────────────────────

    def _publish_markers(self, self_pose: Pose, other_pose: Pose,
                         scenario: Scenario):
        """Construct and publish visualization markers array."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame = self.static_frame_id

        # ── 1. Safety Radius Circle ──
        radius_marker = Marker()
        radius_marker.header.frame_id = frame
        radius_marker.header.stamp = stamp
        radius_marker.ns = 'local_planner'
        radius_marker.id = 0
        radius_marker.type = Marker.CYLINDER
        radius_marker.action = Marker.ADD
        radius_marker.pose.position.x = self_pose.x
        radius_marker.pose.position.y = self_pose.y
        radius_marker.pose.position.z = 0.05
        radius_marker.scale.x = self.local_planner.safety_radius * 2.0
        radius_marker.scale.y = self.local_planner.safety_radius * 2.0
        radius_marker.scale.z = 0.01
        
        # Color based on threat status
        if scenario == Scenario.NONE:
            radius_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.15)
        else:
            radius_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.25)
        
        radius_marker.lifetime = Duration(seconds=0, nanoseconds=200_000_000).to_msg()
        markers.markers.append(radius_marker)

        # ── 2. Direction Vectors (Arrows) ──
        other_dir = self.local_planner.other_estimator.get_unit_direction()
        if other_dir is not None:
            other_arrow = self._make_arrow(
                2, frame, stamp,
                other_pose.position, other_dir, length=2.0,
                color=ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9))
            markers.markers.append(other_arrow)

        # ── 3. Path Extensions (Line Strips) ──
        if other_dir is not None:
            other_line = self._make_line(
                4, frame, stamp,
                other_pose.position, other_dir, length=8.0,
                color=ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.3))
            markers.markers.append(other_line)

        # ── 4. Scenario Label (Text) ──
        text_marker = Marker()
        text_marker.header.frame_id = frame
        text_marker.header.stamp = stamp
        text_marker.ns = 'local_planner'
        text_marker.id = 5
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = self_pose.x
        text_marker.pose.position.y = self_pose.y
        text_marker.pose.position.z = 1.5
        text_marker.scale.z = 0.6

        scenario_labels = {
            Scenario.NONE: '',
            Scenario.STOP: 'S1: STOP',
            Scenario.SPEED_UP: 'S2: SPEED UP',
            Scenario.DODGE: 'S3: DODGE',
            Scenario.WAIT: 'S4: WAIT',
        }
        text_marker.text = scenario_labels.get(scenario, '')
        
        if scenario == Scenario.NONE:
            text_marker.action = Marker.DELETE
        else:
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            
        text_marker.lifetime = Duration(seconds=0, nanoseconds=300_000_000).to_msg()
        markers.markers.append(text_marker)

        # ── 5. Dodge Target (Sphere) ──
        dodge_wp = self.local_planner.dodge_waypoint
        wp_marker = Marker()
        wp_marker.header.frame_id = frame
        wp_marker.header.stamp = stamp
        wp_marker.ns = 'local_planner'
        wp_marker.id = 6
        
        if dodge_wp is not None and scenario == Scenario.DODGE:
            wp_marker.type = Marker.SPHERE
            wp_marker.action = Marker.ADD
            wp_marker.pose.position.x = dodge_wp.x
            wp_marker.pose.position.y = dodge_wp.y
            wp_marker.pose.position.z = 0.3
            wp_marker.scale.x = 0.4
            wp_marker.scale.y = 0.4
            wp_marker.scale.z = 0.4
            wp_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)
            wp_marker.lifetime = Duration(seconds=0, nanoseconds=300_000_000).to_msg()
        else:
            wp_marker.action = Marker.DELETE
            
        markers.markers.append(wp_marker)

        self.marker_pub.publish(markers)

    def _make_arrow(self, marker_id: int, frame: str, stamp,
                    origin: Point, direction: Vector, length: float,
                    color: ColorRGBA) -> Marker:
        """Create an arrow marker for direction visualization."""
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = 'local_planner'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = GeoPoint(x=origin.x, y=origin.y, z=0.3)
        end_pt = origin + direction * length
        end = GeoPoint(x=end_pt.x, y=end_pt.y, z=0.3)
        marker.points = [start, end]

        marker.scale.x = 0.12  # Shaft diameter
        marker.scale.y = 0.25  # Head diameter
        marker.scale.z = 0.3   # Head length
        marker.color = color
        marker.lifetime = Duration(seconds=0, nanoseconds=200_000_000).to_msg()
        return marker

    def _make_line(self, marker_id: int, frame: str, stamp,
                   origin: Point, direction: Vector, length: float,
                   color: ColorRGBA) -> Marker:
        """Create a line strip marker representing trajectory projection."""
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = 'local_planner'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Extend line in both directions
        half = length / 2.0
        unit = direction / max(direction.norm(), 1e-6)
        p1 = origin + unit * (-half)
        p2 = origin + unit * half
        
        marker.points = [
            GeoPoint(x=p1.x, y=p1.y, z=0.2),
            GeoPoint(x=p2.x, y=p2.y, z=0.2),
        ]
        
        marker.scale.x = 0.06  # Line width
        marker.color = color
        marker.lifetime = Duration(seconds=0, nanoseconds=200_000_000).to_msg()
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
