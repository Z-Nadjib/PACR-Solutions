"""ROS2 Node for Path Following with DWA Deliberative Local Planning.

This node provides the `get_cmd` service using the Dynamic Window Approach
(DWA) for deliberative obstacle avoidance. It tracks the concurrent robot
via TF, estimates its velocity, predicts its trajectory, and applies the
DWA algorithm to select the best velocity command while following the
global A* path.

A separate DWAPlanner class is used (see dwa_planner.py) because the DWA
algorithm  — this makes it testable in isolation and keeps this ROS node focused on I/O and
orchestration.

Topics subscribed:
    - costmap (nav_msgs/OccupancyGrid): Static occupancy grid.

Topics published:
    - dwa_planner/markers (visualization_msgs/MarkerArray): DWA trajectories
      and obstacle predictions visualised for RViz.

Services provided:
    - get_cmd (pacr_interfaces/srv/GetCmd): Velocity command service.

TF lookups:
    - world → concurrent_robot/base_link: Moving obstacle pose.

Parameters:
    - v_max, w_max: Velocity limits [m/s, rad/s].
    - v_acc_max, w_acc_max: Acceleration limits [m/s², rad/s²].
    - prediction_horizon: Trajectory look-ahead [s].
    - dt: Simulation time step [s].
    - v_samples, w_samples: Velocity sample counts.
    - heading_weight, clearance_weight, velocity_weight: Objective weights.
    - goal_tolerance: Goal arrival threshold [m].
    - safety_margin: Obstacle safety distance [m].
    - lookahead_distance: Path lookahead for heading target [m].
    - static_frame_id, other_frame_id: TF frames.
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

from pacr_simulation.geometry2d import Pose, Point, Vector, Transform, normalize_angle
from pacr_simulation.map_utils import GridMap
from pacr_solutions.dwa_planner import DWAPlanner, Trajectory, VelocitySmoother


# ─────────────────────────────────────────────
# DWA Path Following Node
# ─────────────────────────────────────────────

class DWAPathFollowingNode(Node):
    """ROS2 Node for path following using the DWA deliberative planner.

    Provides the get_cmd service expected by the test_goto executive node.
    Visualises all candidate DWA trajectories in RViz for debugging.
    """

    def __init__(self):
        super().__init__('dwa_path_following')

        # ── Declare parameters ──
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('v_acc_max', 0.5)
        self.declare_parameter('w_acc_max', 2.0)
        self.declare_parameter('prediction_horizon', 3.0)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_samples', 13)
        self.declare_parameter('w_samples', 31)
        self.declare_parameter('heading_weight', 0.45)
        self.declare_parameter('clearance_weight', 0.35)
        self.declare_parameter('velocity_weight', 0.20)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('safety_margin', 0.50)
        self.declare_parameter('lookahead_distance', 0.8)
        self.declare_parameter('static_frame_id', 'world')
        self.declare_parameter('other_frame_id', 'concurrent_robot/base_link')

        # ── Read parameters ──
        v_max = self._p('v_max')
        w_max = self._p('w_max')
        v_acc_max = self._p('v_acc_max')
        w_acc_max = self._p('w_acc_max')
        prediction_horizon = self._p('prediction_horizon')
        dt = self._p('dt')
        v_samples = self.get_parameter('v_samples').get_parameter_value().integer_value
        w_samples = self.get_parameter('w_samples').get_parameter_value().integer_value
        heading_weight = self._p('heading_weight')
        clearance_weight = self._p('clearance_weight')
        velocity_weight = self._p('velocity_weight')
        goal_tolerance = self._p('goal_tolerance')
        safety_margin = self._p('safety_margin')
        lookahead_distance = self._p('lookahead_distance')
        self.static_frame_id = self.get_parameter(
            'static_frame_id').get_parameter_value().string_value
        self.other_frame_id = self.get_parameter(
            'other_frame_id').get_parameter_value().string_value

        # Store limits for velocity clamping
        self.v_max = v_max
        self.w_max = w_max

        # ── DWA Planner ──
        self.dwa = DWAPlanner(
            v_max=v_max,
            w_max=w_max,
            v_acc_max=v_acc_max,
            w_acc_max=w_acc_max,
            dt=dt,
            prediction_horizon=prediction_horizon,
            v_samples=v_samples,
            w_samples=w_samples,
            heading_weight=heading_weight,
            clearance_weight=clearance_weight,
            velocity_weight=velocity_weight,
            goal_tolerance=goal_tolerance,
            safety_margin=safety_margin,
            lookahead_distance=lookahead_distance,
        )

        # ── TF infrastructure for tracking the concurrent robot ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Previous obstacle pose for velocity estimation
        self._prev_other_pose: Pose = None
        self._prev_other_time: Time = None
        # Smoothed obstacle velocity estimator
        self._obstacle_vel_smoother = VelocitySmoother(alpha=0.3)
        # Cached obstacle velocity for visualization
        self._cached_other_v: float = 0.0
        self._cached_other_w: float = 0.0
        self._cached_other_theta: float = 0.0

        # ── Map awareness ──
        self.grid_map = None
        latching_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, 'costmap', self._costmap_cb,
            qos_profile=latching_qos)

        # ── Visualisation ──
        self.marker_pub = self.create_publisher(
            MarkerArray, 'dwa_planner/markers', 1)

        # ── Service interface ──
        self.srv = self.create_service(GetCmd, 'get_cmd', self.get_cmd_cb)

        self.get_logger().info(
            f'DWAPathFollowingNode initialized '
            f'(v_max={v_max}, w_max={w_max}, '
            f'samples={v_samples}x{w_samples}, '
            f'horizon={prediction_horizon}s, '
            f'lookahead={lookahead_distance}m).')

    def _p(self, name: str) -> float:
        """Shorthand to read a double parameter."""
        return self.get_parameter(name).get_parameter_value().double_value

    # ─────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────

    def _costmap_cb(self, msg: OccupancyGrid):
        """Buffer the incoming costmap for spatial reasoning."""
        self.grid_map = GridMap.from_msg(msg)
        self.get_logger().info('Costmap received for DWA planner.')

    def _get_other_robot_state(self) -> tuple:
        """Retrieve the concurrent robot's pose and estimate its velocity.

        Returns:
            (pose, v_linear, v_angular, heading) or (None, 0, 0, 0)
            if the transform cannot be looked up.
        """
        try:
            transform_msg = self.tf_buffer.lookup_transform(
                self.static_frame_id,
                self.other_frame_id,
                Time(),
            )
            pose = Pose.from_transform(
                Transform.from_msg(transform_msg.transform))
            now = Time.from_msg(transform_msg.header.stamp)

            # Estimate velocity from consecutive poses
            raw_v = 0.0
            raw_w = 0.0

            if self._prev_other_pose is not None and self._prev_other_time is not None:
                dt_ns = (now - self._prev_other_time).nanoseconds
                if dt_ns > 0:
                    dt_s = dt_ns / 1e9
                    dx = pose.x - self._prev_other_pose.x
                    dy = pose.y - self._prev_other_pose.y
                    dist = math.hypot(dx, dy)
                    # Determine sign from heading
                    heading_vec = Vector(
                        math.cos(pose.theta), math.sin(pose.theta))
                    move_vec = Vector(dx, dy)
                    sign = 1.0 if heading_vec.dot(move_vec) >= 0 else -1.0
                    raw_v = sign * dist / dt_s
                    dtheta = normalize_angle(
                        pose.theta - self._prev_other_pose.theta)
                    raw_w = dtheta / dt_s

            # Smooth the velocity estimate
            self._obstacle_vel_smoother.update(raw_v, raw_w, pose.theta)
            other_v = self._obstacle_vel_smoother.v
            other_w = self._obstacle_vel_smoother.w

            self._prev_other_pose = pose
            self._prev_other_time = now
            # Cache for visualization
            self._cached_other_v = other_v
            self._cached_other_w = other_w
            self._cached_other_theta = pose.theta

            return pose, other_v, other_w, pose.theta

        except TransformException:
            return None, 0.0, 0.0, 0.0

    def get_cmd_cb(self, request, response):
        """Main control service callback: compute velocity via DWA.

        Reads robot state from the request, queries TF for the moving
        obstacle, runs the DWA algorithm, and returns the best command.
        """
        robot_pose = Pose.from_msg(request.pose)
        goal_pose = Pose.from_msg(request.goal)
        path = [Point.from_msg(pt) for pt in request.path]

        # Current robot velocity from the request
        robot_v = request.velocity.linear
        robot_w = request.velocity.angular

        # ── Get moving obstacle state ──
        other_pose, other_v, other_w, other_theta = \
            self._get_other_robot_state()

        other_pos = other_pose.position if other_pose is not None else None

        # ── Run DWA ──
        v, w, finished, trajectories = self.dwa.compute_command(
            robot_pose=robot_pose,
            robot_v=robot_v,
            robot_w=robot_w,
            path=path,
            goal=goal_pose,
            grid_map=self.grid_map,
            other_pos=other_pos,
            other_v=other_v,
            other_w=other_w,
            other_theta=other_theta,
        )

        # Clamp to allowed velocities
        v = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))

        response.command = Velocity2D(linear=v, angular=w)
        response.finished = finished

        # ── Visualise ──
        self._publish_markers(
            robot_pose, other_pose, trajectories,
            best_v=v, best_w=w, path=path, finished=finished)

        if not finished:
            self.get_logger().debug(
                f'[DWA] v={v:.3f}, w={w:.3f}, '
                f'trajs={len(trajectories)}, '
                f'admissible={sum(1 for t in trajectories if t.admissible)}')

        return response

    # ─────────────────────────────────────
    # RViz visualisation
    # ─────────────────────────────────────

    def _publish_markers(self, robot_pose: Pose,
                         other_pose, trajectories: list,
                         best_v: float, best_w: float,
                         path: list, finished: bool):
        """Publish DWA trajectory and state markers to RViz."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame = self.static_frame_id
        lifetime = Duration(seconds=0, nanoseconds=150_000_000).to_msg()

        marker_id = 0

        # ── 1. Delete all previous markers ──
        delete_marker = Marker()
        delete_marker.header.frame_id = frame
        delete_marker.header.stamp = stamp
        delete_marker.ns = 'dwa'
        delete_marker.action = Marker.DELETEALL
        delete_marker.id = marker_id
        markers.markers.append(delete_marker)
        marker_id += 1

        if not trajectories:
            self.marker_pub.publish(markers)
            return

        # ── 2. Find the best trajectory for highlighting ──
        best_traj = None
        best_score = -float('inf')
        for traj in trajectories:
            if traj.admissible and traj.total_score > best_score:
                best_score = traj.total_score
                best_traj = traj

        # ── 3. Candidate trajectories (sampled subset) ──
        step = max(1, len(trajectories) // 40)
        for idx in range(0, len(trajectories), step):
            traj = trajectories[idx]
            if len(traj.points) < 2:
                continue

            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'dwa'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.02
            m.lifetime = lifetime

            if not traj.admissible:
                m.color = ColorRGBA(r=0.8, g=0.1, b=0.1, a=0.25)
            elif traj is best_traj:
                continue  # Draw best separately
            else:
                t = max(0.0, min(1.0, traj.total_score))
                m.color = ColorRGBA(
                    r=0.2 * (1.0 - t),
                    g=0.3 + 0.7 * t,
                    b=0.8 * (1.0 - t),
                    a=0.35)

            m.points = [GeoPoint(x=pt.x, y=pt.y, z=0.05)
                        for pt in traj.points]
            markers.markers.append(m)

        # ── 4. Best trajectory (thick green line) ──
        if best_traj is not None and len(best_traj.points) >= 2:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'dwa'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.08
            m.lifetime = lifetime
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.95)
            m.points = [GeoPoint(x=pt.x, y=pt.y, z=0.1)
                        for pt in best_traj.points]
            markers.markers.append(m)

        # ── 5. Moving obstacle predicted trajectory (orange) ──
        if other_pose is not None:
            obs_traj = self.dwa._predict_obstacle_trajectory(
                other_pose.position,
                self._cached_other_v,
                self._cached_other_w,
                self._cached_other_theta)
            if len(obs_traj) >= 2:
                m = Marker()
                m.header.frame_id = frame
                m.header.stamp = stamp
                m.ns = 'dwa'
                m.id = marker_id
                marker_id += 1
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.06
                m.lifetime = lifetime
                m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)
                m.points = [GeoPoint(x=pt.x, y=pt.y, z=0.1)
                            for pt in obs_traj]
                markers.markers.append(m)

        # ── 6. Lookahead point on path (sphere) ──
        if path:
            la_pt = self.dwa._get_lookahead_point(robot_pose, path)
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'dwa'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = la_pt.x
            m.pose.position.y = la_pt.y
            m.pose.position.z = 0.3
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
            m.lifetime = lifetime
            markers.markers.append(m)

        # ── 7. Status text ──
        text = Marker()
        text.header.frame_id = frame
        text.header.stamp = stamp
        text.ns = 'dwa'
        text.id = marker_id
        marker_id += 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = robot_pose.x
        text.pose.position.y = robot_pose.y
        text.pose.position.z = 1.5
        text.scale.z = 0.4
        text.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        text.lifetime = lifetime

        if finished:
            text.text = 'GOAL REACHED'
        elif best_traj is None:
            text.text = 'DWA: EMERGENCY STOP'
        else:
            n_adm = sum(1 for t in trajectories if t.admissible)
            text.text = f'DWA: v={best_v:.2f} w={best_w:.2f} [{n_adm}/{len(trajectories)}]'

        markers.markers.append(text)

        self.marker_pub.publish(markers)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DWAPathFollowingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
