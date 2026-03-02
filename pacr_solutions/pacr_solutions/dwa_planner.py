"""Dynamic Window Approach (DWA) Deliberative Local Planner.

Implements the DWA algorithm for mobile robot motion control.
 Based on the integration of a global path (A*) with
the Dynamic Window local obstacle avoidance approach.

Reference:
    Seder, M. & Petrovic, I. — "Dynamic window based approach to mobile robot
    motion control in the presence of moving obstacles."

The planner samples velocity tuples (v, w) within a dynamic window around the
current velocity, simulates trajectories as circular arcs, predicts the moving
obstacle trajectory, and selects the velocity that maximises a weighted
objective function combining heading alignment, clearance, and velocity.

Key design decisions:
  - Adaptive weights: when the obstacle is nearby, clearance weight is
    automatically boosted to override path following.
  - Conservative collision detection: trajectory points are checked with
    an expanding safety margin that accounts for prediction uncertainty.
  - Danger-zone slowdown: the robot caps its velocity when the obstacle
    is within a proximity threshold, giving it more time to react.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pacr_simulation.geometry2d import Point, Vector, Pose, normalize_angle
from pacr_simulation.map_utils import GridMap


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Trajectory:
    """A candidate DWA trajectory with its evaluation scores."""
    v: float
    w: float
    points: List[Point] = field(default_factory=list)
    admissible: bool = True
    collision_time: float = float('inf')
    min_obstacle_dist: float = float('inf')
    heading_score: float = 0.0
    clearance_score: float = 0.0
    velocity_score: float = 0.0
    total_score: float = 0.0


# ─────────────────────────────────────────────
# Velocity smoother (exponential moving average)
# ─────────────────────────────────────────────

class VelocitySmoother:
    """Smooth noisy velocity estimates with an EMA filter."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._v: float = 0.0
        self._w: float = 0.0
        self._theta: float = 0.0
        self._initialised: bool = False

    def update(self, v: float, w: float, theta: float):
        if not self._initialised:
            self._v = v
            self._w = w
            self._theta = theta
            self._initialised = True
        else:
            self._v = self.alpha * v + (1.0 - self.alpha) * self._v
            self._w = self.alpha * w + (1.0 - self.alpha) * self._w
            self._theta = theta

    @property
    def v(self) -> float:
        return self._v

    @property
    def w(self) -> float:
        return self._w

    @property
    def theta(self) -> float:
        return self._theta

    def reset(self):
        self._initialised = False
        self._v = 0.0
        self._w = 0.0
        self._theta = 0.0


# ─────────────────────────────────────────────
# DWA Planner
# ─────────────────────────────────────────────

class DWAPlanner:
    """Dynamic Window Approach deliberative local planner.

    The DWA acts as a LOCAL planner only —  Its job is:
      1. Follow the A* path (heading toward the lookahead point)
      2. Avoid obstacles by rejecting dangerous trajectories
      3. Prefer faster progress when safe

    When the moving obstacle is nearby, the planner automatically
    shifts to a cautious mode:
      - Clearance weight is boosted (up to 2×)
      - Heading weight is reduced
      - Maximum velocity is capped proportionally to distance
    """

    # Distance thresholds for adaptive behavior
    DANGER_DIST = 3.5      # Start adapting weights at this distance
    CRITICAL_DIST = 1.2    # Maximum caution at this distance
    SLOWDOWN_DIST = 1.0   # Start slowing down at this distance

    def __init__(self,
                 v_max: float = 0.5,
                 w_max: float = 1.0,
                 v_acc_max: float = 0.5,
                 w_acc_max: float = 2.0,
                 dt: float = 0.1,
                 prediction_horizon: float = 3.0,
                 v_samples: int = 13,
                 w_samples: int = 31,
                 heading_weight: float = 0.45,
                 clearance_weight: float = 0.35,
                 velocity_weight: float = 0.20,
                 goal_tolerance: float = 0.15,
                 safety_margin: float = 0.50,
                 robot_radius: float = 0.30,
                 lookahead_distance: float = 0.8):
        # Kinematic limits
        self.v_max = v_max
        self.w_max = w_max
        self.v_acc_max = v_acc_max
        self.w_acc_max = w_acc_max

        # Trajectory simulation
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.n_steps = max(1, int(prediction_horizon / dt))

        # Sampling resolution
        self.v_samples = v_samples
        self.w_samples = w_samples

        # Base objective weights (adapted dynamically)
        self.base_heading_weight = heading_weight
        self.base_clearance_weight = clearance_weight
        self.base_velocity_weight = velocity_weight

        # Thresholds
        self.goal_tolerance = goal_tolerance
        self.safety_margin = safety_margin
        self.robot_radius = robot_radius
        self.lookahead_distance = lookahead_distance

        # Collision check distance = robot_radius + safety_margin
        self._collision_dist = self.robot_radius + self.safety_margin

    # ─────────────────────────────────────
    # Public API
    # ─────────────────────────────────────

    def compute_command(self,
                        robot_pose: Pose,
                        robot_v: float,
                        robot_w: float,
                        path: List[Point],
                        goal: Pose,
                        grid_map: Optional[GridMap],
                        other_pos: Optional[Point] = None,
                        other_v: float = 0.0,
                        other_w: float = 0.0,
                        other_theta: float = 0.0
                        ) -> Tuple[float, float, bool, List[Trajectory]]:
        """Compute the best velocity command using DWA."""
        if not path:
            return 0.0, 0.0, True, []

        # ── Goal check ──
        dist_to_goal = robot_pose.position.dist(goal.position)
        if dist_to_goal < self.goal_tolerance:
            angle_err = normalize_angle(goal.theta - robot_pose.theta)
            if abs(angle_err) < 0.05:
                return 0.0, 0.0, True, []
            w_cmd = max(-self.w_max, min(self.w_max, 1.5 * angle_err))
            return 0.0, w_cmd, False, []

        # ── Obstacle distance and adaptive behavior ──
        obstacle_dist = float('inf')
        if other_pos is not None:
            obstacle_dist = robot_pose.position.dist(other_pos)

        # Adaptive weights: boost clearance when obstacle is near
        hw, cw, vw = self._adaptive_weights(obstacle_dist)

        # Adaptive speed cap: slow down near obstacle
        v_cap = self._adaptive_speed_cap(obstacle_dist)

        # ── Find lookahead point on the A* path ──
        lookahead_pt = self._get_lookahead_point(robot_pose, path)

        # ── Predict moving obstacle trajectory ──
        obstacle_traj = self._predict_obstacle_trajectory(
            other_pos, other_v, other_w, other_theta
        ) if other_pos is not None else []

        # ── Build dynamic window ──
        effective_v_max = min(self.v_max, v_cap)
        v_min_dw = max(0.0, robot_v - self.v_acc_max * self.dt)
        v_max_dw = min(effective_v_max, robot_v + self.v_acc_max * self.dt)
        w_min_dw = max(-self.w_max, robot_w - self.w_acc_max * self.dt)
        w_max_dw = min(self.w_max, robot_w + self.w_acc_max * self.dt)

        # ── Sample velocities ──
        if self.v_samples > 1:
            v_step = (v_max_dw - v_min_dw) / (self.v_samples - 1)
        else:
            v_step = 0.0
        if self.w_samples > 1:
            w_step = (w_max_dw - w_min_dw) / (self.w_samples - 1)
        else:
            w_step = 0.0

        trajectories: List[Trajectory] = []

        for i in range(self.v_samples):
            v_sample = v_min_dw + i * v_step
            for j in range(self.w_samples):
                w_sample = w_min_dw + j * w_step

                traj = Trajectory(v=v_sample, w=w_sample)

                # Simulate trajectory (with headings for better checking)
                traj.points, headings = self._simulate_trajectory(
                    robot_pose, v_sample, w_sample)

                # ── Static obstacle collisions ──
                t_col_static = self._check_static_collisions(
                    traj.points, grid_map)

                # ── Moving obstacle: time-indexed collision ──
                t_col_moving = float('inf')
                min_obs_dist = float('inf')

                if other_pos is not None:
                    t_col_moving, min_obs_dist = self._check_moving_collisions(
                        traj.points, obstacle_traj, other_pos)

                traj.collision_time = min(t_col_static, t_col_moving)
                traj.min_obstacle_dist = min_obs_dist

                # ── Admissibility: can the robot brake before collision? ──
                brake_time = abs(v_sample) / self.v_acc_max if self.v_acc_max > 0 else 0.0
                traj.admissible = traj.collision_time > brake_time * 1.5

                if not traj.admissible:
                    traj.total_score = -1.0
                    trajectories.append(traj)
                    continue

                # ── Evaluate objective function ──
                traj.heading_score = self._heading_objective(
                    traj.points, robot_pose, lookahead_pt)
                traj.clearance_score = self._clearance_objective(
                    v_sample, traj, obstacle_dist)
                traj.velocity_score = v_sample / self.v_max if self.v_max > 0 else 0.0

                traj.total_score = (
                    hw * traj.heading_score +
                    cw * traj.clearance_score +
                    vw * traj.velocity_score
                )
                trajectories.append(traj)

        # ── Select best trajectory ──
        best = None
        best_score = -float('inf')
        for traj in trajectories:
            if traj.admissible and traj.total_score > best_score:
                best_score = traj.total_score
                best = traj

        if best is None:
            return 0.0, 0.0, False, trajectories

        # Slow down near goal
        best_v = best.v
        if dist_to_goal < 0.5:
            best_v = min(best_v, dist_to_goal * 0.8)

        return best_v, best.w, False, trajectories

    # ─────────────────────────────────────
    # Adaptive behavior
    # ─────────────────────────────────────

    def _adaptive_weights(self, obstacle_dist: float
                          ) -> Tuple[float, float, float]:
        """Dynamically adjust objective weights based on obstacle proximity.

        When the obstacle is far away: use base weights (prioritise path).
        When the obstacle is near: boost clearance, reduce heading/velocity.

        Returns (heading_weight, clearance_weight, velocity_weight).
        """
        if obstacle_dist >= self.DANGER_DIST:
            return (self.base_heading_weight,
                    self.base_clearance_weight,
                    self.base_velocity_weight)

        # Linear interpolation between DANGER_DIST and CRITICAL_DIST
        t = max(0.0, min(1.0,
                         (self.DANGER_DIST - obstacle_dist) /
                         (self.DANGER_DIST - self.CRITICAL_DIST)))

        # At t=1 (critical): clearance dominates
        hw = self.base_heading_weight * (1.0 - 0.6 * t)   # 0.45 → 0.18
        cw = self.base_clearance_weight + 0.45 * t         # 0.35 → 0.80
        vw = self.base_velocity_weight * (1.0 - 0.8 * t)   # 0.20 → 0.04

        # Normalise so they sum to 1
        total = hw + cw + vw
        return hw / total, cw / total, vw / total

    def _adaptive_speed_cap(self, obstacle_dist: float) -> float:
        """Cap the maximum velocity when the obstacle is nearby.

        This gives the robot more time to react and makes braking
        distances shorter.
        """
        if obstacle_dist >= self.SLOWDOWN_DIST:
            return self.v_max

        # Linear: at CRITICAL_DIST → 30% of v_max
        t = max(0.0, min(1.0,
                         (self.SLOWDOWN_DIST - obstacle_dist) /
                         (self.SLOWDOWN_DIST - self.CRITICAL_DIST)))

        return self.v_max * (1.0 - 0.7 * t)

    # ─────────────────────────────────────
    # Trajectory simulation
    # ─────────────────────────────────────

    def _simulate_trajectory(self, pose: Pose, v: float,
                             w: float) -> Tuple[List[Point], List[float]]:
        """Simulate a trajectory as a circular arc from the current pose.

        Returns (points, headings) — both lists have n_steps entries.
        """
        points = []
        headings = []
        x, y, theta = pose.x, pose.y, pose.theta

        for _ in range(self.n_steps):
            if abs(w) < 1e-6:
                x += v * math.cos(theta) * self.dt
                y += v * math.sin(theta) * self.dt
            else:
                x += (v / w) * (math.sin(theta + w * self.dt) - math.sin(theta))
                y -= (v / w) * (math.cos(theta + w * self.dt) - math.cos(theta))
                theta = normalize_angle(theta + w * self.dt)
            points.append(Point(x, y))
            headings.append(theta)

        return points, headings

    def _predict_obstacle_trajectory(self, pos: Point, v: float,
                                     w: float, theta: float
                                     ) -> List[Point]:
        """Predict the moving obstacle's trajectory as a circular arc."""
        points = []
        x, y = pos.x, pos.y
        th = theta

        for _ in range(self.n_steps):
            if abs(w) < 1e-6:
                x += v * math.cos(th) * self.dt
                y += v * math.sin(th) * self.dt
            else:
                x += (v / w) * (math.sin(th + w * self.dt) - math.sin(th))
                y -= (v / w) * (math.cos(th + w * self.dt) - math.cos(th))
                th = normalize_angle(th + w * self.dt)
            points.append(Point(x, y))

        return points

    # ─────────────────────────────────────
    # Collision checks
    # ─────────────────────────────────────

    def _check_static_collisions(self, traj_points: List[Point],
                                 grid_map: Optional[GridMap]
                                 ) -> float:
        """Return time until collision with static obstacles."""
        if grid_map is None:
            return float('inf')

        for i, pt in enumerate(traj_points):
            try:
                val = grid_map.get(pt)
                if val >= 50:
                    return (i + 1) * self.dt
            except (IndexError, ValueError):
                return (i + 1) * self.dt

        return float('inf')

    def _check_moving_collisions(self, robot_traj: List[Point],
                                 obstacle_traj: List[Point],
                                 obstacle_current_pos: Point
                                 ) -> Tuple[float, float]:
        """Check time-indexed collisions between robot and obstacle.
        
        Pairs the robot's predicted position at time 't' with the 
        obstacle's predicted position at time 't'.
        
        Returns (collision_time, min_distance_to_obstacle).
        """
        min_dist = float('inf')

        # Fallback: If the obstacle isn't moving (no trajectory), treat it as static
        if not obstacle_traj:
            for i, rpt in enumerate(robot_traj):
                d = rpt.dist(obstacle_current_pos)
                if d < min_dist:
                    min_dist = d
                if d < self._collision_dist:
                    return (i + 1) * self.dt, min_dist
            return float('inf'), min_dist

        # ── Time-indexed check against predicted trajectory ──
        n = min(len(robot_traj), len(obstacle_traj))
        for i in range(n):
            # Expanding collision radius: +20% per second of prediction
            uncertainty_factor = 1.0 + 0.20 * (i * self.dt)
            check_dist = self._collision_dist * uncertainty_factor

            # Compare robot position at time i vs obstacle position at time i
            d = robot_traj[i].dist(obstacle_traj[i])
            if d < min_dist:
                min_dist = d
            if d < check_dist:
                return (i + 1) * self.dt, min_dist

            # Midpoint check (catches fast-moving obstacles skipping over points)
            if i > 0:
                mid_robot = Point(
                    (robot_traj[i - 1].x + robot_traj[i].x) / 2.0,
                    (robot_traj[i - 1].y + robot_traj[i].y) / 2.0)
                mid_obs = Point(
                    (obstacle_traj[i - 1].x + obstacle_traj[i].x) / 2.0,
                    (obstacle_traj[i - 1].y + obstacle_traj[i].y) / 2.0)
                d_mid = mid_robot.dist(mid_obs)
                if d_mid < min_dist:
                    min_dist = d_mid
                if d_mid < check_dist:
                    return ((i + 0.5)) * self.dt, min_dist

        return float('inf'), min_dist

    # ─────────────────────────────────────
    # Objective function components
    # ─────────────────────────────────────

    def _heading_objective(self, traj_points: List[Point],
                           robot_pose: Pose,
                           lookahead_pt: Point) -> float:
        """Heading alignment objective — steers toward the A* lookahead.

        Two components:
          1. Direction: does the trajectory head toward the lookahead?
          2. Progress: does the endpoint get closer to the lookahead?
        """
        if not traj_points:
            return 0.0

        goal_vec = lookahead_pt - robot_pose.position
        goal_dist = goal_vec.norm()

        if goal_dist < 1e-6:
            return 1.0

        # Direction alignment
        goal_angle = math.atan2(goal_vec.y, goal_vec.x)

        if len(traj_points) >= 2:
            traj_vec = traj_points[-1] - robot_pose.position
            traj_dist = traj_vec.norm()
            if traj_dist > 1e-6:
                traj_angle = math.atan2(traj_vec.y, traj_vec.x)
                angle_diff = abs(normalize_angle(traj_angle - goal_angle))
                direction_score = 1.0 - (angle_diff / math.pi)
            else:
                heading_diff = abs(normalize_angle(robot_pose.theta - goal_angle))
                direction_score = 1.0 - (heading_diff / math.pi)
        else:
            direction_score = 0.5

        # Distance progress
        endpoint = traj_points[-1]
        end_dist = endpoint.dist(lookahead_pt)
        progress_score = max(0.0, 1.0 - end_dist / max(goal_dist * 2.0, 1.0))

        return 0.7 * direction_score + 0.3 * progress_score

    def _clearance_objective(self, v: float, traj: Trajectory,
                             obstacle_dist: float) -> float:
        """Clearance objective — rewards staying far from obstacles.

        Combines:
          - Time-based clearance 
          - Proximity penalty (penalises passing near the obstacle)
          - Min-distance bonus (rewards trajectories with maximum clearance)
        """
        # Time-based clearance
        t_brake = abs(v) / self.v_acc_max if self.v_acc_max > 0 else 0.0
        t_max = self.prediction_horizon
        t_col = traj.collision_time

        if t_col <= t_brake:
            time_score = 0.0
        elif t_col >= t_max:
            time_score = 1.0
        else:
            time_score = (t_col - t_brake) / (t_max - t_brake)

        # Proximity penalty
        prox_score = 1.0
        if traj.min_obstacle_dist < float('inf'):
            safe_dist = self._collision_dist * 4.0
            if traj.min_obstacle_dist < safe_dist:
                prox_score = traj.min_obstacle_dist / safe_dist
                prox_score = max(0.0, prox_score)

        # When obstacle is nearby, proximity matters more
        if obstacle_dist < self.DANGER_DIST:
            return time_score * 0.4 + prox_score * 0.6
        else:
            return time_score * 0.7 + prox_score * 0.3

    # ─────────────────────────────────────
    # Path utilities
    # ─────────────────────────────────────

    def _get_lookahead_point(self, robot_pose: Pose,
                             path: List[Point]) -> Point:
        """Find a lookahead point on the A* path.

        Simple and robust: find closest path point, walk forward
        by lookahead_distance.
        """
        if not path:
            return robot_pose.position

        # Find closest path point
        closest_idx = 0
        min_dist = float('inf')
        for i, pt in enumerate(path):
            d = robot_pose.position.dist(pt)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Walk forward on the path
        for i in range(closest_idx, len(path)):
            if robot_pose.position.dist(path[i]) >= self.lookahead_distance:
                return path[i]

        return path[-1]

    @staticmethod
    def find_closest_path_index(pos: Point, path: List[Point]) -> int:
        """Find index of the closest point on the path."""
        min_dist = float('inf')
        idx = 0
        for i, pt in enumerate(path):
            d = pos.dist(pt)
            if d < min_dist:
                min_dist = d
                idx = i
        return idx
