"""Behavior-Based Local Planner for Collision Avoidance.

Coordinate-based planner using TF positions of the other robot.Detects collision
scenarios and overrides the Pure Pursuit controller(A*) when needed.

Scenario 1 — "Let him pass" (STOP):
    Other robot crosses our path but is NOT targeting us.
    Action: v=0, w=0 until it clears our safety radius.

Scenario 2 — "Run" (SPEED UP):
    Other robot IS targeting us, but we are NOT targeting it.
    Action: boost velocity along the existing path.

Scenario 3 — "Give way" (LATERAL DODGE):
    Head-on collision — both robots targeting each other on
    the same trajectory line.
    Action: steer to a lateral escape point, then resume path.

Scenario 4 — "Wait behind" (WAIT):
    We are approaching the other robot, but it is not moving (itermediate points).
    Action: stop to maintain safety distance.
"""

import math
from collections import deque
from enum import Enum
from typing import List, Optional, Tuple

from pacr_simulation.geometry2d import Point, Vector, Pose, normalize_angle
from pacr_simulation.map_utils import GridMap


# ─────────────────────────────────────────────
# Scenario enumeration
# ─────────────────────────────────────────────
class Scenario(Enum):
    """Active collision-avoidance scenario."""
    NONE = 0
    STOP = 1        # Scenario 1: let the other robot pass
    SPEED_UP = 2    # Scenario 2: accelerate away
    DODGE = 3       # Scenario 3: lateral detour
    WAIT = 4        # Scenario 4: wait behind (stationary/blocking)


# ─────────────────────────────────────────────
# Direction Estimator (sliding window)
# ─────────────────────────────────────────────
class DirectionEstimator:
    """Estimates a robot's direction from a sliding window of positions.

    Maintains a short history buffer of N positions. The direction
    vector is computed from the oldest stored position to the newest,
    giving a stable, noise-resilient heading estimate.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._history: deque[Point] = deque(maxlen=window_size)

    def update(self, position: Point) -> None:
        """Record a new position sample."""
        self._history.append(position)

    def get_direction(self) -> Optional[Vector]:
        """Return the direction vector, or None if not enough data."""
        if len(self._history) < 2:
            return None
        oldest = self._history[0]
        newest = self._history[-1]
        diff = newest - oldest
        if diff.norm() < 1e-6:
            return None  # robot is stationary
        return diff

    def get_unit_direction(self) -> Optional[Vector]:
        """Return the normalized direction vector."""
        d = self.get_direction()
        if d is None:
            return None
        n = d.norm()
        if n < 1e-6:
            return None
        return d / n

    def reset(self) -> None:
        """Clear the history buffer."""
        self._history.clear()


# ─────────────────────────────────────────────
# Geometric helper functions
# ─────────────────────────────────────────────
def is_pointing_at(pos_a: Point, dir_a: Vector,
                   pos_b: Point, threshold_rad: float) -> bool:
    """Check if direction from A points toward B within threshold.

    True if the angle between dir_a and the vector A→B is less
    than threshold_rad.
    """
    vec_ab = pos_b - pos_a
    dist = vec_ab.norm()
    if dist < 1e-6:
        return True  # same position
    dir_norm = dir_a.norm()
    if dir_norm < 1e-6:
        return False
    # Angle between dir_a and vec_ab
    cos_angle = dir_a.dot(vec_ab) / (dir_norm * dist)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for acos
    angle = math.acos(cos_angle)
    return angle < threshold_rad


# ─────────────────────────────────────────────
# Local Planner
# ─────────────────────────────────────────────
class LocalPlanner:
    """Behavior-based local planner for collision avoidance.

    Uses coordinate-based tracking of the other robot to detect
    collision scenarios and compute avoidance commands.
    """

    def __init__(self,
                 safety_radius: float = 2.5,
                 pointing_angle_deg: float = 15.0,
                 escape_offset: float = 1.0,
                 v_boost_factor: float = 1.5,
                 v_max: float = 0.5,
                 w_max: float = 1.0):
        self.safety_radius = safety_radius
        self.pointing_threshold = math.radians(pointing_angle_deg)
        self.escape_offset = escape_offset
        self.v_boost_factor = v_boost_factor
        self.v_max = v_max
        self.w_max = w_max

        # Direction estimators
        self.self_estimator = DirectionEstimator(window_size=5)
        self.other_estimator = DirectionEstimator(window_size=5)

        # State for Scenario 3 dodge
        self._dodge_waypoint: Optional[Point] = None
        self._active_scenario = Scenario.NONE

    def update(self, self_pos: Point, other_pos: Point) -> None:
        """Feed new positions to the direction estimators."""
        self.self_estimator.update(self_pos)
        self.other_estimator.update(other_pos)

    def evaluate(self,
                 self_pose: Pose,
                 other_pos: Point,
                 path: list,
                 grid_map: Optional[GridMap] = None
                 ) -> Tuple[Scenario, Optional[Tuple[float, float]]]:
        """Evaluate collision threat and return avoidance command.

        Returns:
            (scenario, command) where command is (v, w) or None
            if no avoidance is needed (Pure Pursuit should run).
        """
        self_pos = self_pose.position
        distance = self_pos.dist(other_pos)

        # ── Outside safety radius: no threat ──
        if distance > self.safety_radius:
            self._active_scenario = Scenario.NONE
            self._dodge_waypoint = None
            return (Scenario.NONE, None)

        # ── Get other robot's direction (buffer-based ) ──
        other_dir = self.other_estimator.get_direction()

        # ── Path-based checks for OUR robot ──
        other_on_our_path = self._is_other_on_our_path_ahead(
            self_pos, other_pos, path)

        # ─ Commit to the Dodge ──
        # If we are already dodging, keep doing it until we reach the waypoint
        if self._active_scenario == Scenario.DODGE and self._dodge_waypoint is not None:
            if self_pos.dist(self._dodge_waypoint) < 0.3:
                # We reached the dodge point! Clear it to re-evaluate next tick.
                self._dodge_waypoint = None
                self._active_scenario = Scenario.NONE
            else:
                # We haven't reached it yet, keep steering towards it.
                path_dir = self._get_path_direction(self_pose, path)
                if path_dir is None:
                    path_dir = Vector(math.cos(self_pose.theta),
                                     math.sin(self_pose.theta))
                return (Scenario.DODGE,
                        self._compute_dodge(self_pose, other_pos,
                                            path_dir, grid_map))

        # ── Handle Stationary Target (Scenario 4 / Wait) ──
        if other_dir is None:
            if other_on_our_path:
                self._active_scenario = Scenario.WAIT
                return (Scenario.WAIT, (0.0, 0.0))
            else:
                self._active_scenario = Scenario.NONE
                return (Scenario.NONE, None)

        # ── Other robot's direction-based checks ──
        other_points_at_us = is_pointing_at(
            other_pos, other_dir, self_pos, self.pointing_threshold)
        other_traj_crosses = self._other_trajectory_crosses_our_path(
            self_pos, path, other_pos, other_dir)

        # ── Scenario 3: Head-on collision ──
        # Other is on our A* path AND pointing at us → head-on, must dodge
        if other_points_at_us and other_on_our_path:
            self._active_scenario = Scenario.DODGE
            path_dir = self._get_path_direction(self_pose, path)
            if path_dir is None:
                path_dir = Vector(math.cos(self_pose.theta),
                                 math.sin(self_pose.theta))
            return (Scenario.DODGE,
                    self._compute_dodge(self_pose, other_pos,
                                        path_dir, grid_map))

        # ── Scenario 2: Being charged at (speed up) ──
        # Other points at us but is NOT on our path → we don't collide, speed away
        if other_points_at_us and not other_on_our_path:
            self._active_scenario = Scenario.SPEED_UP
            return (Scenario.SPEED_UP,
                    self._compute_speed_up(self_pose, path))

        # ── Scenario 1: Crossing paths (stop) ──
        # Other's trajectory crosses our A* path but is not targeting us
        if not other_points_at_us and other_traj_crosses:
            self._active_scenario = Scenario.STOP
            return (Scenario.STOP, (0.0, 0.0))

        # ── Scenario 4: Wait Behind (other blocking our path) ──
        # Other is on our A* path but not pointing at us → wait for them to clear
        if other_on_our_path and not other_points_at_us:
            self._active_scenario = Scenario.WAIT
            return (Scenario.WAIT, (0.0, 0.0))

        # ── No scenario matched — no override ──
        self._active_scenario = Scenario.NONE
        self._dodge_waypoint = None
        return (Scenario.NONE, None)

    @property
    def active_scenario(self) -> Scenario:
        """Currently active avoidance scenario."""
        return self._active_scenario

    @property
    def dodge_waypoint(self) -> Optional[Point]:
        """Current lateral escape waypoint."""
        return self._dodge_waypoint

    # ─────────── Scenario action helpers ───────────

    def _compute_speed_up(self, self_pose: Pose,
                          path: list) -> Tuple[float, float]:
        """Scenario 2: Accelerate along the existing path.

        Uses the same steering logic as Pure Pursuit but with
        a boosted linear velocity.
        """
        v_boost = min(self.v_max * self.v_boost_factor, self.v_max * 2.0)

        # Find a lookahead target on the path
        if not path:
            return (v_boost, 0.0)

        # Convert path points 
        target = self._get_path_lookahead(self_pose, path, lookahead=0.6)
        diff = target - self_pose.position
        alpha = normalize_angle(diff.orientation - self_pose.theta)

        # Curvature-based steering
        dist = max(0.01, self_pose.position.dist(target))
        curvature = 2.0 * math.sin(alpha) / dist
        w = v_boost * curvature
        w = max(-self.w_max, min(self.w_max, w))

        # Reduce speed if very misaligned
        v = v_boost * max(0.0, 1.0 - abs(alpha) / (math.pi / 2.0))
        v = max(0.0, min(v_boost, v))

        return (v, w)

    def _compute_dodge(self, self_pose: Pose, other_pos: Point,
                       self_dir: Vector,
                       grid_map: Optional[GridMap]
                       ) -> Tuple[float, float]:
        """Scenario 3: Compute lateral dodge maneuver."""
        
        # 1. LATCH THE WAYPOINT: Only calculate if we don't already have one
        if self._dodge_waypoint is None:
            # Compute perpendicular direction (try left first, then right)
            unit_dir = self_dir / max(self_dir.norm(), 1e-6)
            perp_left = Vector(-unit_dir.y, unit_dir.x)
            perp_right = Vector(unit_dir.y, -unit_dir.x)

            # Choose side
            vec_to_other = other_pos - self_pose.position
            cross_val = unit_dir.cross(vec_to_other)

            if cross_val > 0:
                primary, fallback = perp_right, perp_left
            else:
                primary, fallback = perp_left, perp_right

            escape = self_pose.position + primary * self.escape_offset

            # Check if escape point is free on the map
            if grid_map is not None:
                try:
                    val = grid_map.get(escape)
                    if val >= 50:
                        # Try the other side
                        escape = self_pose.position + fallback * self.escape_offset
                        try:
                            val2 = grid_map.get(escape)
                            if val2 >= 50:
                                # Both sides blocked — just stop
                                self._dodge_waypoint = None
                                return (0.0, 0.0)
                        except (IndexError, ValueError):
                            self._dodge_waypoint = None
                            return (0.0, 0.0)
                except (IndexError, ValueError):
                    # Out of map bounds — try other side
                    escape = self_pose.position + fallback * self.escape_offset

            # Lock in the decision
            self._dodge_waypoint = escape

        # 2. STEER TOWARD THE LOCKED WAYPOINT
        escape = self._dodge_waypoint
        
        diff = escape - self_pose.position
        alpha = normalize_angle(diff.orientation - self_pose.theta)

        # Use moderate speed during dodge
        v_dodge = self.v_max * 1.2
        dist_to_escape = max(0.01, self_pose.position.dist(escape))
        curvature = 2.0 * math.sin(alpha) / dist_to_escape
        w = v_dodge * curvature
        w = max(-self.w_max, min(self.w_max, w))

        v = v_dodge * max(0.0, 1.0 - abs(alpha) / (math.pi / 2.0))
        v = max(0.0, min(v_dodge, v))

        return (v, w)

    def _get_path_direction(self, self_pose: Pose, path: list) -> Optional[Vector]:
        """Compute our robot's direction from the A* planned path.
        Falls back to the buffer-based estimator only if the path is
        empty or too short to compute a direction.
        """
        if not path:
            return self.self_estimator.get_direction()

        target = self._get_path_lookahead(self_pose, path, lookahead=5.0)
        diff = target - self_pose.position
        if diff.norm() < 1e-6:
            return self.self_estimator.get_direction()
        return diff

    def _is_other_on_our_path_ahead(self, self_pos: Point, other_pos: Point,
                                     path: list,
                                     corridor_width: float = 0.8) -> bool:
        """Check if the other robot is within a corridor of our A* path ahead."""
        points = self._path_to_points(path)
        if not points:
            return False
        closest_idx = self._find_closest_path_index(self_pos, points)
        for i in range(closest_idx, len(points)):
            if other_pos.dist(points[i]) < corridor_width:
                return True
        return False

    def _other_trajectory_crosses_our_path(self, self_pos: Point, path: list,
                                            other_pos: Point,
                                            other_dir: Vector,
                                            corridor: float = 0.8) -> bool:
        """Check if the other robot's projected trajectory crosses our A* path."""
        points = self._path_to_points(path)
        if not points:
            return False
        nd = other_dir.norm()
        if nd < 1e-6:
            return False
        other_unit = other_dir / nd
        closest_idx = self._find_closest_path_index(self_pos, points)
        for i in range(closest_idx, len(points)):
            vec = points[i] - other_pos
            dot = other_unit.dot(vec)
            if dot <= 0:
                continue  # Behind the other robot
            perp = abs(other_unit.cross(vec))
            if perp < corridor:
                return True
        return False

    @staticmethod
    def _path_to_points(path: list) -> list:
        """Convert path elements to Point objects."""
        from pacr_interfaces.msg import Point2D
        points = []
        for pt in path:
            if isinstance(pt, Point):
                points.append(pt)
            elif isinstance(pt, Point2D):
                points.append(Point(pt.x, pt.y))
            else:
                points.append(Point(pt.x, pt.y))
        return points

    @staticmethod
    def _find_closest_path_index(pos: Point, points: list) -> int:
        """Find the index of the closest point to pos on the path."""
        min_dist = float('inf')
        idx = 0
        for i, pt in enumerate(points):
            d = pos.dist(pt)
            if d < min_dist:
                min_dist = d
                idx = i
        return idx

    @staticmethod
    def _get_path_lookahead(pose: Pose, path: list,
                            lookahead: float = 0.6) -> Point:
        """Find a lookahead point on the path from current pose."""
        points = LocalPlanner._path_to_points(path)
        if not points:
            return pose.position
        closest_idx = LocalPlanner._find_closest_path_index(
            pose.position, points)
        for i in range(closest_idx, len(points)):
            if pose.position.dist(points[i]) >= lookahead:
                return points[i]
        return points[-1]
