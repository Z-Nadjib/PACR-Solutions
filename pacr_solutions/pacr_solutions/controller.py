"""Pure Pursuit Path Following Controller.

This module implements the Pure Pursuit algorithm, which computes
velocity commands (v, w) to drive the robot along a path by
steering toward a lookahead point.
"""

import math
from typing import List, Optional, Tuple

from pacr_simulation.geometry2d import Point, Pose, normalize_angle


# ─────────────────────────────────────────────
# Path Follower (Pure Pursuit)
# ─────────────────────────────────────────────
class PathFollower:
    """Implementation of the Pure Pursuit controller.

    Computes the linear and angular velocities required to track
    a reference path based on the robot's current pose and a 
    lookahead distance.
    """

    def __init__(self, v_max: float = 0.5, w_max: float = 1.0, 
                 lookahead_distance: float = 0.4):
        self.v_max = v_max
        self.w_max = w_max
        self.lookahead_distance = lookahead_distance
        self.goal_tolerance = 0.1 

    def get_command(self, robot_pose: Pose, path: List[Point], 
                    goal: Pose) -> Tuple[float, float, bool]:
        """
        Compute control (v, w) and tracking status.

        Args:
            robot_pose: Current pose of the robot.
            path: List of points defining the reference path.
            goal: Target pose at the end of the path.

        Returns:
            A tuple of (linear_velocity, angular_velocity, finished_flag).
        """
        if not path:
            return 0.0, 0.0, True
            
        # ── 1. Check distance to final goal position ──
        dist_to_goal = robot_pose.position.dist(goal.position)
        if dist_to_goal < self.goal_tolerance:
            # We are translationally at the goal, align orientation 
            angle_error = normalize_angle(goal.theta - robot_pose.theta)
            if abs(angle_error) < 0.05:
                return 0.0, 0.0, True # Completely finished
            else:
                # Rotate in place
                w = 1.5 * angle_error
                w = max(-self.w_max, min(self.w_max, w))
                return 0.0, w, False
                
        # ── 2. Pure Pursuit: Find the lookahead point ──
        lookahead_point = path[-1] # default to the end
        
        # Traverse path to find the furthest point within lookahead_distance
        # Note: Track closest point ahead of us on the discretized path.
        
        # Find closest point on path first
        min_dist = float('inf')
        closest_idx = 0
        for i, pt in enumerate(path):
            d = robot_pose.position.dist(pt)
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        # Look ahead from closest point
        for i in range(closest_idx, len(path)):
            if robot_pose.position.dist(path[i]) >= self.lookahead_distance:
                lookahead_point = path[i]
                break
                
        # ── 3. Compute Steering Command (w) ──
        # Vector to lookahead point
        diff = lookahead_point - robot_pose.position
        
        # Angle to target relative to robot's current heading
        alpha = normalize_angle(diff.orientation - robot_pose.theta)
        
        # Steering control law: Curvature to the lookahead point
        # k = 2 * sin(alpha) / Ld
        dist_to_lookahead = max(0.01, robot_pose.position.dist(lookahead_point)) # prevent div/0
        curvature = 2.0 * math.sin(alpha) / dist_to_lookahead
        
        #  P-control for steering mapping curvature to w
        w = self.v_max * curvature
        w = max(-self.w_max, min(self.w_max, w))

        # ── 4. Compute Velocity Command (v) ──
        # Slow down if facing away from the path to prevent wide turns
        v = self.v_max * max(0.0, 1.0 - abs(alpha) / (math.pi / 2.0))
        
        # Reduce speed as we get closer to the goal
        if dist_to_goal < 0.5:
            v = min(v, dist_to_goal)
        v = max(0.0, min(self.v_max, v))
        
        return v, w, False

    def get_command_with_speed(self, robot_pose: Pose, path: List[Point],
                               goal: Pose, speed_override: float) -> Tuple[float, float, bool]:
        """Same as get_command but with a custom max speed.

        Used by the local planner for speed-boost scenarios.
        """
        original_v_max = self.v_max
        self.v_max = speed_override
        result = self.get_command(robot_pose, path, goal)
        self.v_max = original_v_max
        return result
