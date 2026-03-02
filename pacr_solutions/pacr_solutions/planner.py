"""A* Path Planning Implementation.

This module provides a discrete A* search algorithm for finding optimal
paths on a 2D occupancy grid. It converts continuous world coordinates
to grid cells, performs a heuristic guided search, and reconstructs
the final path for the robot to follow.
"""

import math
import heapq
import time
from typing import List, Tuple

from pacr_simulation.geometry2d import Point, Vector
from pacr_simulation.map_utils import GridMap


# ─────────────────────────────────────────────
# Global Planner (A*)
# ─────────────────────────────────────────────

class Planner:
    """Search-based path planner for grid maps.

    Implements the A* algorithm with 8-connectivity and Euclidean 
    distance heuristics to ensure efficient and optimal global paths.
    """

    def __init__(self):
        """Initialize the planner."""
        pass

    def plan(self, start: Point, goal: Point, grid_map: GridMap, 
             timeout: float = 10.0) -> List[Point]:
        """
        Plans a path from start to goal in the given GridMap using A*.

        Args:
            start: Continuous world coordinate for path start.
            goal: Continuous world coordinate for path destination.
            grid_map: The occupancy grid map to plan on.
            timeout: Maximum allowed computation time in seconds.

        Returns:
            A list of continuous Points representing the optimal path.
        """
        start_time = time.time()

        # ── 1. Coordinate conversion helpers ──

        def point_to_cell(p: Point) -> Tuple[int, int]:
            """Convert continuous Point to grid cell indices (ints)."""
            float_indices = (p - grid_map.origin) / grid_map.resolution
            return tuple(map(round, float_indices))

        def cell_to_point(c: Tuple[int, int]) -> Point:
            """Convert grid cell indices back to continuous world Point."""
            return grid_map.origin + Vector(c[0], c[1]) * grid_map.resolution

        # ── 2. Grid validation helpers ──

        def is_valid_cell(c: Tuple[int, int]) -> bool:
            """Check if cell indices are within the map boundaries."""
            return (0 <= c[0] < grid_map.shape[0] and
                    0 <= c[1] < grid_map.shape[1])

        def is_free_cell(c: Tuple[int, int]) -> bool:
            """Check if a cell is navigateable (occupancy < 50)."""
            return grid_map.data[c[0], c[1]] < 50

        # Discretize start and goal
        start_cell = point_to_cell(start)
        goal_cell = point_to_cell(goal)

        # Sanity check: start/goal must be within map
        if not is_valid_cell(start_cell) or not is_valid_cell(goal_cell):
            return []

        # ── 3. A* Algorithm setup ──

        # Priority queue: (f_score, cell)
        open_set = []
        heapq.heappush(open_set, (0.0, start_cell))

        # Reconstructed path tracing
        came_from = {}

        # G-score: exact cost from start to current node
        g_score = {start_cell: 0.0}

        def heuristic(c1: Tuple[int, int], c2: Tuple[int, int]) -> float:
            """Euclidean distance heuristic in world coordinates."""
            return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) * grid_map.resolution

        # F-score: estimated total cost (g + h)
        f_score = {start_cell: heuristic(start_cell, goal_cell)}

        # 8-connected grid transitions (N, E, S, W + Diagonals)
        transitions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        # ── 4. Main search loop ──

        while open_set:
            # Check for timeout to maintain node responsiveness
            if time.time() - start_time > timeout:
                break

            # Process node with lowest estimated total cost
            _, current = heapq.heappop(open_set)

            # Target reached: Reconstruct and return the path
            if current == goal_cell:
                path = []
                while current in came_from:
                    path.append(cell_to_point(current))
                    current = came_from[current]

                # Prepend start point and reverse to correct order
                path.append(start)
                path.reverse()

                # Ensure path ends precisely at the continuous goal point
                if not path[-1].isclose(goal):
                    path.append(goal)

                return path

            # Expand neighbors
            for dx, dy in transitions:
                neighbor = (current[0] + dx, current[1] + dy)

                if not is_valid_cell(neighbor):
                    continue
                if not is_free_cell(neighbor):
                    continue

                # Collision safety: Prevent cutting corners on diagonals
                if dx != 0 and dy != 0:
                    if not is_free_cell((current[0] + dx, current[1])) or \
                       not is_free_cell((current[0], current[1] + dy)):
                        continue

                # Calculate spatial distance to neighbor
                step_cost = math.hypot(dx, dy) * grid_map.resolution
                tentative_g_score = g_score[current] + step_cost

                # Found a better path to neighbor?
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + heuristic(neighbor, goal_cell)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        # Return empty if unreachable or timed out
        return []
