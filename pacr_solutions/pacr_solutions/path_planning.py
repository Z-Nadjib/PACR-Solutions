"""ROS2 Node for Global Path Planning.

This node provides a path planning service by subscribing to the 
environment's costmap and utilizing an A* search algorithm
to find the optimal path from a start pose to a goal pose.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from nav_msgs.msg import OccupancyGrid
from pacr_interfaces.srv import GetPath
from pacr_interfaces.msg import Point2D

from pacr_simulation.map_utils import GridMap
from pacr_simulation.geometry2d import Point, Pose

from pacr_solutions.planner import Planner


# ─────────────────────────────────────────────
# Path Planning Node
# ─────────────────────────────────────────────

class PathPlanningNode(Node):
    """ROS2 Node for Global Path Planning.

    Interfaces with the A* search planner to compute optimal paths
    traversing the costmap's occupancy grid. Provides the 'get_path' 
    service to higher-level task planners and the executive node.
    """

    def __init__(self):
        super().__init__('path_planning')
        
        # ── Parameters ──
        self.declare_parameter('timeout', 5.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value
        
        # ── Internal Planner State ──
        self.grid_map = None
        self.planner = Planner()
        
        # ── Costmap Subscription (Latching) ──
        latching_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, 'costmap', self.costmap_cb,
            qos_profile=latching_qos)
            
        # ── Service Interface ──
        self.srv = self.create_service(GetPath, 'get_path', self.get_path_cb)
        
        self.get_logger().info('PathPlanningNode initialized.')
        
    # ─────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────

    def costmap_cb(self, msg: OccupancyGrid):
        self.grid_map = GridMap.from_msg(msg)
        self.get_logger().info('Costmap received and updated.')
        
    def get_path_cb(self, request, response):
        """
        Main path planning service callback.
        
        Args:
            request: GetPath request containing start and goal poses.
            response: GetPath response to be populated with the path.

        Returns:
            The populated GetPath response.
        """
        if self.grid_map is None:
            self.get_logger().warn('No costmap received yet. Cannot plan path.')
            return response
            
        # ── Extract Start and Goal ──
        start_pose = Pose.from_msg(request.start)
        goal_pose = Pose.from_msg(request.goal)
        
        self.get_logger().info(
            f'Planning path from {start_pose.position} to {goal_pose.position}...')
        
        # ── Invoke A* Planner ──
        path_points = self.planner.plan(
            start=start_pose.position, 
            goal=goal_pose.position, 
            grid_map=self.grid_map, 
            timeout=self.timeout
        )
        
        if not path_points:
            self.get_logger().warn('No path found! Returning empty path.')
            return response
            
        # ── Construct Response Path ──
        response_path = []
        for p in path_points:
            p2d = Point2D()
            p2d.x = p.x
            p2d.y = p.y
            response_path.append(p2d)
            
        response.path = response_path
        self.get_logger().info(f'Path found with {len(response_path)} points.')
        
        return response


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
