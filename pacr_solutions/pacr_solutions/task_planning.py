"""ROS2 Node for High-Level Task Planning.

This node provides a bridge between the ROS2 message system and the
Markov Decision Process (MDP) solver. It maintains a pre-solved 
Value Iteration policy and serves high-level actions (Pick, Place, 
Transform, Wait, Goto) to the executive node.

── PROJECT EXTENSION ──────────────────────────────────────────────────────────
This node integrates the project 'Extension' tp3 by relaying the concurrent 
robot's workshop occupancy status (from GetAction's workshop_state) to the 
MDP model.
──────────────────────────────────────────────────────────────────────────────
"""

import rclpy
from rclpy.node import Node

from pacr_interfaces.srv import GetAction
from pacr_solutions.mdp import MDP, RobotLocation, ObjectState, WorkshopOccupancy, Action


# ─────────────────────────────────────────────
# ROS to MDP Mapping
# ─────────────────────────────────────────────

# Mapping from ROS service constants to internal MDP Enums
LOCATION_MAP = {
    GetAction.Request.R_OTHER: RobotLocation.OTHER,
    GetAction.Request.R_START1: RobotLocation.START1,
    GetAction.Request.R_START2: RobotLocation.START2,
    GetAction.Request.R_WORKSHOP1: RobotLocation.WORKSHOP1,
    GetAction.Request.R_WORKSHOP2: RobotLocation.WORKSHOP2,
    GetAction.Request.R_INTERMEDIARY1: RobotLocation.INTERMEDIARY1,
    GetAction.Request.R_INTERMEDIARY2: RobotLocation.INTERMEDIARY2,
    GetAction.Request.R_INTERMEDIARY3: RobotLocation.INTERMEDIARY3,
}

OBJECT_MAP = {
    GetAction.Request.O_NO_OBJECT: ObjectState.NO_OBJECT,
    GetAction.Request.O_START1: ObjectState.START1,
    GetAction.Request.O_START2: ObjectState.START2,
    GetAction.Request.O_CARRIED_UNTRANSFORMED: ObjectState.CARRIED_UNTRANSFORMED,
    GetAction.Request.O_CARRIED_TRANSFORMED: ObjectState.CARRIED_TRANSFORMED,
}

WORKSHOP_MAP = {
    GetAction.Request.CR_NONE: WorkshopOccupancy.NONE,
    GetAction.Request.CR_WORKSHOP1: WorkshopOccupancy.WORKSHOP1,
    GetAction.Request.CR_WORKSHOP2: WorkshopOccupancy.WORKSHOP2,
}

# Mapping from internal MDP Action types back to ROS service constants
RESPONSE_MAP = {
    Action.PICK: GetAction.Response.PICK,
    Action.PLACE: GetAction.Response.PLACE,
    Action.TRANSFORM: GetAction.Response.TRANSFORM,
    Action.WAIT: GetAction.Response.WAIT,
    Action.GOTO_OTHER: GetAction.Response.GOTO_OTHER,
    Action.GOTO_START1: GetAction.Response.GOTO_START1,
    Action.GOTO_START2: GetAction.Response.GOTO_START2,
    Action.GOTO_WORKSHOP1: GetAction.Response.GOTO_WORKSHOP1,
    Action.GOTO_WORKSHOP2: GetAction.Response.GOTO_WORKSHOP2,
    Action.GOTO_INTERMEDIARY1: GetAction.Response.GOTO_INTERMEDIARY1,
    Action.GOTO_INTERMEDIARY2: GetAction.Response.GOTO_INTERMEDIARY2,
    Action.GOTO_INTERMEDIARY3: GetAction.Response.GOTO_INTERMEDIARY3,
}


# ─────────────────────────────────────────────
# Task Planning Node
# ─────────────────────────────────────────────

class TaskPlanningNode(Node):
    """ROS2 Node for task planning via MDP.

    Encapsulates the decision-making logic of the robot by solving
    an MDP and providing a service interface for command lookup.
    """

    def __init__(self):
        super().__init__('task_planning')
        
        # ── Solver Initialization ──
        self.get_logger().info('Initializing MDP and solving Value Iteration...')
        self.mdp = MDP()
        self.mdp.solve_value_iteration()
        self.get_logger().info('Value Iteration complete. Ready to serve.')
        
        # ── Service Interface ──
        self.srv = self.create_service(GetAction, 'get_action', self.get_action_cb)

    def get_action_cb(self, request, response):
        """
        Main decision-making service callback.

        Args:
            request: GetAction request with current robot, object, and map state.
            response: GetAction response to be populated with the best action.

        Returns:
            The populated GetAction response.
        """
        # ── Parse Incoming State ──
        loc_val = request.controlled_robot_location
        obj_val = request.object_state
        workshop_val = request.workshop_state
        
        loc = LOCATION_MAP.get(loc_val, RobotLocation.OTHER)
        obj = OBJECT_MAP.get(obj_val, ObjectState.NO_OBJECT)
        w = WORKSHOP_MAP.get(workshop_val, WorkshopOccupancy.NONE)
        
        self.get_logger().info(f"Received request: loc={loc.name}, obj={obj.name}, occ={w.name}")
        
        # ── Policy Lookup ──
        state = (loc, obj, w)
        
        if state in self.mdp.policy:
            best_action = self.mdp.policy[state]
            response.action = RESPONSE_MAP[best_action]
            self.get_logger().info(f"Commanded action: {best_action.name}")
        else:
            # Safe default fallback
            self.get_logger().warn("Unknown state, defaulting to WAIT")
            response.action = GetAction.Response.WAIT
            
        return response


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
