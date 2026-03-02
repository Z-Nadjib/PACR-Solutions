"""Markov Decision Process (MDP) for High-Level Task Planning.

This module defines the state space, action space, and transition
dynamics for the robot's high-level task planning. It includes
Value Iteration to compute the optimal policy for picking, 
transforming, and placing objects in the environment.

── PROJECT EXTENSION ──────────────────────────────────────────────────────────
This module implements the project 'Extension' from tp3 by accounting for the 
concurrent robot in high-level task planning. Workshop occupancy by the 
other agent is integrated into the state space (WorkshopOccupancy) and 
transition dynamics (probabilistic clearing during the WAIT action).
──────────────────────────────────────────────────────────────────────────────
"""

from enum import Enum
import math


# ─────────────────────────────────────────────
# State space definitions 
# ─────────────────────────────────────────────

class RobotLocation(Enum):
    """Discrete locations in the environment."""
    OTHER = 0
    START1 = 1
    START2 = 2
    WORKSHOP1 = 3
    WORKSHOP2 = 4
    INTERMEDIARY1 = 5
    INTERMEDIARY2 = 6
    INTERMEDIARY3 = 7


class ObjectState(Enum):
    """The state of the object being handled."""
    NO_OBJECT = 0
    START1 = 1
    START2 = 2
    CARRIED_UNTRANSFORMED = 3
    CARRIED_TRANSFORMED = 4


class WorkshopOccupancy(Enum):
    """Occupancy state of the workshops by the concurrent robot."""
    NONE = 0
    WORKSHOP1 = 1
    WORKSHOP2 = 2


class Action(Enum):
    """Set of high-level actions the robot can perform."""
    PICK = 0
    PLACE = 1
    TRANSFORM = 2
    WAIT = 3
    GOTO_OTHER = 4
    GOTO_START1 = 5
    GOTO_START2 = 6
    GOTO_WORKSHOP1 = 7
    GOTO_WORKSHOP2 = 8
    GOTO_INTERMEDIARY1 = 9
    GOTO_INTERMEDIARY2 = 10
    GOTO_INTERMEDIARY3 = 11


# ─────────────────────────────────────────────
# Transition logic and helpers
# ─────────────────────────────────────────────

def get_allowed_goto(loc: RobotLocation, workshop_occ: WorkshopOccupancy) -> list[Action]:
    """
    Returns valid GOTO actions from a given location and workshop occupancy condition.

    Args:
        loc: Current discrete location of the robot.
        workshop_occ: Current occupancy state of the workshops.

    Returns:
        A list of valid GOTO actions based on spatial connectivity and occupancy.
    """
    actions = []
    if loc == RobotLocation.START1:
        actions = [Action.GOTO_INTERMEDIARY1, Action.GOTO_INTERMEDIARY2, Action.GOTO_START2]
    elif loc == RobotLocation.START2:
        actions = [Action.GOTO_INTERMEDIARY2, Action.GOTO_INTERMEDIARY3, Action.GOTO_START1]
    elif loc == RobotLocation.WORKSHOP1:
        actions = [Action.GOTO_INTERMEDIARY1, Action.GOTO_INTERMEDIARY2]
    elif loc == RobotLocation.WORKSHOP2:
        actions = [Action.GOTO_INTERMEDIARY3]
    elif loc == RobotLocation.INTERMEDIARY1:
        actions = [Action.GOTO_START1, Action.GOTO_WORKSHOP1, Action.GOTO_INTERMEDIARY2]
    elif loc == RobotLocation.INTERMEDIARY2:
        actions = [Action.GOTO_START1, Action.GOTO_START2, Action.GOTO_WORKSHOP1, Action.GOTO_INTERMEDIARY1, Action.GOTO_INTERMEDIARY3]
    elif loc == RobotLocation.INTERMEDIARY3:
        actions = [Action.GOTO_START2, Action.GOTO_WORKSHOP2, Action.GOTO_INTERMEDIARY2]
        
    # Remove GOTO for occupied workshops
    if workshop_occ == WorkshopOccupancy.WORKSHOP1 and Action.GOTO_WORKSHOP1 in actions:
        actions.remove(Action.GOTO_WORKSHOP1)
    if workshop_occ == WorkshopOccupancy.WORKSHOP2 and Action.GOTO_WORKSHOP2 in actions:
        actions.remove(Action.GOTO_WORKSHOP2)
        
    return actions


def get_action_duration(act: Action) -> float:
    """Returns the estimated time (cost) for a given action."""
    if act == Action.PICK: return 5.0
    if act == Action.PLACE: return 5.0
    if act == Action.TRANSFORM: return 10.0
    if act == Action.WAIT: return 2.0
    if act in [Action.GOTO_START1, Action.GOTO_START2, 
               Action.GOTO_WORKSHOP1, Action.GOTO_WORKSHOP2,
               Action.GOTO_INTERMEDIARY1, Action.GOTO_INTERMEDIARY2, Action.GOTO_INTERMEDIARY3]:
        return 7.0 
    return 1.0


# ─────────────────────────────────────────────
# MDP Model and Solver
# ─────────────────────────────────────────────

class MDP:
    """Markov Decision Process solver using Value Iteration."""

    def __init__(self):
        # Generate State Space
        self.states = []
        for loc in RobotLocation:
            if loc == RobotLocation.OTHER: continue
            for obj in ObjectState:
                for w in WorkshopOccupancy:
                    self.states.append((loc, obj, w))
        
        self.gamma = 0.99
        self.V = {s: 0.0 for s in self.states}
        self.policy = {s: Action.WAIT for s in self.states}
        
    def get_actions(self, state) -> list[Action]:
        """Returns valid actions in a given state."""
        loc, obj, w = state
        actions = []
        
        # Picking an object
        if obj in [ObjectState.START1, ObjectState.START2] and loc.name == obj.name:
            actions.append(Action.PICK)
        
        # Placing an object
        if obj == ObjectState.CARRIED_TRANSFORMED and loc in [RobotLocation.START1, RobotLocation.START2]:
            actions.append(Action.PLACE)
            
        # Transforming an object
        if obj == ObjectState.CARRIED_UNTRANSFORMED and loc in [RobotLocation.WORKSHOP1, RobotLocation.WORKSHOP2]:
            actions.append(Action.TRANSFORM)
            
        # Navigation actions
        actions.extend(get_allowed_goto(loc, w))
        actions.append(Action.WAIT)
        return actions

    def transition(self, state, action):
        """
        Computes the transition dynamics of the environment.

        Returns:
            List of (probability, next_state, reward).
        """
        loc, obj, w = state
        duration = get_action_duration(action)
        reward = -duration 
        
        if action == Action.PICK:
            return [(1.0, (loc, ObjectState.CARRIED_UNTRANSFORMED, w), reward)]
            
        elif action == Action.PLACE:
            final_reward = reward + 1000.0
            return [
                (0.5, (loc, ObjectState.START1, w), final_reward),
                (0.5, (loc, ObjectState.START2, w), final_reward)
            ]
            
        elif action == Action.TRANSFORM:
            return [(1.0, (loc, ObjectState.CARRIED_TRANSFORMED, w), reward)]
            
        elif action.name.startswith("GOTO_"):
            target_str = action.name.replace("GOTO_", "")
            target_loc = RobotLocation[target_str]
            return [(1.0, (target_loc, obj, w), reward)]
            
        elif action == Action.WAIT:
            # If we wait, the workshop might clear up (50% chance move to NONE)
            if w != WorkshopOccupancy.NONE:
                return [(0.5, (loc, obj, w), reward), (0.5, (loc, obj, WorkshopOccupancy.NONE), reward)]
            return [(1.0, (loc, obj, w), reward)]
            
        return [(1.0, state, reward)]

    def solve_value_iteration(self, epsilon=1e-3, max_iterations=1000):
        """Runs the Value Iteration algorithm to convergence."""
        for i in range(max_iterations):
            delta = 0
            new_V = self.V.copy()
            
            for s in self.states:
                best_val = -float('inf')
                best_act = None
                
                valid_actions = self.get_actions(s)
                for a in valid_actions:
                    transitions = self.transition(s, a)
                    expected_val = 0
                    for prob, next_s, reward in transitions:
                        expected_val += prob * (reward + self.gamma * self.V[next_s])
                        
                    if expected_val > best_val:
                        best_val = expected_val
                        best_act = a
                
                new_V[s] = best_val
                self.policy[s] = best_act
                delta = max(delta, abs(new_V[s] - self.V[s]))
                
            self.V = new_V
            if delta < epsilon:
                print(f"Value iteration converged in {i} steps.")
                break


# ─────────────────────────────────────────────
# Verification script (CLI)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    mdp = MDP()
    mdp.solve_value_iteration()
    
    # Verify optimal actions for specific states
    print(f"{'State (Loc, Obj, Occ)':<40} | Optimal Action")
 
