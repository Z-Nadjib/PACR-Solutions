# Running the System

This page provides the launch commands for each TP, including how to select between the **reactive** and **deliberative (DWA)** local planners.

---

## Prerequisites

Before launching any TP, make sure the workspace is built:

```bash
cd ~/ros2_ws
colcon build --packages-select pacr_solutions
source install/setup.bash
```

---

## TP1 — Path Planning (A* Visualization)

**What it does**: Launches the simulation environment and the A* path planning node. The node computes and publishes optimal paths for visualization in RViz — it does **not** control the robot.

```bash
ros2 launch pacr_solutions test_path_planning.launch.xml
```

**Nodes launched**: `path_planning` (A* search)

---

## TP2 — Path Following

**What it does**: Launches the simulation with the `test_goto` executive, the A* path planning node, and a local planner. The robot follows the planned path while avoiding the concurrent robot.

### Reactive Planner (Behavior-Based)

Uses the scenario-based reactive planner with Pure Pursuit controller.

```bash
ros2 launch pacr_solutions test_path_following.launch.xml local_planner:=reactive
```

**Nodes launched**: `path_planning` + `path_following` (Pure Pursuit + LocalPlanner)

### Deliberative Planner (DWA)

Uses the Dynamic Window Approach deliberative planner. This is the **default** configuration:

```bash
ros2 launch pacr_solutions test_path_following.launch.xml
```

Or explicitly:

```bash
ros2 launch pacr_solutions test_path_following.launch.xml local_planner:=dwa
```

**Nodes launched**: `path_planning` + `dwa_path_following` (DWA velocity-space optimization)

---

## TP3 — Task Planning (Full System)

**What it does**: Launches the complete system with MDP task planning, A* path planning, and a local planner. The robot autonomously performs pick-transform-place cycles using the optimal MDP policy.

### With DWA Planner (Default)

```bash
ros2 launch pacr_solutions test_task_planning.launch.xml
```

Or explicitly:

```bash
ros2 launch pacr_solutions test_task_planning.launch.xml local_planner:=dwa
```

**Nodes launched**: `task_planning` + `path_planning` + `dwa_path_following`

### With Reactive Planner

```bash
ros2 launch pacr_solutions test_task_planning.launch.xml local_planner:=reactive
```

**Nodes launched**: `task_planning` + `path_planning` + `path_following`

---

## Summary Table

| TP | Launch Command | Default Planner | Nodes |
|----|---------------|-----------------|-------|
| **TP1** | `ros2 launch pacr_solutions test_path_planning.launch.xml` | — (no local planner) | `path_planning` |
| **TP2** | `ros2 launch pacr_solutions test_path_following.launch.xml` | DWA | `path_planning` + `dwa_path_following` |
| **TP2 (Reactive)** | `...test_path_following.launch.xml local_planner:=reactive` | Reactive | `path_planning` + `path_following` |
| **TP3** | `ros2 launch pacr_solutions test_task_planning.launch.xml` | DWA | `task_planning` + `path_planning` + `dwa_path_following` |
| **TP3 (Reactive)** | `...test_task_planning.launch.xml local_planner:=reactive` | Reactive | `task_planning` + `path_planning` + `path_following` |

!!! note "Default Planner Behavior"
    - **TP2** defaults to the **DWA** planner.
    - **TP3** defaults to the **DWA** planner because the full task planning mission benefits from the deliberative approach's superior collision avoidance across multiple navigation cycles.
