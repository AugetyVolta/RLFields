"""Project displacement and task sampling interfaces for stock IR-SIM."""
import numpy as np
import shapely
from irsim.lib import GeometryFactory


def step_displacements(world, increments, robot_indices):
    # The policy already multiplies speed by dt. External mode keeps these
    # world-frame position increments and lets IR-SIM update sensors/status.
    # External mode does not infer velocity from a new pose. Clear robots
    # without an action, then supply the executed velocity in the body frame.
    for robot in world.robot_list:
        robot.set_velocity(np.zeros(robot.vel_shape))
    for index, increment in zip(robot_indices, increments):
        robot = world.robot_list[index]
        if robot.static or robot.stop_flag:
            continue
        state = robot.state.astype(float, copy=True)
        delta = np.asarray(increment, dtype=float).reshape(2, 1)
        state[:2] += delta
        robot.set_state(state)
        robot.set_velocity(robot.vel_world2body(delta / world.step_time))
    world.step()


def sample_free_position(world, robot, *, free=True, filter_start_goal=False,
                         check_radius=0.2, range_limits=None,
                         max_attempts=200, dis_tol=1.0):
    # Keep the custom project's start/goal constraints in this environment;
    # no simulator classes or global object lists are patched.
    limits = range_limits if range_limits is not None else [robot.rl, robot.rh]
    low, high = (np.asarray(bound).reshape(-1, 1) for bound in limits)
    others = [obj for obj in world.objects if obj is not robot]
    other_positions = []
    if filter_start_goal:
        for obj in others:
            if obj.role == 'robot' and obj.position is not None and obj.goal is not None:
                other_positions.extend([obj.position[:2, 0].copy(), obj.goal[:2, 0].copy()])
    circle = GeometryFactory.create_geometry(name='circle', radius=check_radius)
    for _ in range(max_attempts):
        position = world._env_param.rng.uniform(low, high)
        if free:
            geometry = circle.step(position)
            if any(shapely.intersects(geometry, obj._geometry) for obj in others):
                continue
        if any(np.linalg.norm(position[:2, 0] - xy) < dis_tol for xy in other_positions):
            continue
        return position
    robot.logger.warning('No free position found within max_attempts; returning the final candidate.')
    return position
