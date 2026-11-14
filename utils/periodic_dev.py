"""Training-internal, parallel CPU DEV on fixed 20-robot tasks."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys


def launch_dev_evaluation(log_dir, iteration):
    """Call after checkpoint saving; training continues while DEV runs."""
    if iteration % 20:
        return None
    log_dir = Path(log_dir).resolve()
    output = log_dir / 'evaluation'
    output.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')
    with (output / f'iter_{iteration:06d}.log').open('w') as log:
        worker_code = (
            'from utils.periodic_dev import _run_dev_evaluation\n'
            'import sys\n'
            '_run_dev_evaluation(sys.argv[1], int(sys.argv[2]))\n'
        )
        process = subprocess.Popen(
            [sys.executable, '-u', '-c', worker_code, str(log_dir), str(iteration)],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=environment, stdout=log, stderr=subprocess.STDOUT)
    print(f'[dev] iter={iteration} CPU evaluation started, pid={process.pid}', flush=True)
    return process


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _fixed_cases(env, path, config_digest, episodes):
    if path.exists():
        bank = json.loads(path.read_text())
        if bank['config_sha256'] != config_digest or len(bank['cases']) != episodes:
            raise ValueError('Fixed DEV configuration changed; use a new experiment output directory.')
        return bank['cases']
    initial = [(robot.state.copy(), robot.goal.copy()) for robot in env.sim.robot_list]
    cases = []
    for index in range(episodes):
        # Sampling must not depend on the previously evaluated policy trajectory.
        for robot, (state, goal) in zip(env.sim.robot_list, initial):
            robot.set_state(state.copy(), init=True)
            robot.set_goal(goal.copy(), init=True)
            robot.reset()
        seed = 20261114 + index
        env.reset(seed=seed)
        cases.append(dict(id=f'dev-{index:03d}', seed=seed,
                          positions=[robot.state[:2, 0].tolist() for robot in env.sim.robot_list],
                          goals=[robot.goal[:2, 0].tolist() for robot in env.sim.robot_list]))
    _write_json(path, dict(config_sha256=config_digest, robots=20, cases=cases))
    return cases


def _episode(model, env, case):
    import numpy as np
    import torch

    env.sim.env.set_random_seed(case['seed'])
    obs, _ = env.sim.reset(robot_states=[p + [0.] for p in case['positions']],
                           robot_goals=[g + [0.] for g in case['goals']])
    assert np.allclose([r.state[:2, 0] for r in env.sim.robot_list], case['positions'], atol=1e-5, rtol=0)
    assert np.allclose([r.goal[:2, 0] for r in env.sim.robot_list], case['goals'], atol=1e-5, rtol=0)
    active = list(obs)
    outcomes = {}
    for _ in range(env.sim.single_max_timestep):
        batch = torch.as_tensor(np.stack([obs[name] for name in active]), dtype=torch.float32)
        with torch.no_grad():
            actions = model.forward({'obs': batch}, [], None)[0][:, :2].cpu().numpy()
        assert np.isfinite(actions).all()
        obs, _, terminated, truncated, infos = env.step(dict(zip(active, actions)))
        for name in active:
            if terminated[name] or truncated[name]:
                info = infos[name]
                robot_collision = info['r_coll_robot'] < 0
                static_collision = info['r_coll_obst'] < 0
                outcomes[name] = dict(
                    arrive=bool(info['r_arrive'] > 0 and not robot_collision
                                and not static_collision and not truncated[name]),
                    collision_robot=bool(robot_collision), collision_static=bool(static_collision),
                    timeout=bool(truncated[name]))
        active = [name for name in active if name not in outcomes]
        if terminated['__all__'] or truncated['__all__']:
            break
    assert len(outcomes) == env.num_robots, 'Incomplete DEV episode'
    return dict(case=case['id'], outcomes=outcomes)


def _evaluate_cases(checkpoint, cases):
    import torch
    from env.multi_robot_env import MultiRobotEnv
    from env.params import SceneConfig
    from model.policy_value_network import RLFieldsActorModel

    torch.set_num_threads(1)
    checkpoint = Path(checkpoint)
    cfg = SceneConfig.from_checkpoint(checkpoint)
    cfg.robot_params.number_robots = 20
    cfg.network_params.device = 'cpu'
    cfg.experiment_params.disable_plotting = True
    env = MultiRobotEnv(cfg.get_env_config())
    try:
        with (checkpoint / 'policies/shared_policy/policy_state.pkl').open('rb') as stream:
            state = pickle.load(stream)
        model_config = dict(state['policy_spec']['config']['model'])
        model_config['custom_model_config'] = dict(model_config.get('custom_model_config', {}))
        model_config['custom_model_config'].update(device='cpu', nt_pretrained_path=cfg.network_params.nt_pretrained_path)
        model = RLFieldsActorModel(env.get_observation_space('robot_0'), env.get_action_space('robot_0'),
                                  4, model_config, 'fixed_dev')
        model.load_state_dict({key: torch.as_tensor(value) for key, value in state['weights'].items()}, strict=True)
        model.eval().requires_grad_(False)
        return dict(rows=[_episode(model, env, case) for case in cases], pid=os.getpid())
    finally:
        env.close()


def evaluate_checkpoint(log_dir, iteration, episodes=16):
    import torch
    from tensorboardX import SummaryWriter
    from env.multi_robot_env import MultiRobotEnv
    from env.params import SceneConfig

    torch.set_num_threads(1)
    log_dir = Path(log_dir).resolve()
    output = log_dir / 'evaluation'
    output.mkdir(exist_ok=True)
    checkpoint = log_dir / 'ckpts' / f'iter_{iteration:06d}'
    policy_file = checkpoint / 'policies/shared_policy/policy_state.pkl'
    cfg = SceneConfig.from_checkpoint(checkpoint)
    trained_robots = cfg.robot_params.number_robots
    cfg.robot_params.number_robots = 20
    cfg.network_params.device = 'cpu'
    cfg.experiment_params.disable_plotting = True
    simulation_config = cfg.get_simulation_config()
    config_digest = hashlib.sha256(json.dumps(simulation_config, sort_keys=True).encode()).hexdigest()
    env = MultiRobotEnv(cfg.get_env_config())
    try:
        cases_path = output / 'cases.json'
        cases = _fixed_cases(env, cases_path, config_digest, episodes)
        worker_count = min(4, len(cases))
        partitions = [cases[index::worker_count] for index in range(worker_count)]
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context('spawn')) as executor:
            batches = list(executor.map(_evaluate_cases, [str(checkpoint)] * worker_count, partitions))
        rows = [row for batch in batches for row in batch['rows']]
        by_case = {row['case']: row for row in rows}
        assert len(rows) == len(cases) and set(by_case) == {case['id'] for case in cases}
        rows = [by_case[case['id']] for case in cases]
        outcomes = [value for row in rows for value in row['outcomes'].values()]
        n = len(outcomes)
        metrics = dict(
            PSR=sum(v['arrive'] for v in outcomes) / n,
            TSR=sum(all(v['arrive'] for v in row['outcomes'].values()) for row in rows) / len(rows),
            Coll=sum(v['collision_robot'] or v['collision_static'] for v in outcomes) / n,
            CollRobot=sum(v['collision_robot'] for v in outcomes) / n,
            CollStatic=sum(v['collision_static'] for v in outcomes) / n,
            Timeout=sum(v['timeout'] for v in outcomes) / n)
        result = dict(iteration=iteration, checkpoint=str(checkpoint), device='cpu',
                      workers=worker_count, worker_pids=sorted({batch['pid'] for batch in batches}),
                      checkpoint_sha256=hashlib.sha256(policy_file.read_bytes()).hexdigest(),
                      cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                      trained_robots=trained_robots, robots_per_episode=20, episodes=len(rows),
                      metrics=metrics, episode_results=rows)
        _write_json(output / f'iter_{iteration:06d}.json', result)
        # Separate run directory keeps the active training event file intact.
        with SummaryWriter(str(log_dir / 'tensorboard' / 'evaluation'),
                           filename_suffix=f'.iter_{iteration:06d}') as writer:
            for name, value in metrics.items():
                writer.add_scalar('evaluation/' + name, value, iteration)
        print(json.dumps(dict(iteration=iteration, episodes=len(rows), metrics=metrics)), flush=True)
        return result
    finally:
        env.close()


def _run_dev_evaluation(log_dir, iteration):
    output = Path(log_dir) / 'evaluation'
    output.mkdir(exist_ok=True)
    # Serialize checkpoint evaluations; each uses a four-worker CPU pool.
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluate_checkpoint(log_dir, iteration)
