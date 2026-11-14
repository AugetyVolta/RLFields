import os
import torch

import argparse
from collections import defaultdict, deque
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger

from model.policy_value_network import RLFieldsActorModel
from env.multi_robot_env import MultiRobotEnv
from env.params import SceneConfig

def build_ppo_config_multiagent(
    cfg: SceneConfig, *, num_env_runners=0, num_envs_per_env_runner=1,
    num_cpus_per_env_runner=1, num_gpus_per_env_runner=0,
    rollout_fragment_length=None, sample_timeout_s=None,
) -> PPOConfig:
    ModelCatalog.register_custom_model("rlfields_actor", RLFieldsActorModel)
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=MultiRobotEnv, env_config=cfg.get_env_config())
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            create_env_on_local_worker=num_env_runners == 0,
            rollout_fragment_length=(cfg.experiment_params.rollout_fragment_length
                                     if rollout_fragment_length is None else rollout_fragment_length),
            sample_timeout_s=sample_timeout_s,
        )
        .training(model={"custom_model": "rlfields_actor",
                         "custom_model_config": cfg.get_network_config()})
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(num_gpus=1 if str(cfg.network_params.device).startswith("cuda") else 0,
                   num_cpus_for_main_process=cfg.experiment_params.num_cpus_for_main_process)
    )
    # PPO only restores policies here. Align its unused training-batch setting
    # so RLlib accepts arbitrary evaluation fragment/environment counts.
    if config.rollout_fragment_length != "auto":
        fragment = max(num_env_runners, 1) * num_envs_per_env_runner * config.rollout_fragment_length
        batch_size = config.train_batch_size
        config.training(train_batch_size=((batch_size + fragment - 1) // fragment) * fragment)
    return config

def reset_compat(env):
    out = env.reset()
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})

def step_compat(env, action_dict):
    out = env.step(action_dict)
    if isinstance(out, tuple) and len(out) == 5:
        return out
    if isinstance(out, tuple) and len(out) == 4:
        obs, rewards, dones, infos = out
        return obs, rewards, dones, {k: False for k in dones}, infos
    raise RuntimeError(f"Unexpected env.step return: {out}")

def _robot_pos(env, agent_id: str) -> np.ndarray:
    idx = int(agent_id.split("_")[-1])
    return np.asarray(env.sim.robot_list[idx].state[:2].ravel(), dtype=np.float32)

class _EpisodeMetrics:
    def __init__(self, env, episode_id, max_steps):
        self.env, self.episode_id, self.max_steps = env, episode_id, max_steps
        self.obs, _ = reset_compat(env)
        self.term_agents, self.trunc_agents = set(), set()
        self.all_agents = set(self.obs)
        self.terminateds = self.truncateds = {"__all__": False}
        self.steps = 0
        self.agent_comp = defaultdict(lambda: defaultdict(float))
        self.agent_path = defaultdict(float)
        self.agent_arrived = defaultdict(bool)
        self.agent_timeout = defaultdict(bool)
        self.agent_collision = defaultdict(bool)
        self.agent_steps = defaultdict(int)
        self.agent_speed_sum = defaultdict(float)
        self.last_pos = {aid: _robot_pos(env, aid) for aid in self.obs}

    def observations(self):
        self.all_agents.update(self.obs)
        return [(aid, obs) for aid, obs in self.obs.items()
                if aid not in self.term_agents and aid not in self.trunc_agents]

    def step(self, actions):
        self.obs, _, self.terminateds, self.truncateds, infos = step_compat(self.env, actions)
        self.steps += 1
        self.term_agents.update(aid for aid, done in self.terminateds.items() if aid != "__all__" and done)
        self.trunc_agents.update(aid for aid, done in self.truncateds.items() if aid != "__all__" and done)
        for aid in actions:
            self.agent_steps[aid] += 1
            cur_pos = _robot_pos(self.env, aid)
            if aid in self.last_pos:
                self.agent_path[aid] += float(np.linalg.norm(cur_pos - self.last_pos[aid]))
            self.last_pos[aid] = cur_pos
            info = infos.get(aid, {}) or {}
            for key, value in info.items():
                if key.startswith("r_") and isinstance(value, (bool, np.bool_, int, float, np.number)):
                    self.agent_comp[aid][key] += float(value)
            if "speed" in info and isinstance(info["speed"], (int, float, np.number)):
                self.agent_speed_sum[aid] += float(info["speed"])
            self.agent_arrived[aid] |= info.get("r_arrive", 0.0) > 0.0
            self.agent_timeout[aid] |= info.get("r_timeout", 0.0) < 0.0
            self.agent_collision[aid] |= info.get("r_coll_robot", 0.0) < 0.0 or info.get("r_coll_obst", 0.0) < 0.0

    def done(self):
        return (self.terminateds.get("__all__", False) or self.truncateds.get("__all__", False)
                or self.steps >= self.max_steps or not self.observations())

    def summary(self):
        agents = sorted(self.all_agents)
        n_agents = max(1, len(agents))
        per_agent = {}
        team_comp_sum = defaultdict(float)
        team_path = team_return = team_steps_sum = team_steps_max = team_speed_sum = 0.0
        for aid in agents:
            comp = dict(self.agent_comp[aid])
            ret = float(comp.get("r_total", 0.0))
            path = float(self.agent_path.get(aid, 0.0))
            steps = int(self.agent_steps.get(aid, 0))
            per_agent[aid] = {
                "arrived": bool(self.agent_arrived.get(aid, False)),
                "timeout": bool(self.agent_timeout.get(aid, False)),
                "collision": bool(self.agent_collision.get(aid, False)),
                "steps": steps, "path_len": path, "return": ret,
                "avg_speed": float(self.agent_speed_sum.get(aid, 0.0) / max(steps, 1)),
                "reward_components": comp,
            }
            team_path += path
            team_return += ret
            team_steps_sum += steps
            team_steps_max = max(team_steps_max, steps)
            team_speed_sum += self.agent_speed_sum.get(aid, 0.0)
            for key, value in comp.items():
                team_comp_sum[key] += float(value)
        return {
            "ep": self.episode_id,
            "team_success": bool(all(per_agent[aid]["arrived"] for aid in agents)) if agents else False,
            "agent_success_rate": float(np.mean([float(per_agent[aid]["arrived"]) for aid in agents])) if agents else 0.0,
            "team_path_len": float(team_path),
            "team_path_mean_per_agent": float(team_path / n_agents),
            "team_return": float(team_return),
            "team_avg_speed": float(team_speed_sum / max(team_steps_sum, 1.0)),
            "team_steps_mean_per_agent": float(team_steps_sum / n_agents),
            "team_steps_max": float(team_steps_max),
            "team_reward_components_sum": dict(team_comp_sum), "per_agent": per_agent,
        }

class _MetricsEvaluator:
    def __init__(self, policy, envs, episode_ids, max_steps, verbose):
        if not envs:
            raise RuntimeError("Evaluation runner has no environments")
        self.policy, self.envs = policy, envs
        self.pending = deque(episode_ids)
        self.max_steps, self.verbose = max_steps, verbose
        self.slots = [self._next(env) for env in envs]

    def _next(self, env):
        return _EpisodeMetrics(env, self.pending.popleft(), self.max_steps) if self.pending else None

    def advance(self, fragment_length):
        completed = []
        for _ in range(fragment_length):
            items = [(index, aid, obs) for index, state in enumerate(self.slots) if state is not None
                     for aid, obs in state.observations()]
            if not any(state is not None for state in self.slots):
                break
            commands = [{} for _ in self.slots]
            if items:
                actions, _, _ = self.policy.compute_actions(np.stack([item[2] for item in items]), explore=False)
                for (index, aid, _), action in zip(items, actions):
                    commands[index][aid] = action
            for index, state in enumerate(self.slots):
                if state is None:
                    continue
                if commands[index]:
                    state.step(commands[index])
                if state.done():
                    row = state.summary()
                    completed.append(row)
                    if self.verbose:
                        print(f"[eval] ep={row['ep']:03d} team_success={int(row['team_success'])} "
                              f"agent_success_rate={row['agent_success_rate']:.3f} "
                              f"team_path={row['team_path_len']:.3f} team_return={row['team_return']:.3f}", flush=True)
                    self.slots[index] = self._next(self.envs[index])
        return completed, not any(state is not None for state in self.slots)

def _summarize(rows):
    def mean(values):
        return float(np.mean(values)) if values else 0.0
    components = defaultdict(list)
    for row in rows:
        for key, value in row["team_reward_components_sum"].items():
            components[key].append(value)
    return {
        "episodes": len(rows),
        "team_success_mean": mean([float(row["team_success"]) for row in rows]),
        "agent_success_rate_mean": mean([row["agent_success_rate"] for row in rows]),
        "team_path_len_mean": mean([row["team_path_len"] for row in rows]),
        "team_return_mean": mean([row["team_return"] for row in rows]),
        "team_avg_speed_mean": mean([row["team_avg_speed"] for row in rows]),
        "team_steps_mean_per_agent_mean": mean([row["team_steps_mean_per_agent"] for row in rows]),
        "team_steps_max_mean": mean([row["team_steps_max"] for row in rows]),
        "team_reward_components_mean": {key: mean(values) for key, values in components.items()},
        "team_reward_components_std": {key: float(np.std(values, ddof=0)) for key, values in components.items()},
    }

def evaluate(algo, env, policy_id="shared_policy", episodes=20, max_steps=5000, verbose=True):
    """Keep the existing single-environment Python interface."""
    evaluator = _MetricsEvaluator(algo.get_policy(policy_id), [env], range(episodes), max_steps, verbose)
    rows, done = [], False
    while not done:
        completed, done = evaluator.advance(150)
        rows.extend(completed)
    return rows, _summarize(rows)

def evaluate_runners(algo, cfg, *, episodes, max_steps, num_env_runners, sample_timeout_s=None, verbose=True):
    """Run complete episodes in parallel; fragment length controls RPC chunks."""
    indices = list(range(1, num_env_runners + 1)) if num_env_runners else [0]
    assignments = {index: list(range(episodes))[offset::len(indices)] for offset, index in enumerate(indices)}
    local = num_env_runners == 0
    env_config, model_config = cfg.get_env_config(), cfg.get_network_config()

    def setup(worker):
        torch.set_num_threads(1)
        policy = worker.get_policy("shared_policy")
        policy.config["env_config"] = env_config
        policy.config["model"]["custom_model_config"] = model_config
        envs = worker.foreach_env(lambda env: env)
        worker._navigation_metrics_evaluator = _MetricsEvaluator(
            policy, envs, assignments[worker.worker_index], max_steps, verbose=False)
        return dict(worker_index=worker.worker_index, pid=os.getpid(), environments=len(envs))

    def advance(worker):
        fragment = worker.config.get_rollout_fragment_length(worker_index=worker.worker_index)
        rows, done = worker._navigation_metrics_evaluator.advance(fragment)
        return dict(worker_index=worker.worker_index, rows=rows, done=done)

    group = algo.env_runner_group
    runner_info = group.foreach_env_runner(setup, local_env_runner=local, timeout_seconds=sample_timeout_s)
    if {row["worker_index"] for row in runner_info} != set(indices):
        raise TimeoutError("Evaluation runner setup timed out or a runner failed")
    rows, done = [], False
    while not done:
        results = group.foreach_env_runner(advance, local_env_runner=local, timeout_seconds=sample_timeout_s)
        if {row["worker_index"] for row in results} != set(indices):
            raise TimeoutError("Evaluation chunk timed out or a runner failed; incomplete results were not summarized")
        for result in results:
            rows.extend(result["rows"])
            if verbose:
                for row in result["rows"]:
                    print(f"[eval] runner={result['worker_index']} ep={row['ep']:03d} "
                          f"team_success={int(row['team_success'])} agent_success_rate={row['agent_success_rate']:.3f}", flush=True)
        done = all(result["done"] for result in results)
    if len(rows) != episodes or {row["ep"] for row in rows} != set(range(episodes)):
        raise RuntimeError("Evaluation episodes are missing or duplicated")
    rows.sort(key=lambda row: row["ep"])
    return rows, _summarize(rows), runner_info

def _fragment_length(value):
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("rollout-fragment-length must be positive or auto")
    return parsed

def _timeout(value):
    if value.lower() in ("none", "null"):
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample-timeout-s must be positive or none")
    return parsed

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", "--log_dir", required=True, help="训练实验目录")
    parser.add_argument("--iter", type=int, required=True, help="保存的 checkpoint 轮次")
    parser.add_argument("--num-robots", "--num_robots", type=int, default=None, help="评测机器人数量；默认沿用训练配置")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=5000)
    parser.add_argument("--cpus", type=int, default=8, help="本次 Ray 实例可调度的总 CPU 数")
    parser.add_argument("--num-env-runners", "--num_env_runners", type=int, default=0, help="并行评测进程数；0 为本地评测")
    parser.add_argument("--num-envs-per-env-runner", "--num_envs_per_env_runner", type=int, default=1, help="每个 runner 的环境数，合批推理、同步推进")
    parser.add_argument("--num-cpus-per-env-runner", "--num_cpus_per_env_runner", type=float, default=1, help="每个远程 runner 的逻辑 CPU 资源")
    parser.add_argument("--num-gpus-per-env-runner", "--num_gpus_per_env_runner", type=float, default=0, help="每个远程 runner 的 GPU 资源，可为小数")
    parser.add_argument("--rollout-fragment-length", "--rollout_fragment_length", type=_fragment_length, default=None, help="每批推进的环境步数；默认沿用 checkpoint 配置，支持 auto")
    parser.add_argument("--sample-timeout-s", "--sample_timeout_s", type=_timeout, default=None, help="每批远程结果的等待秒数；none 为不限时，本地模式不适用")
    parser.add_argument("--out-json", "--out_json", type=str, default="", help="评测结果 JSON 路径")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="远程 runner 默认 CPU；申请 GPU 后默认 CUDA，本地沿用自动选择")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if min(args.iter, args.episodes, args.max_steps, args.cpus, args.num_envs_per_env_runner) < 1 or (args.num_robots is not None and args.num_robots < 1):
        parser.error("iter、episodes、max-steps、cpus、每 runner 环境数和指定的机器人数量必须为正整数。")
    if min(args.num_env_runners, args.num_cpus_per_env_runner, args.num_gpus_per_env_runner) < 0:
        parser.error("runner 数量与每 runner 的 CPU/GPU 资源不得为负数。")
    if args.num_env_runners * args.num_cpus_per_env_runner > args.cpus:
        parser.error("远程 runner 的 CPU 资源总和超过 --cpus。")
    if args.render and (args.num_env_runners or args.num_envs_per_env_runner != 1):
        parser.error("--render 需使用本地单环境：--num-env-runners 0 --num-envs-per-env-runner 1。")
    if args.device is None:
        args.device = ("cuda" if args.num_gpus_per_env_runner > 0 else "cpu") if args.num_env_runners else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_env_runners and args.device == "cuda" and args.num_gpus_per_env_runner == 0:
        parser.error("CUDA 并行评测需设置 --num-gpus-per-env-runner 为正数。")
    args.log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    ckpt_path = os.path.join(args.log_dir, 'ckpts', f'iter_{args.iter:06d}')
    cfg = SceneConfig.from_checkpoint(ckpt_path)
    trained_num_robots = cfg.robot_params.number_robots
    if args.num_robots is not None:
        cfg.robot_params.number_robots = args.num_robots
    cfg.experiment_params.disable_plotting = not args.render
    cfg.network_params.device = args.device
    torch.set_num_threads(1)
    ray.init(address='local', num_cpus=args.cpus, log_to_driver=False, include_dashboard=False,
             ignore_reinit_error=True,
             **({'num_gpus': 0} if args.device == 'cpu' and args.num_gpus_per_env_runner == 0 else {}))
    algo = None
    try:
        if args.num_env_runners * args.num_gpus_per_env_runner > ray.cluster_resources().get('GPU', 0):
            raise ValueError('Requested runner GPU resources exceed the GPUs available to this Ray instance.')
        config = build_ppo_config_multiagent(
            cfg, num_env_runners=args.num_env_runners, num_envs_per_env_runner=args.num_envs_per_env_runner,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner, num_gpus_per_env_runner=args.num_gpus_per_env_runner,
            rollout_fragment_length=args.rollout_fragment_length, sample_timeout_s=args.sample_timeout_s)
        config = config.resources(num_cpus_for_main_process=args.cpus)
        algo = config.build(logger_creator=lambda c: NoopLogger(c, args.log_dir))
        algo.restore(ckpt_path)
        if algo.iteration != args.iter:
            raise ValueError(f'Checkpoint iteration {algo.iteration} does not match --iter {args.iter}.')
        print(f"[eval] log_dir={args.log_dir} iter={algo.iteration} robots={cfg.robot_params.number_robots} "
              f"trained_robots={trained_num_robots} episodes={args.episodes} max_steps={args.max_steps} "
              f"runners={args.num_env_runners} envs_per_runner={args.num_envs_per_env_runner}", flush=True)
        ep_summaries, summary, runner_info = evaluate_runners(
            algo, cfg, episodes=args.episodes, max_steps=args.max_steps,
            num_env_runners=args.num_env_runners, sample_timeout_s=args.sample_timeout_s)
        print("\n===== EVAL SUMMARY =====")
        print(f"Episodes: {summary['episodes']}")
        print(f"Team success(mean): {summary['team_success_mean']:.3f}")
        print(f"Agent success rate(mean): {summary['agent_success_rate_mean']:.3f}")
        print(f"Team path length(mean): {summary['team_path_len_mean']:.3f}")
        print(f"Team return(mean): {summary['team_return_mean']:.3f}")
        print("\nReward components (team sum per episode):")
        for key in sorted(summary["team_reward_components_mean"]):
            print(f"  {key:20s} mean={summary['team_reward_components_mean'][key]:10.4f} "
                  f"std={summary['team_reward_components_std'][key]:10.4f}")
        if args.out_json:
            import json
            with open(args.out_json, 'w', encoding='utf-8') as stream:
                json.dump({"episodes": ep_summaries, "summary": summary,
                           "evaluation_config": {
                               "log_dir": args.log_dir, "iteration": algo.iteration,
                               "training_num_robots": trained_num_robots, "num_robots": cfg.robot_params.number_robots,
                               "episodes": args.episodes, "max_steps": args.max_steps, "device": args.device,
                               "num_env_runners": args.num_env_runners, "num_envs_per_env_runner": args.num_envs_per_env_runner,
                               "num_cpus_per_env_runner": args.num_cpus_per_env_runner,
                               "num_gpus_per_env_runner": args.num_gpus_per_env_runner,
                               "rollout_fragment_length": config.rollout_fragment_length,
                               "sample_timeout_s": args.sample_timeout_s, "runners": runner_info}},
                          stream, ensure_ascii=False, indent=2)
            print(f"\nSaved json to: {args.out_json}")
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
