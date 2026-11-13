import os
import torch

import argparse
from collections import defaultdict
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger

from model.policy_value_network import RLFieldsActorModel
from env.multi_robot_env import MultiRobotEnv
from env.params import SceneConfig


def build_ppo_config_multiagent(cfg: SceneConfig) -> PPOConfig:
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
            # 手动推进主进程中的评测环境，不创建远程采样进程。
            num_env_runners=0,
            num_envs_per_env_runner=1,
            # 评估不需要太多 CPU；你也可以从命令行覆盖
            num_cpus_per_env_runner=20,
            num_gpus_per_env_runner=0,
            rollout_fragment_length=cfg.experiment_params.rollout_fragment_length,
            sample_timeout_s=None,
        )
        .training(
            model={
                "custom_model": "rlfields_actor",
                "custom_model_config": cfg.get_network_config(),
            },
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(num_gpus=1,
                   num_cpus_for_main_process=cfg.experiment_params.num_cpus_for_main_process,
        )
    )
    return config


def reset_compat(env):
    out = env.reset()
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})


def step_compat(env, action_dict):
    out = env.step(action_dict)
    if isinstance(out, tuple) and len(out) == 5:
        return out  # obs, rewards, terminateds, truncateds, infos
    if isinstance(out, tuple) and len(out) == 4:
        obs, rewards, dones, infos = out
        terminateds = dones
        truncateds = {k: False for k in dones}
        return obs, rewards, terminateds, truncateds, infos
    raise RuntimeError(f"Unexpected env.step return: {out}")


def _robot_pos(env, agent_id: str) -> np.ndarray:
    # env 是 MultiRobotEnv, env.sim 是 MARL_SIM
    idx = int(agent_id.split("_")[-1])
    return np.asarray(env.sim.robot_list[idx].state[:2].ravel(), dtype=np.float32)


def evaluate(
    algo,
    env,
    policy_id: str = "shared_policy",
    episodes: int = 20,
    max_steps: int = 5000,
    verbose: bool = True,
):
    policy = algo.get_policy(policy_id)

    ep_summaries = []
    # 累计（跨 episode）
    agg_team_success = []
    agg_agent_success_rate = []
    agg_team_path = []
    agg_team_return = []
    agg_team_avg_speed = []
    agg_team_steps_mean = []
    agg_team_steps_max = []
    agg_team_comp = defaultdict(list)  # k -> [ep_sum_k]

    for ep in range(episodes):
        obs, _ = reset_compat(env)

        term_agents = set()
        trunc_agents = set()
        all_agents = set(obs.keys())

        terminateds = {"__all__": False}
        truncateds = {"__all__": False}

        # per-agent accumulators
        agent_comp = defaultdict(lambda: defaultdict(float))  # agent -> (k -> sum)
        agent_path = defaultdict(float)
        agent_arrived = defaultdict(bool)
        agent_timeout = defaultdict(bool)
        agent_collision = defaultdict(bool)
        agent_steps = defaultdict(int)
        agent_speed_sum = defaultdict(float)

        # path: store last position for each seen agent
        last_pos = {aid: _robot_pos(env, aid) for aid in obs.keys()}

        for t in range(max_steps):
            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                break

            action_dict = {}
            for agent_id, o in obs.items():
                all_agents.add(agent_id)
                if (agent_id in term_agents) or (agent_id in trunc_agents):
                    continue
                a, _, _ = policy.compute_single_action(o, explore=False)
                action_dict[agent_id] = a

            if not action_dict:
                # 所有 agent 都 done 了，但 "__all__" 还没被置上（理论上不该发生）
                break

            obs, rewards, terminateds, truncateds, infos = step_compat(env, action_dict)

            # 更新 done 集合
            for agent_id, d in terminateds.items():
                if agent_id != "__all__" and d:
                    term_agents.add(agent_id)
            for agent_id, d in truncateds.items():
                if agent_id != "__all__" and d:
                    trunc_agents.add(agent_id)

            # 累计：对本步有行动的 agent 统计（与你 env.step 的“alive 才 action”一致）
            for agent_id in action_dict.keys():
                agent_steps[agent_id] += 1

                # path increment
                cur_pos = _robot_pos(env, agent_id)
                if agent_id in last_pos:
                    agent_path[agent_id] += float(np.linalg.norm(cur_pos - last_pos[agent_id]))
                last_pos[agent_id] = cur_pos

                info = infos.get(agent_id, {}) or {}

                # reward components: 默认只统计 r_*（你在 env 里已经把各项写进 info 了）
                for k, v in info.items():
                    if not k.startswith("r_"):
                        continue
                    if isinstance(v, (bool, np.bool_)):
                        v = float(v)
                    elif isinstance(v, (int, float, np.number)):
                        v = float(v)
                    else:
                        continue
                    agent_comp[agent_id][k] += v

                # speed: 统计每步速度（如果 info 没有就跳过）
                if "speed" in info and isinstance(info["speed"], (int, float, np.number)):
                    agent_speed_sum[agent_id] += float(info["speed"])

                # success / failure flags
                if info.get("r_arrive", 0.0) > 0.0:
                    agent_arrived[agent_id] = True
                if info.get("r_timeout", 0.0) < 0.0:
                    agent_timeout[agent_id] = True
                if info.get("r_coll_robot", 0.0) < 0.0 or info.get("r_coll_obst", 0.0) < 0.0:
                    agent_collision[agent_id] = True

        # -------- episode summary --------
        agents = sorted(list(all_agents))
        n_agents = max(1, len(agents))

        per_agent = {}
        team_comp_sum = defaultdict(float)
        team_path = 0.0
        team_return = 0.0
        team_steps_sum = 0.0
        team_steps_max = 0.0
        team_speed_sum = 0.0

        for aid in agents:
            comp = dict(agent_comp[aid])
            ret = float(comp.get("r_total", 0.0))
            path = float(agent_path.get(aid, 0.0))
            arrived = bool(agent_arrived.get(aid, False))
            timeout = bool(agent_timeout.get(aid, False))
            collision = bool(agent_collision.get(aid, False))
            steps = int(agent_steps.get(aid, 0))
            avg_speed = float(agent_speed_sum.get(aid, 0.0) / max(steps, 1))

            per_agent[aid] = {
                "arrived": arrived,
                "timeout": timeout,
                "collision": collision,
                "steps": steps,
                "path_len": path,
                "return": ret,
                "avg_speed": avg_speed,
                "reward_components": comp,
            }

            team_path += path
            team_return += ret
            team_steps_sum += steps
            team_steps_max = max(team_steps_max, steps)
            team_speed_sum += agent_speed_sum.get(aid, 0.0)
            for k, v in comp.items():
                team_comp_sum[k] += float(v)

        agent_success_rate = float(np.mean([1.0 if per_agent[aid]["arrived"] else 0.0 for aid in agents])) if agents else 0.0
        team_success = bool(all(per_agent[aid]["arrived"] for aid in agents)) if agents else False
        team_steps_mean_per_agent = float(team_steps_sum / n_agents)
        team_avg_speed = float(team_speed_sum / max(team_steps_sum, 1.0))

        ep_summary = {
            "ep": ep,
            "team_success": team_success,
            "agent_success_rate": agent_success_rate,
            "team_path_len": float(team_path),
            "team_path_mean_per_agent": float(team_path / n_agents),
            "team_return": float(team_return),
            "team_avg_speed": team_avg_speed,
            "team_steps_mean_per_agent": team_steps_mean_per_agent,
            "team_steps_max": float(team_steps_max),
            "team_reward_components_sum": dict(team_comp_sum),
            "per_agent": per_agent,
        }
        ep_summaries.append(ep_summary)

        # 汇总（跨 episode）
        agg_team_success.append(1.0 if team_success else 0.0)
        agg_agent_success_rate.append(agent_success_rate)
        agg_team_path.append(float(team_path))
        agg_team_return.append(float(team_return))
        agg_team_avg_speed.append(float(team_avg_speed))
        agg_team_steps_mean.append(float(team_steps_mean_per_agent))
        agg_team_steps_max.append(float(team_steps_max))
        for k, v in team_comp_sum.items():
            agg_team_comp[k].append(float(v))

        if verbose:
            print(
                f"[eval] ep={ep:03d} team_success={int(team_success)} "
                f"agent_success_rate={agent_success_rate:.3f} "
                f"team_path={team_path:.3f} team_return={team_return:.3f}"
            )

    def _mean_std(xs):
        if not xs:
            return (0.0, 0.0)
        m = float(np.mean(xs))
        s = float(np.std(xs, ddof=0))
        return (m, s)

    summary = {
        "episodes": episodes,
        "team_success_mean": _mean_std(agg_team_success)[0],
        "agent_success_rate_mean": _mean_std(agg_agent_success_rate)[0],
        "team_path_len_mean": _mean_std(agg_team_path)[0],
        "team_return_mean": _mean_std(agg_team_return)[0],
        "team_avg_speed_mean": _mean_std(agg_team_avg_speed)[0],
        "team_steps_mean_per_agent_mean": _mean_std(agg_team_steps_mean)[0],
        "team_steps_max_mean": _mean_std(agg_team_steps_max)[0],
        "team_reward_components_mean": {k: _mean_std(vs)[0] for k, vs in agg_team_comp.items()},
        "team_reward_components_std": {k: _mean_std(vs)[1] for k, vs in agg_team_comp.items()},
    }

    return ep_summaries, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", "--log_dir", required=True, help="训练实验目录")
    parser.add_argument("--iter", type=int, required=True, help="保存的 checkpoint 轮次")
    parser.add_argument("--num-robots", "--num_robots", type=int, default=None, help="评测机器人数量；默认沿用训练配置")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=5000)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--out-json", "--out_json", type=str, default="", help="Optional path to dump episode summaries as json")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.iter, args.episodes, args.max_steps, args.cpus) < 1 or (args.num_robots is not None and args.num_robots < 1):
        parser.error("iter、episodes、max-steps、cpus 和指定的 num-robots 必须为正整数。")
    args.log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    ckpt_path = os.path.join(args.log_dir, 'ckpts', f'iter_{args.iter:06d}')

    cfg = SceneConfig.from_checkpoint(ckpt_path)
    trained_num_robots = cfg.robot_params.number_robots
    if args.num_robots is not None:
        cfg.robot_params.number_robots = args.num_robots
    cfg.experiment_params.disable_plotting = not args.render
    cfg.network_params.device = args.device
    ray.init(num_cpus=args.cpus, log_to_driver=False, ignore_reinit_error=True)
    config = build_ppo_config_multiagent(cfg).env_runners(num_env_runners=0, num_envs_per_env_runner=1, num_cpus_per_env_runner=args.cpus)
    config = config.resources(num_gpus=1 if args.device == "cuda" else 0, num_cpus_for_main_process=args.cpus)
    # 评测结果由脚本输出，不创建默认的 ~/ray_results 日志。
    algo = config.build(logger_creator=lambda c: NoopLogger(c, os.path.abspath(args.log_dir)))
    algo.restore(ckpt_path)
    if algo.iteration != args.iter:
        actual_iteration = algo.iteration
        algo.stop()
        ray.shutdown()
        raise ValueError(f'Checkpoint iteration {actual_iteration} does not match --iter {args.iter}.')

    env = algo.env_runner.env
    policy = algo.get_policy("shared_policy")
    policy.config['env_config'] = cfg.get_env_config()
    policy.config['model']['custom_model_config'] = cfg.get_network_config()
    print(f"[eval] log_dir={args.log_dir} iter={algo.iteration} robots={env.num_robots} trained_robots={trained_num_robots} episodes={args.episodes} max_steps={args.max_steps}")

    ep_summaries, summary = evaluate(algo, env, episodes=args.episodes, max_steps=args.max_steps, verbose=True)

    print("\n===== EVAL SUMMARY =====")
    print(f"Episodes: {summary['episodes']}")
    print(f"Team success(mean): {summary['team_success_mean']:.3f}")
    print(f"Agent success rate(mean): {summary['agent_success_rate_mean']:.3f}")
    print(f"Team path length(mean): {summary['team_path_len_mean']:.3f}")
    print(f"Team return(mean): {summary['team_return_mean']:.3f}")

    print("\nReward components (team sum per episode):")
    keys = sorted(summary["team_reward_components_mean"].keys())
    for k in keys:
        m = summary["team_reward_components_mean"][k]
        s = summary["team_reward_components_std"][k]
        print(f"  {k:20s} mean={m:10.4f} std={s:10.4f}")

    if args.out_json:
        import json
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"episodes": ep_summaries, "summary": summary,
                       "evaluation_config": {"log_dir": args.log_dir, "iteration": algo.iteration,
                                             "training_num_robots": trained_num_robots, "num_robots": env.num_robots,
                                             "episodes": args.episodes, "max_steps": args.max_steps, "device": args.device}},
                      f, ensure_ascii=False, indent=2)
        print(f"\nSaved json to: {args.out_json}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
