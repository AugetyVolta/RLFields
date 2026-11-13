import os
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger

from model.policy_value_network import RLFieldsActorModel
from env.multi_robot_env import MultiRobotEnv
from env.params import SceneConfig


def build_ppo_config_multiagent(cfg: SceneConfig):
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
            num_envs_per_env_runner=cfg.experiment_params.num_envs_per_env_runner,
            num_cpus_per_env_runner=cfg.experiment_params.num_cpus_per_env_runner,
            num_gpus_per_env_runner=0,
            rollout_fragment_length=cfg.experiment_params.rollout_fragment_length,
            sample_timeout_s=None,
        )
        .training(
            model={
                "custom_model": "rlfields_actor",
                "custom_model_config": cfg.get_network_config(),
            },
            lr=cfg.algorithm_params.lr,
            gamma=cfg.algorithm_params.gamma,
            num_epochs=cfg.algorithm_params.num_epochs,
            minibatch_size=cfg.algorithm_params.minibatch_size,
            train_batch_size=cfg.algorithm_params.train_batch_size,
            use_gae=cfg.algorithm_params.use_gae,
            lambda_=cfg.algorithm_params.lambda_,
            use_critic=cfg.algorithm_params.use_critic,
            use_kl_loss=cfg.algorithm_params.use_kl_loss,
            kl_coeff=cfg.algorithm_params.kl_coeff,
            kl_target=cfg.algorithm_params.kl_target,
            entropy_coeff=cfg.algorithm_params.entropy_coeff,
            vf_loss_coeff=cfg.algorithm_params.vf_loss_coeff,
            vf_clip_param=cfg.algorithm_params.vf_clip_param,
            clip_param=cfg.algorithm_params.clip_param,
            grad_clip=cfg.algorithm_params.grad_clip,
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(
            num_gpus=1,
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", "--log_dir", required=True, help="训练实验目录")
    parser.add_argument("--iter", type=int, required=True, help="保存的 checkpoint 轮次")
    parser.add_argument("--num-robots", "--num_robots", type=int, default=None, help="评测机器人数量；默认沿用训练配置")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=5000)
    parser.add_argument("--cpus", type=int, default=8)
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
    config = build_ppo_config_multiagent(cfg)

    config = (
        config
        .env_runners(num_env_runners=0,
                     num_envs_per_env_runner=1,
                     num_cpus_per_env_runner=args.cpus)
    )

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

    episodes = args.episodes
    max_steps = args.max_steps
    print(f"[infer] log_dir={args.log_dir} iter={algo.iteration} robots={env.num_robots} trained_robots={trained_num_robots} episodes={episodes} max_steps={max_steps}")

    for ep in range(episodes):
        obs, info = reset_compat(env)

        term_agents = set()   # 结束
        trunc_agents = set()  # 超时
        all_agents = set(obs.keys())  # 统计全集（如果后续会新增 agent id，可再并进去）

        terminateds = {"__all__": False}
        truncateds  = {"__all__": False}
        ep_ret = 0.0

        for t in range(max_steps):
            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                break

            # 只对“活着”的 agent 算动作
            action_dict = {}
            for agent_id, o in obs.items():
                all_agents.add(agent_id)
                if (agent_id in term_agents) or (agent_id in trunc_agents):
                    continue
                a, _, _ = policy.compute_single_action(o, explore=False)
                action_dict[agent_id] = a

            obs, rewards, terminateds, truncateds, infos = step_compat(env, action_dict)

            # 累计统计：把本步 newly done/timeout 的 agent 放入集合（集合自动去重）
            for agent_id, d in terminateds.items():
                if agent_id != "__all__" and d:
                    term_agents.add(agent_id)
            for agent_id, d in truncateds.items():
                if agent_id != "__all__" and d:
                    trunc_agents.add(agent_id)

            if isinstance(rewards, dict):
                ep_ret += float(sum(rewards.values()))
            else:
                ep_ret += float(rewards)

        still_active = len(all_agents - term_agents - trunc_agents)
        print(
            f"[infer] ep={ep} return={ep_ret:.3f} "
            f"terminated={len(term_agents)} truncated={len(trunc_agents)} active={still_active}"
        )

    algo.stop()
    ray.shutdown()
