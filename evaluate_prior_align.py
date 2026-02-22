import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from model.policy_value_network_prior_align import RLFieldsActorModelPriorAlign
from env.multi_robot_env_prior_align import MultiRobotEnvPriorAlign
from env.params import SceneConfig


def build_ppo_config_multiagent(cfg: SceneConfig):
    ModelCatalog.register_custom_model("rlfields_actor_prior", RLFieldsActorModelPriorAlign)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=MultiRobotEnvPriorAlign, env_config=cfg.get_env_config())
        .framework("torch")
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=10,
            num_gpus_per_env_runner=1,
            rollout_fragment_length=cfg.experiment_params.rollout_fragment_length,
            sample_timeout_s=None,
        )
        .training(
            model={
                "custom_model": "rlfields_actor_prior",
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
    ckpt_path = "/home/zby/Desktop/iter_000077"

    ray.init(log_to_driver=False, ignore_reinit_error=True)

    cfg = SceneConfig()
    config = build_ppo_config_multiagent(cfg)

    config = (
        config
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
    )

    algo = config.build()
    algo.restore(ckpt_path)

    env = algo.env_runner.env
    policy = algo.get_policy("shared_policy")

    episodes = 2
    max_steps = 5000

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
