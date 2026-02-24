import argparse
import json
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

# 复用你原评估脚本里的指标计算逻辑（保证指标定义完全一致）
from evaluate_metrics import evaluate

from env.params import SceneConfig

# 最终版基线 env / model（prior-align）
from env.multi_robot_env_prior_align import MultiRobotEnvPriorAlign
from model.policy_value_network_prior_align import RLFieldsActorModelPriorAlign

# 消融 env / model（prior-align 基线）
from env.multi_robot_env_reward_ablations import (
    MultiRobotEnv_PriorAlign_NoPriorAlignReward,
    MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward,
)
from model.policy_value_network_ablations import (
    RLFieldsActorModel_NoPCFeaturesPriorAlign,
    RLFieldsActorModel_NoPriorDir,
    RLFieldsActorModel_NoPCFeaturesNoPriorDir,
)


def build_ppo_config_multiagent(cfg: SceneConfig, ablation: str) -> PPOConfig:
    """
    构建与 train_ablations.py 对应的评估配置。
    指标计算仍直接调用 evaluate_metrics.evaluate（与你上传文件一致）。
    """

    if ablation == "orig":
        env_cls = MultiRobotEnvPriorAlign
        model_cls = RLFieldsActorModelPriorAlign
        model_name = "rlfields_actor_prior_eval_full"
    elif ablation == "ab1":
        # 动作 direct + 去 prior-align reward + 不做 prior 旋转
        env_cls = MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModel_NoPriorDir
        model_name = "rlfields_actor_ab1_noprior_eval"
    elif ablation == "ab2":
        # 不使用 PC 特征
        env_cls = MultiRobotEnvPriorAlign
        model_cls = RLFieldsActorModel_NoPCFeaturesPriorAlign
        model_name = "rlfields_actor_prior_ab2_eval"
    elif ablation == "ab3":
        # 动作 direct + 去 prior-align reward + 不做 prior 旋转 + 不使用 PC 特征
        env_cls = MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModel_NoPCFeaturesNoPriorDir
        model_name = "rlfields_actor_ab3_noprior_nopc_eval"
    elif ablation == "r0":
        # reward-only: 去 prior-align reward + 不做 prior 旋转
        env_cls = MultiRobotEnv_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModelPriorAlign
        model_name = "rlfields_actor_prior_r0_eval"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    # 注册 custom model
    ModelCatalog.register_custom_model(model_name, model_cls)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_cls, env_config=cfg.get_env_config())
        .framework("torch")
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            # 评估时 CPU 可从命令行覆盖
            num_cpus_per_env_runner=20,
            num_gpus_per_env_runner=cfg.experiment_params.num_gpus_per_env_runner,
            rollout_fragment_length=cfg.experiment_params.rollout_fragment_length,
            sample_timeout_s=None,
        )
        .training(
            model={
                "custom_model": model_name,
                "custom_model_config": cfg.get_network_config(),
            },
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="RLlib checkpoint dir (algo.restore expects this path)")
    parser.add_argument("--ablation", type=str, default="orig",
                        choices=["orig", "ab1", "ab2", "ab3", "r0"],
                        help="选择与你 checkpoint 对应的实验设置")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="可选：评估前设置 Ray/环境随机种子（若你的环境支持）")
    parser.add_argument("--out_json", type=str, default="", help="Optional path to dump episode summaries as json")
    args = parser.parse_args()

    ray.init(log_to_driver=False, ignore_reinit_error=True)

    cfg = SceneConfig()
    if args.seed is not None:
        try:
            cfg.experiment_params.seed = args.seed
        except Exception:
            pass

    config = build_ppo_config_multiagent(cfg, args.ablation).env_runners(
        num_env_runners=1,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=args.cpus,
    )
    algo = config.build()
    algo.restore(args.ckpt)

    # 与你原 evaluate_metrics.py 保持一致的访问方式
    env = algo.env_runner.env

    ep_summaries, summary = evaluate(
        algo, env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=True
    )

    print("\n===== EVAL SUMMARY =====")
    print(f"Ablation: {args.ablation}")
    print(f"Episodes: {summary['episodes']}")
    print(f"Team success(mean): {summary['team_success_mean']:.3f}")
    print(f"Agent success rate(mean): {summary['agent_success_rate_mean']:.3f}")
    print(f"Team path length(mean): {summary['team_path_len_mean']:.3f}")
    print(f"Team return(mean): {summary['team_return_mean']:.3f}")
    print(f"Team avg speed(mean): {summary.get('team_avg_speed_mean', 0.0):.3f}")
    print(f"Team steps mean/agent(mean): {summary.get('team_steps_mean_per_agent_mean', 0.0):.3f}")
    print(f"Team steps max(mean): {summary.get('team_steps_max_mean', 0.0):.3f}")

    print("\nReward components (team sum per episode):")
    keys = sorted(summary["team_reward_components_mean"].keys())
    for k in keys:
        m = summary["team_reward_components_mean"][k]
        s = summary["team_reward_components_std"][k]
        print(f"  {k:20s} mean={m:10.4f} std={s:10.4f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({
                "ablation": args.ablation,
                "episodes": ep_summaries,
                "summary": summary
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved json to: {args.out_json}")


if __name__ == "__main__":
    main()
