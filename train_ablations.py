import os
import shutil
import argparse
from datetime import datetime

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger

from env.params import SceneConfig

# 最终版基线 env / model（prior-align）
from env.multi_robot_env_prior_align import MultiRobotEnvPriorAlign
from model.policy_value_network_prior_align import RLFieldsActorModelPriorAlign

# Ablation env / model（prior-align 基线上的消融）
from env.multi_robot_env_reward_ablations import (
    MultiRobotEnv_PriorAlign_NoPriorAlignReward,
    MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward,
)
from model.policy_value_network_ablations import (
    RLFieldsActorModel_NoPCFeaturesPriorAlign,
    RLFieldsActorModel_NoPriorDir,
    RLFieldsActorModel_NoPCFeaturesNoPriorDir,
)


def get_experiment_name(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def make_logger_creator(logdir: str):
    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)
    return logger_creator


def build_ppo_config_multiagent(cfg: SceneConfig, ablation: str):
    # --- choose env/model ---
    if ablation == "orig":
        # 最终版 full model（SDF/neighbor + prior-align）
        env_cls = MultiRobotEnvPriorAlign
        model_cls = RLFieldsActorModelPriorAlign
        model_name = "rlfields_actor_prior_full"

    elif ablation == "ab1":  # direct action + remove prior-align reward + no prior rotation
        env_cls = MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModel_NoPriorDir
        model_name = "rlfields_actor_ab1_noprior"

    elif ablation == "ab2":  # no PC-planner features (prior-align trunk 保留)
        env_cls = MultiRobotEnvPriorAlign
        model_cls = RLFieldsActorModel_NoPCFeaturesPriorAlign
        model_name = "rlfields_actor_prior_ab2_nopc_feat"

    elif ablation == "ab3":  # ab1 + ab2 (其中 ab1 部分不做 prior 旋转)
        env_cls = MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModel_NoPCFeaturesNoPriorDir
        model_name = "rlfields_actor_ab3_noprior_nopc"

    elif ablation == "r0":   # reward-only: remove prior-align reward + no prior rotation
        env_cls = MultiRobotEnv_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModelPriorAlign
        model_name = "rlfields_actor_prior_r0_no_prior_align_reward"

    else:
        raise ValueError(f"unknown ablation: {ablation}")

    # 注册自定义 ModelV2（每次运行用不同名字，避免重复覆盖）
    ModelCatalog.register_custom_model(model_name, model_cls)

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=env_cls, env_config=cfg.get_env_config())
        .framework("torch")
        .env_runners(
            num_env_runners=cfg.experiment_params.num_env_runners,
            num_envs_per_env_runner=cfg.experiment_params.num_envs_per_env_runner,
            num_cpus_per_env_runner=cfg.experiment_params.num_cpus_per_env_runner,
            num_gpus_per_env_runner=cfg.experiment_params.num_gpus_per_env_runner,
            rollout_fragment_length=cfg.experiment_params.rollout_fragment_length,
            sample_timeout_s=None,
        )
        .training(
            model={"custom_model": model_name, "custom_model_config": cfg.get_network_config()},
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


def prepare_exp(cfg: SceneConfig, ablation: str):
    exp_name = get_experiment_name(prefix=f"multi_robot_ppo_{ablation}")
    exp_dir = os.path.join("logs", exp_name)
    ckpt_dir = os.path.join(exp_dir, "ckpts")
    tb_dir = os.path.join(exp_dir, "tensorboard")

    os.makedirs("logs", exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    cfg.experiment_params.exp_dir = exp_dir
    cfg.dump(os.path.join(exp_dir, "scene_config.yaml"))
    shutil.copyfile(cfg.experiment_params.env_yaml_path, os.path.join(exp_dir, "env.yaml"))
    shutil.copyfile(cfg.network_params.nt_pretrained_path, os.path.join(exp_dir, "ntfields.pt"))

    ray.init(log_to_driver=True, logging_level="ERROR", _temp_dir='/mnt/nas_9/group/zhouboyang/tmp')

    algo = build_ppo_config_multiagent(cfg, ablation).build(logger_creator=make_logger_creator(tb_dir))

    exp_info = {"exp_name": exp_name, "exp_dir": exp_dir, "ckpt_dir": ckpt_dir, "tb_dir": tb_dir}
    return algo, exp_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="orig",
                        choices=["orig", "ab1", "ab2", "ab3", "r0"])
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    cfg = SceneConfig()
    algo, exp_info = prepare_exp(cfg, ablation=args.ablation)

    for it in range(args.iters):
        result = algo.train()
        print(f"[{exp_info['exp_name']}] iter={it}", result)

        iter_ckpt_root = os.path.join(exp_info["ckpt_dir"], f"iter_{it + 1:06d}")
        algo.save(iter_ckpt_root)
        print(f"Checkpoint saved at {iter_ckpt_root}")

    final_root = os.path.join(exp_info["ckpt_dir"], "final")
    final_ckpt = algo.save(final_root)
    print(f"Final checkpoint saved at {final_ckpt}")
