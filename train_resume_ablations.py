import os
import re
import shutil
import argparse
from datetime import datetime

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger

from env.params import SceneConfig

# Baseline env / model
from env.multi_robot_env_prior_align import MultiRobotEnvPriorAlign
from model.policy_value_network_prior_align import RLFieldsActorModelPriorAlign

# Ablation env / model
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


def _choose_env_model(ablation: str):
    # Keep mapping consistent with train_ablations.py
    if ablation == "orig":
        env_cls = MultiRobotEnvPriorAlign
        model_cls = RLFieldsActorModelPriorAlign
        model_name = "rlfields_actor_prior_full"
    elif ablation == "ab1":
        env_cls = MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModel_NoPriorDir
        model_name = "rlfields_actor_ab1_noprior"
    elif ablation == "ab2":
        env_cls = MultiRobotEnvPriorAlign
        model_cls = RLFieldsActorModel_NoPCFeaturesPriorAlign
        model_name = "rlfields_actor_prior_ab2_nopc_feat"
    elif ablation == "ab3":
        env_cls = MultiRobotEnv_ActionDirect_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModel_NoPCFeaturesNoPriorDir
        model_name = "rlfields_actor_ab3_noprior_nopc"
    elif ablation == "r0":
        env_cls = MultiRobotEnv_PriorAlign_NoPriorAlignReward
        model_cls = RLFieldsActorModelPriorAlign
        model_name = "rlfields_actor_prior_r0_no_prior_align_reward"
    else:
        raise ValueError(f"unknown ablation: {ablation}")
    return env_cls, model_cls, model_name


def build_ppo_config_multiagent(cfg: SceneConfig, ablation: str):
    env_cls, model_cls, model_name = _choose_env_model(ablation)
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


def infer_resume_it_from_ckpt_path(restore_ckpt_path: str):
    # Support: .../iter_000019
    # Also support when iter token appears before trailing segments.
    path = restore_ckpt_path.rstrip("/")
    m = re.search(r"iter_(\d+)$", path)
    if m:
        return int(m.group(1))
    m = re.search(r"iter_(\d+)", path)
    if m:
        return int(m.group(1))
    return None


def prepare_exp(cfg: SceneConfig, ablation: str, resume_exp_name: str | None = None, ray_temp_dir: str | None = None):
    prefix = f"multi_robot_ppo_{ablation}"
    exp_name = resume_exp_name if resume_exp_name is not None else get_experiment_name(prefix=prefix)
    exp_dir = os.path.join("logs", exp_name)
    ckpt_dir = os.path.join(exp_dir, "ckpts")
    tb_dir = os.path.join(exp_dir, "tensorboard")

    os.makedirs("logs", exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    cfg.experiment_params.exp_dir = exp_dir
    cfg.dump(os.path.join(exp_dir, "scene_config_resume.yaml"))

    try:
        shutil.copyfile(cfg.experiment_params.env_yaml_path, os.path.join(exp_dir, "env.yaml"))
    except Exception as e:
        print(f"[warn] copy env yaml failed: {e}")

    try:
        shutil.copyfile(cfg.network_params.nt_pretrained_path, os.path.join(exp_dir, "ntfields.pt"))
    except Exception as e:
        print(f"[warn] copy ntfields ckpt failed: {e}")

    ray_init_kwargs = dict(log_to_driver=True, logging_level="ERROR", _temp_dir="/mnt/nas_9/group/zhouboyang/tmp")
    if ray_temp_dir:
        ray_init_kwargs["_temp_dir"] = ray_temp_dir
    ray.init(**ray_init_kwargs)

    algo = build_ppo_config_multiagent(cfg, ablation).build(logger_creator=make_logger_creator(tb_dir))
    exp_info = {
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "ckpt_dir": ckpt_dir,
        "tb_dir": tb_dir,
    }
    return algo, exp_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, required=True, choices=["orig", "ab1", "ab2", "ab3", "r0"])
    parser.add_argument("--restore_ckpt_path", type=str, required=True, help="例如 logs/xxx/ckpts/iter_000019")
    parser.add_argument("--resume_exp_name", type=str, default=None, help="继续写回哪个实验目录名；不填则新建一个")
    parser.add_argument("--resume_it", type=int, default=None, help="起始迭代号；不填则尝试从路径自动解析")
    parser.add_argument("--total_iters", type=int, default=200, help="训练到的总迭代数")
    parser.add_argument("--ray_temp_dir", type=str, default="/mnt/nas_9/group/zhouboyang/tmp")
    args = parser.parse_args()

    restore_ckpt_path = args.restore_ckpt_path.rstrip("/")
    resume_it = args.resume_it if args.resume_it is not None else infer_resume_it_from_ckpt_path(restore_ckpt_path)
    if resume_it is None:
        raise ValueError("无法从 restore_ckpt_path 自动解析 resume_it，请手动传 --resume_it")

    cfg = SceneConfig()
    try:
        cfg.experiment_params.restore_ckpt_path = restore_ckpt_path
    except Exception:
        pass

    algo, exp_info = prepare_exp(
        cfg=cfg,
        ablation=args.ablation,
        resume_exp_name=args.resume_exp_name,
        ray_temp_dir=args.ray_temp_dir,
    )

    print(f"[resume] ablation={args.ablation}")
    print(f"[resume] restore_ckpt_path={restore_ckpt_path}")
    print(f"[resume] resume_it={resume_it}")
    print(f"[resume] total_iters={args.total_iters}")
    print(f"[resume] exp_dir={exp_info['exp_dir']}")

    algo.restore(restore_ckpt_path)

    for it in range(resume_it, args.total_iters):
        result = algo.train()
        print(f"[{exp_info['exp_name']}] iter={it}", result)

        iter_ckpt_root = os.path.join(exp_info["ckpt_dir"], f"iter_{it + 1:06d}")
        algo.save(iter_ckpt_root)
        print(f"Checkpoint saved at {iter_ckpt_root}")

    final_root = os.path.join(exp_info["ckpt_dir"], "final")
    final_ckpt = algo.save(final_root)
    print(f"Final checkpoint saved at {final_ckpt}")

    ray.shutdown()
