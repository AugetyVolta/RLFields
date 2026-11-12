from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict
import yaml
import os
from pathlib import Path
import pickle
import sys
import types

@dataclass
class ESDF_Cfg:
    """
    仅用于 2D ESDF（单位：米）的地图配置与可视化参数。
    注意：实际分辨率在 __init__ 里根据 (x_min..x_max, map_size) 动态计算。
    """
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    map_size: int = 80       # H=W=map_size
    r_max: float = 4.0       # 邻居观察的最大范围

    # 运行时会重算：resolution = (x_max - x_min) / map_size
    resolution: float = (x_max - x_min) / map_size

@dataclass
class SceneParameters:
    width: float = 16
    height: float = 16
    step_time: float = 0.1
    sample_time: float = 0.1
    collision_mode: str = 'stop'

    number_obstacles: int = 13
    min_distance_betw_obs: float = 1.5

    circle_radius_min: float = 0.3
    circle_radius_max: float = 1.5

    rect_edge_min: float = 0.5
    rect_edge_max: float = 2.0
    
@dataclass
class RobotParameters:
    number_robots: int = 10
    kinematics: str = 'omni'
    shape: str = 'circle'
    radius: float = 0.1
    goal_threshold: float = 0.3       # 判断到达目标的距离阈值,世界坐标系下的尺度,不是归一化后的

    lidar_range_min: float = 0.0
    lidar_range_max: float = 6.0
    lidar_angle_range: float = 6.28
    lidar_number: int = 200

    single_max_timestep: int = 500   # 每个机器人单次任务的最大时间步数, 最大500步
    move_single_step: float = 0.1    # 单位速度1,单位时间，每个时间步机器人的最大移动距离, 和dt保持一致，1*dt
    v_max: float = 1.5               # 机器人的最大速度

@dataclass
class RLFieldsConfig:
    self_obs_dim: int = 4
    self_obs_embed_dim: int = 64
    neighbor_obs_dim: int = 4
    neighbor_hidden_size: int = 128
    neighbor_base_channels: int = 64
    obstacle_embedding_dim: int = 128
    n_head: int = 4
    num_use_neighbor_obs: int = 3    # 使用固定范围的最近几个机器人作为邻居观察
    env_output_feature: int = 256
    fusion_mode: str = "concat_ln"
    # -------- SDF/Attention 增强 --------
    use_sdf_grad: bool = True
    sdf_grad_scale: float = 8.0
    sdf_grad_clip: float | None = 1.0
    use_ctx_ffn: bool = True
    use_neigh_self_attn: bool = True
    # -------- NTFields 配置 --------
    nt_dim: int = 2
    nt_vmax: float = 1.0
    nt_h_size: int = 256
    nt_enc_d: int = 7
    nt_dec_d: int = 7
    nt_full_coords_enc: bool = False
    nt_basic_block: str = "resnet"
    nt_use_unet: bool = False
    nt_symm_op_type: str = "min_max"
    nt_pretrained_path: str = 'ckpt.pt'
    device: str = "cuda"

@dataclass
class RewardConfig:
    # -------- 时间/进度 --------
    w_T_progress: float = 10.0
    w_step: float = 0.025
    delta_T_clip: float = 1.0

    # -------- 多机器人势场 --------
    w_peer_delta: float = 2.0
    w_peer_static: float = 0.15
    w_peer_away_factor: float = 0.8

    # -------- 障碍物清距 --------
    w_obst_delta: float = 1.5
    w_obst_static: float = 0.08
    w_obst_away_factor: float = 0.3
    clear_danger: float = 0.10
    clear_safe: float = 0.45

    # -------- 速度项 --------
    w_speed_safe: float = 0.06
    w_speed_danger: float = 0.20

    # -------- 先验方向对齐 --------
    w_prior_align: float = 0.03
    align_speed_power: float = 2.0

    # -------- 终止项 --------
    bonus_arrive: float = 25.0
    pen_timeout: float = 30.0
    pen_collision_robot: float = 30.0
    pen_collision_obstacle: float = 30.0
    collision_peer_margin: float = 0.05

    # -------- 势场形状/阈值 --------
    peer_danger_offset: float = 0.3
    peer_safe_margin: float = 0.5
    peer_sense_margin: float = 1.0
    peer_k_far: float = 0.3
    peer_k_mid: float = 1.0
    peer_k_max: float = 3.0
    peer_potential_beta: float = 5.0

    # -------- 风险 gate --------
    progress_gate_risk_coeff: float = 0.8
    progress_gate_min: float = 0.2

@dataclass
class ExprimentConfig:
    exp_dir: str = None
    env_yaml_path: str = 'config/random_env.yaml'
    disable_plotting: bool = False
    ckpt_save_interval: int = 10
    restore_ckpt_path: str = None
    max_iters: int = 200

    num_env_runners: int = 2
    num_envs_per_env_runner: int = 1
    num_cpus_per_env_runner: int = 10
    num_gpus_per_env_runner: int = 0.5
    rollout_fragment_length: int = 150
    num_cpus_for_main_process: int = 4

@dataclass
class AlgorithmConfig:
    lr: float = 5e-5
    gamma: float = 0.99
    num_epochs: int = 15
    train_batch_size: int = 16000
    minibatch_size: int = 512
    use_gae: bool = True
    lambda_: float = 0.95
    use_critic: bool = True
    use_kl_loss: bool = True
    kl_coeff: float = 0.2
    kl_target: float = 0.02
    entropy_coeff: float = 0.005
    vf_loss_coeff: float = 1.0
    vf_clip_param: float = 10.0
    clip_param: float = 0.2
    grad_clip: float = None

@dataclass
class SceneConfig:
    scene_params: SceneParameters = field(default_factory=SceneParameters)
    robot_params: RobotParameters = field(default_factory=RobotParameters)
    esdf_params : ESDF_Cfg = field(default_factory=ESDF_Cfg)
    network_params : RLFieldsConfig = field(default_factory=RLFieldsConfig)
    reward_params : RewardConfig = field(default_factory=RewardConfig)
    experiment_params : ExprimentConfig = field(default_factory=ExprimentConfig)
    algorithm_params : AlgorithmConfig = field(default_factory=AlgorithmConfig)

    def get_env_config(self) -> Dict[str, Any]:
        """給 env_config."""
        return {
            "self_obs_dim": self.network_params.self_obs_dim,
            "neighbor_obs_dim": self.network_params.neighbor_obs_dim,
            "num_use_neighbor_obs": self.network_params.num_use_neighbor_obs,
            "H": self.esdf_params.map_size,
            "num_robots": self.robot_params.number_robots,
            "action_dim": self.network_params.nt_dim,
            "env_yaml_path": self.experiment_params.env_yaml_path,
            "disable_plotting": self.experiment_params.disable_plotting,
            "scene_config": asdict(self)
        }

    def get_network_config(self) -> Dict[str, Any]:
        """给 custom_model_config."""
        return {
            "self_obs_dim": self.network_params.self_obs_dim,
            "self_obs_embed_dim": self.network_params.self_obs_embed_dim,
            "neighbor_obs_dim": self.network_params.neighbor_obs_dim,
            "neighbor_hidden_size": self.network_params.neighbor_hidden_size,
            "neighbor_base_channels": self.network_params.neighbor_base_channels,
            "obstacle_embedding_dim": self.network_params.obstacle_embedding_dim,
            "n_head": self.network_params.n_head,
            "num_use_neighbor_obs": self.network_params.num_use_neighbor_obs,
            "env_output_feature": self.network_params.env_output_feature,
            "fusion_mode": self.network_params.fusion_mode,
            "use_sdf_grad": self.network_params.use_sdf_grad,
            "sdf_grad_scale": self.network_params.sdf_grad_scale,
            "sdf_grad_clip": self.network_params.sdf_grad_clip,
            "use_ctx_ffn": self.network_params.use_ctx_ffn,
            "use_neigh_self_attn": self.network_params.use_neigh_self_attn,
            "nt_dim": self.network_params.nt_dim,
            "nt_vmax": self.network_params.nt_vmax,
            "nt_h_size": self.network_params.nt_h_size,
            "nt_enc_d": self.network_params.nt_enc_d,
            "nt_dec_d": self.network_params.nt_dec_d,
            "nt_full_coords_enc": self.network_params.nt_full_coords_enc,
            "nt_basic_block": self.network_params.nt_basic_block,
            "nt_use_unet": self.network_params.nt_use_unet,
            "nt_symm_op_type": self.network_params.nt_symm_op_type,
            "nt_pretrained_path": self.network_params.nt_pretrained_path,
            "device": self.network_params.device,
        }
    
    def dump(self, path: str) -> None:
        """
        导出当前 SceneConfig 的所有参数（包含嵌套 dataclass）到给定路径。
        默认写成 YAML 文件。
        """
        data = asdict(self)  # 递归展开所有 dataclass -> dict

        # 确保目录存在
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)


    @classmethod
    def from_dict(cls, data):
        cfg = cls()
        for item in fields(cfg):
            section = getattr(cfg, item.name)
            for key, value in data.get(item.name, {}).items():
                if hasattr(section, key):
                    setattr(section, key, value)
        cfg.esdf_params.resolution = (cfg.esdf_params.x_max - cfg.esdf_params.x_min) / cfg.esdf_params.map_size
        return cfg

    @classmethod
    def from_checkpoint(cls, ckpt_path):
        # 评测/续训读取保存时的结构，不需要手动切换 prior_align 或增强开关。
        path = Path(ckpt_path).expanduser().resolve()
        policy_file = path / 'policies/shared_policy/policy_state.pkl'
        if not policy_file.is_file():
            policy_file = path / 'policy_state.pkl'
        if not policy_file.is_file():
            raise FileNotFoundError(f'No shared_policy checkpoint in {path}')
        # 历史 checkpoint 中的旧类路径只在内存中映射，不保留重复实现文件。
        from .multi_robot_env import MultiRobotEnv
        from .sim.marl_sim_env import MARL_SIM
        from model.models import RLFields
        from model.policy_value_network import RLFieldsActorModel
        for module_name, class_name, target in [
            ("env.multi_robot_env_prior_align", "MultiRobotEnvPriorAlign", MultiRobotEnv),
            ("env.sim.marl_sim_env_prior_align", "MARL_SIM_PriorAlign", MARL_SIM),
            ("model.models_prior_align", "RLFieldsPriorAlign", RLFields),
            ("model.policy_value_network_prior_align", "RLFieldsActorModelPriorAlign", RLFieldsActorModel),
        ]:
            module = types.ModuleType(module_name)
            setattr(module, class_name, target)
            sys.modules.setdefault(module_name, module)
        with policy_file.open('rb') as f:
            state = pickle.load(f)
        policy_cfg = state['policy_spec']['config']
        cfg = cls.from_dict(policy_cfg.get("env_config", {}).get("scene_config", {}))
        run_dir = None
        for parent in path.parents:
            snapshot = parent / 'scene_config.yaml'
            if not snapshot.is_file():
                snapshot = parent / 'scene_config_resume.yaml'
            if snapshot.is_file():
                with snapshot.open(encoding='utf-8') as f:
                    cfg = cls.from_dict(yaml.safe_load(f) or {})
                run_dir = parent
                break
        model_cfg = policy_cfg['model'].get('custom_model_config', {})
        for key, value in model_cfg.items():
            if hasattr(cfg.network_params, key):
                setattr(cfg.network_params, key, value)
        weights = state['weights']
        if 'fused_ln.weight' not in weights:
            raise ValueError('This checkpoint uses non-aligned observations; use a prior_align checkpoint.')
        cfg.network_params.use_sdf_grad = weights['rlfields.env_encoder.obstacle_encoder.feature_extractor.0.weight'].shape[1] == 3
        cfg.network_params.use_ctx_ffn = any('.ctx_ffn.' in key for key in weights)
        cfg.network_params.use_neigh_self_attn = any('.neigh_self_attn.' in key for key in weights)
        env_cfg = policy_cfg.get('env_config', {})
        cfg.robot_params.number_robots = env_cfg.get('num_robots', cfg.robot_params.number_robots)
        cfg.network_params.num_use_neighbor_obs = env_cfg.get('num_use_neighbor_obs', cfg.network_params.num_use_neighbor_obs)
        cfg.esdf_params.map_size = env_cfg.get('H', cfg.esdf_params.map_size)
        cfg.esdf_params.resolution = (cfg.esdf_params.x_max - cfg.esdf_params.x_min) / cfg.esdf_params.map_size
        cfg.experiment_params.env_yaml_path = env_cfg.get('env_yaml_path', cfg.experiment_params.env_yaml_path)
        root = Path(__file__).resolve().parents[1]
        for section, key, archive, default in [
            (cfg.experiment_params, 'env_yaml_path', 'env.yaml', root / 'config/random_env.yaml'),
            (cfg.network_params, 'nt_pretrained_path', 'ntfields.pt', root / 'ckpt.pt'),
        ]:
            value = Path(getattr(section, key)).expanduser()
            candidates = ([run_dir / archive] if run_dir else []) + [value, root / value, default]
            selected = next((p for p in candidates if p.is_file()), None)
            if selected is None:
                raise FileNotFoundError(f'Missing {key}: {default}')
            setattr(section, key, str(selected.resolve()))
        return cfg
