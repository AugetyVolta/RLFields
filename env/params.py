from dataclasses import dataclass, asdict
from typing import Any, Dict
import yaml
import os

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
    number_robots: int = 6
    kinematics: str = 'omni'
    shape: str = 'circle'
    radius: float = 0.1
    goal_threshold: float = 0.3       # 判断到达目标的距离阈值,世界坐标系下的尺度,不是归一化后的

    lidar_range_min: float = 0.0
    lidar_range_max: float = 6.0
    lidar_angle_range: float = 6.28
    lidar_number: int = 200

    single_max_timestep: int = 1000   # 每个机器人单次任务的最大时间步数
    move_single_step: float = 0.0025  # 每个时间步机器人的最大移动距离

@dataclass
class RLFieldsConfig:
    self_obs_dim: int = 4
    self_obs_embed_dim: int = 128
    neighbor_obs_dim: int = 4
    neighbor_hidden_size: int = 128
    neighbor_base_channels: int = 32
    obstacle_embedding_dim: int = 128
    n_head: int = 6
    num_use_neighbor_obs: int = 6    # 使用固定范围的最近几个机器人作为邻居观察
    env_output_feature: int = 128
    fusion_mode: str = "film"
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
    pass

@dataclass
class ExprimentConfig:
    exp_dir: str = None
    env_yaml_path: str = 'config/random_env.yaml'
    disable_plotting: bool = False
    ckpt_save_interval: int = 10

@dataclass
class SceneConfig:
    scene_params: SceneParameters = SceneParameters()
    robot_params: RobotParameters = RobotParameters()
    esdf_params : ESDF_Cfg = ESDF_Cfg()
    network_params : RLFieldsConfig = RLFieldsConfig()
    reward_params : RewardConfig = RewardConfig()
    experiment_params : ExprimentConfig = ExprimentConfig()

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
            "disable_plotting": self.experiment_params.disable_plotting
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

