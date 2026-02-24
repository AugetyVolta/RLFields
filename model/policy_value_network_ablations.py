import math
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from .models import RLFields
from .models_prior_align import RLFieldsPriorAlign


class RLFieldsPriorAlign_NoPCFeatures(RLFieldsPriorAlign):
    """Ablation-2/3（prior-align 基线）：仅使用 env_encoder 特征，不使用 NTFields 特征。"""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        # -------- 1) self_obs（世界坐标）--------
        self_obs = observations[:, : self.self_obs_dim]

        # -------- 2) neighbor_obs + mask --------
        offset = self.self_obs_dim
        neigh_flat_dim = self.num_use_neighbor_obs * self.neighbor_obs_dim

        neighbor_obs_flat = observations[:, offset: offset + neigh_flat_dim]
        neighbor_obs = neighbor_obs_flat.view(B, self.num_use_neighbor_obs, self.neighbor_obs_dim)
        offset += neigh_flat_dim

        neighbor_valid_num = observations[:, offset].long()
        offset += 1

        idx = torch.arange(self.num_use_neighbor_obs, device=observations.device).unsqueeze(0).expand(B, -1)
        neighbor_mask = idx < neighbor_valid_num.unsqueeze(-1)

        # -------- 3) SDF + prior_dir --------
        prior_dir = observations[:, -self.nt_dim:]
        sdf_flat = observations[:, offset:-self.nt_dim]
        sdf_len = sdf_flat.shape[1]
        H = int(math.sqrt(sdf_len))
        assert H * H == sdf_len, f"sdf_input 展平长度 {sdf_len} 不能组成正方形图像"
        sdf_input = sdf_flat.view(B, 1, H, H)

        # -------- 4) prior-aligned goal_rel（仅给 EnvEncoder）--------
        half_dim = self.self_obs_dim // 2
        start = self_obs[:, :half_dim]
        goal = self_obs[:, half_dim:]
        goal_rel = goal - start

        R = self._rot_from_prior(prior_dir[:, :2])
        goal_rel_rot = torch.bmm(R, goal_rel.unsqueeze(-1)).squeeze(-1)

        zeros = torch.zeros_like(start)
        self_obs_for_env = torch.cat([zeros, goal_rel_rot], dim=-1)

        # -------- 5) env encoder（不做 NTFields 融合）--------
        env_embedding = self.env_encoder(
            self_obs=self_obs_for_env,
            neighbor_obs=neighbor_obs,
            sdf_input=sdf_input,
            neighbor_mask=neighbor_mask,
        )

        return env_embedding, prior_dir


class RLFieldsActorModel_NoPCFeaturesPriorAlign(TorchModelV2, nn.Module):
    """Ablation-2/3（prior-align 基线）：不使用 NTFields 特征，保留 prior-align trunk。"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_dim = action_space.shape[0]
        assert num_outputs == 2 * self.action_dim, f"num_outputs={num_outputs} 和 action_dim={self.action_dim} 不匹配"

        cfg = {}
        cfg.update(model_config.get("custom_model_config", {}))
        cfg.update(custom_model_kwargs)

        self.rlfields = RLFieldsPriorAlign_NoPCFeatures(
            self_obs_dim=cfg.get("self_obs_dim", 4),
            self_obs_embed_dim=cfg.get("self_obs_embed_dim", 64),
            neighbor_obs_dim=cfg.get("neighbor_obs_dim", 4),
            neighbor_hidden_size=cfg.get("neighbor_hidden_size", 128),
            neighbor_base_channels=cfg.get("neighbor_base_channels", 64),
            obstacle_embedding_dim=cfg.get("obstacle_embedding_dim", 128),
            n_head=cfg.get("n_head", 4),
            num_use_neighbor_obs=cfg.get("num_use_neighbor_obs", 3),
            env_output_feature=cfg.get("env_output_feature", 256),
            fusion_mode=cfg.get("fusion_mode", "film"),
            use_sdf_grad=cfg.get("use_sdf_grad", True),
            sdf_grad_scale=cfg.get("sdf_grad_scale", 8.0),
            sdf_grad_clip=cfg.get("sdf_grad_clip", 1.0),
            use_ctx_ffn=cfg.get("use_ctx_ffn", True),
            use_neigh_self_attn=cfg.get("use_neigh_self_attn", True),
            nt_dim=cfg.get("nt_dim", 2),
            nt_vmax=cfg.get("nt_vmax", 1.0),
            nt_h_size=cfg.get("nt_h_size", 256),
            nt_enc_d=cfg.get("nt_enc_d", 7),
            nt_dec_d=cfg.get("nt_dec_d", 7),
            nt_full_coords_enc=cfg.get("nt_full_coords_enc", False),
            nt_basic_block=cfg.get("nt_basic_block", "resnet"),
            nt_use_unet=cfg.get("nt_use_unet", False),
            nt_symm_op_type=cfg.get("nt_symm_op_type", "min_max"),
            nt_pretrained_path=cfg.get("nt_pretrained_path", 'ckpt.pt'),
            device=cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.fused_feat_dim = self.rlfields.env_output_feature
        self.nt_dim = self.rlfields.nt_dim

        self.prior_embed_dim = 64
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.nt_dim, self.prior_embed_dim),
            nn.SiLU(),
            nn.Linear(self.prior_embed_dim, self.prior_embed_dim),
        )
        self.fused_ln = nn.LayerNorm(self.fused_feat_dim)
        self.prior_ln = nn.LayerNorm(self.prior_embed_dim)

        delta_input_dim = self.fused_feat_dim + self.prior_embed_dim
        self.delta_head = nn.Sequential(
            nn.Linear(delta_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh(),
        )

        last_linear = self.delta_head[-2]
        with torch.no_grad():
            last_linear.weight.mul_(0.6)
            last_linear.bias[0] = 0.0
            last_linear.bias[1] = 0.0

        self.log_std_param = nn.Parameter(torch.zeros(self.action_dim))
        with torch.no_grad():
            self.log_std_param[0] = -1.0
            self.log_std_param[1] = -1.5

        self.value_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._last_fused_feat = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        fused_feat, prior_dir = self.rlfields(obs)
        self._last_fused_feat = fused_feat

        prior_feat = self.prior_mlp(prior_dir)

        fused_feat = self.fused_ln(fused_feat)
        prior_feat = self.prior_ln(prior_feat)
        delta_in = torch.cat([fused_feat, prior_feat], dim=-1)

        mu = self.delta_head(delta_in)
        log_std = self.log_std_param.expand_as(mu)
        logits = torch.cat([mu, log_std], dim=-1)
        return logits, state

    def value_function(self):
        assert self._last_fused_feat is not None, "must call forward() before value_function()"
        v = self.value_head(self._last_fused_feat)
        return v.squeeze(-1)


class RLFieldsActorModel_NoPriorDir(TorchModelV2, nn.Module):
    """Ablation-1: 去掉先验方向相关分支（不使用 prior_dir embedding），保留 PC 特征。"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_dim = action_space.shape[0]
        assert num_outputs == 2 * self.action_dim, f"num_outputs={num_outputs} 和 action_dim={self.action_dim} 不匹配"

        cfg = {}
        cfg.update(model_config.get("custom_model_config", {}))
        cfg.update(custom_model_kwargs)

        self.rlfields = RLFields(
            self_obs_dim=cfg.get("self_obs_dim", 4),
            self_obs_embed_dim=cfg.get("self_obs_embed_dim", 64),
            neighbor_obs_dim=cfg.get("neighbor_obs_dim", 4),
            neighbor_hidden_size=cfg.get("neighbor_hidden_size", 128),
            neighbor_base_channels=cfg.get("neighbor_base_channels", 64),
            obstacle_embedding_dim=cfg.get("obstacle_embedding_dim", 128),
            n_head=cfg.get("n_head", 4),
            num_use_neighbor_obs=cfg.get("num_use_neighbor_obs", 3),
            env_output_feature=cfg.get("env_output_feature", 256),
            fusion_mode=cfg.get("fusion_mode", "film"),
            use_sdf_grad=cfg.get("use_sdf_grad", True),
            sdf_grad_scale=cfg.get("sdf_grad_scale", 1.0),
            sdf_grad_clip=cfg.get("sdf_grad_clip", 1.0),
            use_ctx_ffn=cfg.get("use_ctx_ffn", True),
            use_neigh_self_attn=cfg.get("use_neigh_self_attn", True),
            nt_dim=cfg.get("nt_dim", 2),
            nt_vmax=cfg.get("nt_vmax", 1.0),
            nt_h_size=cfg.get("nt_h_size", 256),
            nt_enc_d=cfg.get("nt_enc_d", 7),
            nt_dec_d=cfg.get("nt_dec_d", 7),
            nt_full_coords_enc=cfg.get("nt_full_coords_enc", False),
            nt_basic_block=cfg.get("nt_basic_block", "resnet"),
            nt_use_unet=cfg.get("nt_use_unet", False),
            nt_symm_op_type=cfg.get("nt_symm_op_type", "min_max"),
            nt_pretrained_path=cfg.get("nt_pretrained_path", "ckpt.pt"),
            device=cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.fused_feat_dim = self.rlfields.env_output_feature

        self.delta_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh(),
        )

        last_linear = self.delta_head[-2]
        with torch.no_grad():
            last_linear.weight.mul_(0.6)
            last_linear.bias[0] = 0.0
            last_linear.bias[1] = 0.0

        self.log_std_param = nn.Parameter(torch.zeros(self.action_dim))
        with torch.no_grad():
            self.log_std_param[0] = -1.0
            self.log_std_param[1] = -1.5

        self.value_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._last_fused_feat = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        fused_feat, _ = self.rlfields(obs)
        self._last_fused_feat = fused_feat

        mu = self.delta_head(fused_feat)
        log_std = self.log_std_param.expand_as(mu)
        logits = torch.cat([mu, log_std], dim=-1)
        return logits, state

    def value_function(self):
        assert self._last_fused_feat is not None, "must call forward() before value_function()"
        v = self.value_head(self._last_fused_feat)
        return v.squeeze(-1)


class RLFields_NoPCFeatures_NoPrior(RLFields):
    """Ablation-3 子模块：只保留 env_encoder，且不引入 prior 方向分支。"""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        self_obs = observations[:, :self.self_obs_dim]

        offset = self.self_obs_dim
        neigh_flat_dim = self.num_use_neighbor_obs * self.neighbor_obs_dim
        neighbor_obs_flat = observations[:, offset: offset + neigh_flat_dim]
        neighbor_obs = neighbor_obs_flat.view(B, self.num_use_neighbor_obs, self.neighbor_obs_dim)
        offset += neigh_flat_dim

        neighbor_valid_num = observations[:, offset].long()
        offset += 1

        idx = torch.arange(self.num_use_neighbor_obs, device=observations.device).unsqueeze(0).expand(B, -1)
        neighbor_mask = idx < neighbor_valid_num.unsqueeze(-1)

        sdf_flat = observations[:, offset:-self.nt_dim]
        sdf_len = sdf_flat.shape[1]
        H = int(math.sqrt(sdf_len))
        assert H * H == sdf_len, f"sdf_input 展平长度 {sdf_len} 不能组成正方形图像"
        sdf_input = sdf_flat.view(B, 1, H, H)

        env_embedding = self.env_encoder(
            self_obs=self_obs,
            neighbor_obs=neighbor_obs,
            sdf_input=sdf_input,
            neighbor_mask=neighbor_mask,
        )
        return env_embedding


class RLFieldsActorModel_NoPCFeaturesNoPriorDir(TorchModelV2, nn.Module):
    """Ablation-3: 同时去除先验方向分支与 PC-planner 特征。"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.action_dim = action_space.shape[0]
        assert num_outputs == 2 * self.action_dim, f"num_outputs={num_outputs} 和 action_dim={self.action_dim} 不匹配"

        cfg = {}
        cfg.update(model_config.get("custom_model_config", {}))
        cfg.update(custom_model_kwargs)

        self.rlfields = RLFields_NoPCFeatures_NoPrior(
            self_obs_dim=cfg.get("self_obs_dim", 4),
            self_obs_embed_dim=cfg.get("self_obs_embed_dim", 64),
            neighbor_obs_dim=cfg.get("neighbor_obs_dim", 4),
            neighbor_hidden_size=cfg.get("neighbor_hidden_size", 128),
            neighbor_base_channels=cfg.get("neighbor_base_channels", 64),
            obstacle_embedding_dim=cfg.get("obstacle_embedding_dim", 128),
            n_head=cfg.get("n_head", 4),
            num_use_neighbor_obs=cfg.get("num_use_neighbor_obs", 3),
            env_output_feature=cfg.get("env_output_feature", 256),
            fusion_mode=cfg.get("fusion_mode", "film"),
            use_sdf_grad=cfg.get("use_sdf_grad", True),
            sdf_grad_scale=cfg.get("sdf_grad_scale", 1.0),
            sdf_grad_clip=cfg.get("sdf_grad_clip", 1.0),
            use_ctx_ffn=cfg.get("use_ctx_ffn", True),
            use_neigh_self_attn=cfg.get("use_neigh_self_attn", True),
            nt_dim=cfg.get("nt_dim", 2),
            nt_vmax=cfg.get("nt_vmax", 1.0),
            nt_h_size=cfg.get("nt_h_size", 256),
            nt_enc_d=cfg.get("nt_enc_d", 7),
            nt_dec_d=cfg.get("nt_dec_d", 7),
            nt_full_coords_enc=cfg.get("nt_full_coords_enc", False),
            nt_basic_block=cfg.get("nt_basic_block", "resnet"),
            nt_use_unet=cfg.get("nt_use_unet", False),
            nt_symm_op_type=cfg.get("nt_symm_op_type", "min_max"),
            nt_pretrained_path=cfg.get("nt_pretrained_path", "ckpt.pt"),
            device=cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.fused_feat_dim = self.rlfields.env_output_feature

        self.delta_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh(),
        )

        last_linear = self.delta_head[-2]
        with torch.no_grad():
            last_linear.weight.mul_(0.6)
            last_linear.bias[0] = 0.0
            last_linear.bias[1] = 0.0

        self.log_std_param = nn.Parameter(torch.zeros(self.action_dim))
        with torch.no_grad():
            self.log_std_param[0] = -1.0
            self.log_std_param[1] = -1.5

        self.value_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._last_fused_feat = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        fused_feat = self.rlfields(obs)
        self._last_fused_feat = fused_feat

        mu = self.delta_head(fused_feat)
        log_std = self.log_std_param.expand_as(mu)
        logits = torch.cat([mu, log_std], dim=-1)
        return logits, state

    def value_function(self):
        assert self._last_fused_feat is not None, "must call forward() before value_function()"
        v = self.value_head(self._last_fused_feat)
        return v.squeeze(-1)
