import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from .models_prior_align import RLFieldsPriorAlign

class RLFieldsActorModelPriorAlign(TorchModelV2, nn.Module):
    """
    PPO 的 actor-critic 模型封装：
    - RLFields(obs_trunk) -> (fused_feat, prior_dir)
        fused_feat: 融合了 NTFields + 环境/邻居 的高维特征 [B, nt_feat_dim]
        prior_dir:  归一化后的先验方向 [B, nt_dim]（由 RLFields 直接返回）
    - 策略：动作直接输出 (alpha, theta)，范围都在 [-1, 1]。
      其中 alpha 后续在 env 中映射到 [0,1] 并乘 v_max 作为速度幅度；
      theta 后续在 env 中映射到 [-pi,pi] 并以 prior direction 作为基准旋转。
      然后拼接 log_std 形成 logits。
    - 价值函数：用 fused_feat 做 value_head。
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_model_kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 动作维度，例如 Box(shape=(2,))
        self.action_dim = action_space.shape[0]

        # RLlib 的 Gaussian 分布：num_outputs 应该 = 2 * action_dim（mean + log_std）
        assert num_outputs == 2 * self.action_dim, \
            f"num_outputs={num_outputs} 和 action_dim={self.action_dim} 不匹配"

        cfg = {}
        cfg.update(model_config.get("custom_model_config", {}))
        cfg.update(custom_model_kwargs)

        # ==== 构造 RLFieldsPriorAlign ====
        self.rlfields = RLFieldsPriorAlign(
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

        # fused_feat 特征维度 = RLFields 里 fused_feat 的维度
        self.fused_feat_dim = self.rlfields.env_output_feature
        self.nt_dim = self.rlfields.nt_dim  # 一般是 2，对应 prior_dir 的维度

        # ==== 先验 embedding 分支 ====
        # 输入: prior_dir(2) 共 nt_dim 维（RLFields 返回的 prior_dir 已归一化，可直接用）
        self.prior_embed_dim = 64
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.nt_dim, self.prior_embed_dim),
            nn.SiLU(),
            nn.Linear(self.prior_embed_dim, self.prior_embed_dim),
        )
        # 对齐两路特征的尺度，稳定融合
        self.fused_ln = nn.LayerNorm(self.fused_feat_dim)
        self.prior_ln = nn.LayerNorm(self.prior_embed_dim)

        # ==== 残差头：输入 = fused_feat + prior_feat ====
        delta_input_dim = self.fused_feat_dim + self.prior_embed_dim

        # 从 [fused_feat, prior_feat] 输出动作均值 (alpha, theta)
        self.delta_head = nn.Sequential(
            nn.Linear(delta_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh(),  # 输出动作 α/θ ∈ [-1, 1]
        )

        # 动作头的最后一层初始化得很“小”，避免一开始把策略打乱
        last_linear = self.delta_head[-2]  # 最后一个 Linear（Tanh 的前一层）
        with torch.no_grad():
            last_linear.weight.mul_(0.6)
            # alpha/theta 初始都为 0：alpha 映射后约 0.5 幅度；theta≈0 贴近先验方向
            last_linear.bias[0] = 0.0
            last_linear.bias[1] = 0.0

        # log_std：全局可学参数，一开始噪声不要太大
        self.log_std_param = nn.Parameter(torch.zeros(self.action_dim))
        with torch.no_grad():
            self.log_std_param[0] = -1.0
            self.log_std_param[1] = -1.5

        # ==== 价值函数部分 ====
        self.value_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # 缓存最近一次 trunk，用于 value_function()
        self._last_fused_feat = None

    def forward(self, input_dict, state, seq_lens):
        """
        input_dict["obs"]: [B, obs_dim]
        返回:
            logits: [B, 2 * action_dim]，前半 mean，后半 log_std
        """
        obs = input_dict["obs"].float()

        # RLFields: 得到融合特征和先验方向（先验方向已归一化）
        fused_feat, prior_dir = self.rlfields(obs)  # fused_feat: [B, T], prior_dir: [B, nt_dim]
        self._last_fused_feat = fused_feat

        # ===== 先验 embedding =====
        prior_feat = self.prior_mlp(prior_dir)  # [B, prior_embed_dim]

        # ===== policy head 输入：环境特征 + 先验 embedding =====
        fused_feat = self.fused_ln(fused_feat)
        prior_feat = self.prior_ln(prior_feat)
        delta_in = torch.cat([fused_feat, prior_feat], dim=-1)  # [B, T + prior_embed_dim]

        # 输出动作: [alpha_raw, theta_raw]，都在 [-1, 1]
        # - alpha: env 里再做 (alpha+1)/2 -> [0,1]
        # - theta: env 里再做 theta*pi -> [-pi, pi]
        mu = self.delta_head(delta_in)  # [B, action_dim]

        log_std = self.log_std_param.expand_as(mu)
        logits = torch.cat([mu, log_std], dim=-1)    # [B, 2*action_dim]

        return logits, state

    def value_function(self):
        # 用 fused_feat 做 value
        assert self._last_fused_feat is not None, "must call forward() before value_function()"
        v = self.value_head(self._last_fused_feat)                    # [B,1]
        return v.squeeze(-1)
