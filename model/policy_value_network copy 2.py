import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from .models import RLFields

class RLFieldsActorModel(TorchModelV2, nn.Module):
    """
    PPO 的 actor-critic 模型封装：
    - RLFields(obs) -> (fused_feat, grad_all)
        fused_feat: 融合了 NTFields + 环境/邻居 的高维特征 [B, nt_feat_dim]
        grad_all:   NTFields 在 (start, goal) 上的梯度 [B, 4]
    - 策略：用 grad_start 做“先验方向”，再加 residual，得到动作均值；
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

        # ==== 构造 RLFields ====
        self.rlfields = RLFields(
            self_obs_dim=cfg.get("self_obs_dim", 4),
            self_obs_embed_dim=cfg.get("self_obs_embed_dim", 128),
            neighbor_obs_dim=cfg.get("neighbor_obs_dim", 4),
            neighbor_hidden_size=cfg.get("neighbor_hidden_size", 128),
            neighbor_base_channels=cfg.get("neighbor_base_channels", 32),
            obstacle_embedding_dim=cfg.get("obstacle_embedding_dim", 128),
            n_head=cfg.get("n_head", 6),
            num_use_neighbor_obs=cfg.get("num_use_neighbor_obs", 6),
            env_output_feature=cfg.get("env_output_feature", 128),
            fusion_mode=cfg.get("fusion_mode", "film"),
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
        self.fused_feat_dim = self.rlfields.nt_feat_dim
        self.nt_dim = self.rlfields.nt_dim  # 一般是 2，对应 start 的维度

        # ==== 先验 embedding 分支 ====
        # 输入: [g_dir(2), g_norm(1)] 共 nt_dim + 1 维
        self.prior_embed_dim = 64
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.nt_dim + 1, self.prior_embed_dim),
            nn.SiLU(),
            nn.Linear(self.prior_embed_dim, self.prior_embed_dim),
            nn.SiLU(),
        )

        # ==== 残差头：输入 = fused_feat + prior_feat ====
        delta_input_dim = self.fused_feat_dim + self.prior_embed_dim

        # 从 [fused_feat, prior_feat] 学一个小残差 Δμ
        self.delta_head = nn.Sequential(
            nn.Linear(delta_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

        # 残差头的最后一层初始化得很“小”，避免一开始把先验打乱
        last_linear = self.delta_head[-1]
        nn.init.zeros_(last_linear.bias)
        with torch.no_grad():
            last_linear.weight.mul_(0.2)

        # log_std：全局可学参数，一开始噪声不要太大
        self.log_std_param = nn.Parameter(torch.zeros(self.action_dim))
        with torch.no_grad():
            self.log_std_param.data.fill_(-1.0)

        # ==== 价值函数部分 ====
        self.value_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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

        # RLFields: 得到融合特征 + NTFields 梯度
        fused_feat, grad_all = self.rlfields(obs)      # fused_feat: [B, T], grad_all: [B, 2*nt_dim]
        self._last_fused_feat = fused_feat

        # 只用对 start 的梯度作为先验
        grad_start = grad_all[:, :self.nt_dim]         # [B, 2]

        # 模长 + 方向
        g_norm = torch.norm(grad_start, dim=-1, keepdim=True)        # [B, 1]
        g_dir  = grad_start / (g_norm + 1e-8)                        # [B, 2]
        
        # 正交方向 (2D)
        g_perp = torch.cat([-g_dir[:, 1:2], g_dir[:, 0:1]], dim=-1)  # [B, 2]

        # ===== 先验 embedding =====
        prior_raw  = torch.cat([g_dir, g_norm], dim=-1)  # [B, nt_dim + 1] = [B, 3]
        prior_feat = self.prior_mlp(prior_raw)           # [B, prior_embed_dim]

        # ===== 残差头输入：环境特征 + 先验 embedding =====
        delta_in = torch.cat([fused_feat, prior_feat], dim=-1)  # [B, T + prior_embed_dim]

        delta = self.delta_head(delta_in)      # [B, 2]
        delta_par  = delta[:, :1]              # [B,1]
        delta_perp = delta[:, 1:]              # [B,1]

        # 沿先验方向的系数：基线为 g_norm，再加残差
        a_par  = g_norm + delta_par            # [B,1]
        # 正交方向的系数：完全由残差学
        a_perp = delta_perp                    # [B,1]

        # 在 (g_dir, g_perp) 基底重构最终动作
        mu = a_par * g_dir + a_perp * g_perp   # [B,2]

        log_std = self.log_std_param.expand_as(mu)
        logits = torch.cat([mu, log_std], dim=-1)    # [B, 2*action_dim]

        return logits, state

    def value_function(self):
        # 用 fused_feat 做 value
        assert self._last_fused_feat is not None, "must call forward() before value_function()"
        v = self.value_head(self._last_fused_feat)                    # [B,1]
        return v.squeeze(-1)