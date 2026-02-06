import math
import torch

from .models import RLFields


class RLFieldsPriorAlign(RLFields):
    """
    RLFields with goal_rel rotated into prior-aligned coordinate frame.
    - Keeps NTFields branch in world coordinates (self_obs unchanged).
    - Only rotates goal_rel for EnvEncoder input, so local features align with prior.
    """
    @staticmethod
    def _rot_from_prior(prior_dir: torch.Tensor) -> torch.Tensor:
        """
        prior_dir: [B,2]
        return R: [B,2,2] such that R * v_world = v_prior_frame
        """
        pn = torch.norm(prior_dir, dim=-1, keepdim=True).clamp_min(1e-6)
        c = prior_dir[:, 0:1] / pn
        s = prior_dir[:, 1:2] / pn
        row0 = torch.cat([c, s], dim=1)
        row1 = torch.cat([-s, c], dim=1)
        return torch.stack([row0, row1], dim=1)

    def forward(self, observations: torch.Tensor):
        B = observations.shape[0]

        # -------- 1) 解析 self_obs（世界坐标） --------
        self_obs = observations[:, :self.self_obs_dim]          # [B, Ds]

        # -------- 2) 解析 neighbor_obs + neighbor_mask --------
        offset = self.self_obs_dim
        neigh_flat_dim = self.num_use_neighbor_obs * self.neighbor_obs_dim

        neighbor_obs_flat = observations[:, offset: offset + neigh_flat_dim]
        neighbor_obs = neighbor_obs_flat.view(
            B, self.num_use_neighbor_obs, self.neighbor_obs_dim
        )                                                       # [B, K, Dn]
        offset += neigh_flat_dim

        neighbor_valid_num = observations[:, offset]             # [B]
        offset += 1
        neighbor_valid_num = neighbor_valid_num.long()

        idx = torch.arange(self.num_use_neighbor_obs, device=observations.device).unsqueeze(0)
        idx = idx.expand(B, -1)                                  # [B, K]
        neighbor_mask = idx < neighbor_valid_num.unsqueeze(-1)   # [B, K] bool

        # -------- 3) 解析 SDF + prior_dir --------
        prior_dir = observations[:, -self.nt_dim:]               # [B, nt_dim]
        sdf_flat = observations[:, offset:-self.nt_dim]          # [B, H*H]
        sdf_len = sdf_flat.shape[1]
        H = int(math.sqrt(sdf_len))
        assert H * H == sdf_len, f"sdf_input 展平长度 {sdf_len} 不能组成正方形图像"
        sdf_input = sdf_flat.view(B, 1, H, H)                    # [B, 1, H, H]

        # -------- 4) prior-aligned goal_rel（只给 EnvEncoder） --------
        half_dim = self.self_obs_dim // 2
        start = self_obs[:, :half_dim]
        goal = self_obs[:, half_dim:]
        goal_rel = goal - start                                  # [B, 2]

        R = self._rot_from_prior(prior_dir[:, :2])
        goal_rel_rot = torch.bmm(R, goal_rel.unsqueeze(-1)).squeeze(-1)

        # 构造“仅用于 EnvEncoder”的 self_obs：
        # start_rel 恒为 0，所以 start 直接给 0；goal 给旋转后的 goal_rel
        zeros = torch.zeros_like(start)
        self_obs_for_env = torch.cat([zeros, goal_rel_rot], dim=-1)

        # -------- 5) 环境编码（局部坐标） --------
        env_embedding = self.env_encoder(
            self_obs=self_obs_for_env,
            neighbor_obs=neighbor_obs,
            sdf_input=sdf_input,
            neighbor_mask=neighbor_mask,
        )                                                        # [B, env_output_feature]

        # -------- 6) NTFields 特征（世界坐标） --------
        nt_feat, _ = self.ntfields_nn.out_feature(self_obs)      # [B, nt_feat_dim]
        nt_feat = self.ntfields_feat_proj(nt_feat)

        # -------- 7) LateFusionHead 门控融合 --------
        fused_feat = self.late_fusion(env_embedding, nt_feat)    # [B, nt_feat_dim]

        return fused_feat, prior_dir
