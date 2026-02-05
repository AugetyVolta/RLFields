import torch
from torch import nn
import torch.nn.functional as F
import math
from .model_blocks import C_encoder_block, C_encoder_block_next, CrossAttention, UnetBaiscBlock

class ObstacleEncoder(nn.Module):
    """
    SDF 编码器（支持 SDF 梯度通道）。
    - 输入的 esdf 需要归一化到[0,1]
    - 可选: 追加 dSDF/dx, dSDF/dy 两个通道
    Conv2d(bias=False) → IN → SiLU；两次 stride=2 下采样 + 一次膨胀卷积。
    输出: [B, embedding_dim]
    """
    def __init__(self, 
                 embedding_dim: int = 128, 
                 base_channels: int = 64,
                 use_sdf_grad: bool = True,
                 sdf_grad_scale: float = 8.0,
                 sdf_grad_clip: float | None = 1.0)-> None:
        super().__init__()
        C = base_channels
        self.use_sdf_grad = use_sdf_grad
        self.sdf_grad_scale = sdf_grad_scale
        self.sdf_grad_clip = sdf_grad_clip
        in_ch = 3 if self.use_sdf_grad else 1
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1, bias=False),
            nn.InstanceNorm2d(C, affine=True), nn.SiLU(),

            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # 80→40
            nn.InstanceNorm2d(C, affine=True), nn.SiLU(),

            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # 40→20
            nn.InstanceNorm2d(C, affine=True), nn.SiLU(),

            nn.Conv2d(C, C, 3, padding=2, dilation=2, bias=False),  # 20→20
            nn.InstanceNorm2d(C, affine=True), nn.SiLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_head = nn.Linear(C, embedding_dim)

    def _sdf_grad_2d(self, sdf: torch.Tensor) -> torch.Tensor:
        """
        sdf: [B,1,H,W] -> 返回 [B,2,H,W] (dx, dy)
        中心差分 + replicate padding，避免边界尺寸变化。
        """
        sdf_p = F.pad(sdf, (1, 1, 1, 1), mode="replicate")
        dx = (sdf_p[:, :, 1:-1, 2:] - sdf_p[:, :, 1:-1, :-2]) * 0.5
        dy = (sdf_p[:, :, 2:, 1:-1] - sdf_p[:, :, :-2, 1:-1]) * 0.5
        grad = torch.cat([dx, dy], dim=1)
        if self.sdf_grad_scale != 1.0:
            grad = grad * self.sdf_grad_scale
        if self.sdf_grad_clip is not None:
            grad = grad.clamp(-self.sdf_grad_clip, self.sdf_grad_clip)
            
        return grad

    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        if sdf.dim() == 3:  # [B,H,W] -> [B,1,H,W]
            sdf = sdf.unsqueeze(1)
        if self.use_sdf_grad:
            grad = self._sdf_grad_2d(sdf)
            x = torch.cat([sdf, grad], dim=1)
        else:
            x = sdf
        feats = self.feature_extractor(x)                   # [B, C, 20, 20]
        pooled = self.global_pool(feats).flatten(1)         # [B, C]
        global_emb = self.embedding_head(pooled)            # [B, D]
        tokens = feats.flatten(2).transpose(1, 2)           # [B, 400, C]
        return global_emb, tokens

class NeighborEncoder(nn.Module):
    """
    邻居信息注意力编码器（修复 M=0 的 mask-softmax + 可返回 neighbor tokens）

    输入:
      - self_obs      : [B, self_obs_dim]
      - neighbor_obs  : [B, K, neighbor_obs_dim]
      - neighbor_mask : [B, K] (可选) 1=真实邻居, 0=padding

    输出:
      - neigh_global  : [B, H]      (聚合后的邻居特征，全局邻居 embedding)
      - neigh_tokens  : [B, K, H]   (每个邻居一个 token；padding 邻居为 0)
      - attn_weights  : [B, K]      (每个邻居的注意力权重；padding 为 0)
    """
    def __init__(
        self,
        self_obs_dim: int = 4,
        neighbor_obs_dim: int = 4,
        neighbor_hidden_size: int = 128,
        num_use_neighbor_obs: int = 3,
    ) -> None:
        super().__init__()
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs

        H = neighbor_hidden_size

        # e_i = MLP([self_obs, neighbor_obs_i])  -> 邻居 token embedding
        self.embedding_mlp = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        # h_i = MLP(e_i) -> 用于做加权求和的 value
        self.neighbor_value_mlp = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        # score([e_i, e_m]) -> 标量打分（注意力 logit）
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.SiLU(),
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, 1),
        )

        # ===================== NEW: 特征融合（防止 self_obs 压过 neighbor_obs） =====================
        # 1) self / neighbor 分支分别投影到同一维度 H
        self.self_proj = nn.Sequential(
            nn.Linear(self_obs_dim, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )
        self.neigh_proj = nn.Sequential(
            nn.Linear(neighbor_obs_dim, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        # 2) self 作为条件，对 neighbor 做 FiLM 调制（gamma/beta 初始很小）
        self.film = nn.Linear(H, 2 * H)
        nn.init.xavier_uniform_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.gamma_scale = 0.1
        self.beta_scale = 0.1

        # 3) 可学习残差强度：保证表达能力，同时初期不会被 self 盖过
        self.alpha = nn.Parameter(torch.tensor(0.05))

        # 4) 你要的：最后对 token 再做一次 LN（稳定 scale）
        self.token_ln = nn.LayerNorm(H, eps=1e-6)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-6):
        """
        scores: [B, K]
        mask  : [B, K]  (0/1 或 bool)
        返回:
          weights: [B, K]，当某个 batch 没有有效邻居时，全 0（不会泄漏 padding）
        """
        mask = mask.to(dtype=scores.dtype)

        # 数值稳定：减去每行最大值（不影响 softmax）
        scores = scores - scores.max(dim=dim, keepdim=True).values

        exp_scores = torch.exp(scores) * mask              # 无效邻居直接置 0
        denom = exp_scores.sum(dim=dim, keepdim=True).clamp_min(eps)
        weights = exp_scores / denom

        # 关键：如果整行 mask 全 0，则 exp_scores 全 0 -> weights 全 0（不会变成均匀分布）
        return weights

    def forward(self, self_obs, neighbor_obs, neighbor_mask=None, return_tokens: bool = True):
        """
        self_obs      : [B, Ds]
        neighbor_obs  : [B, K, Dn]
        neighbor_mask : [B, K] (可选)
        """
        B, K, Dn = neighbor_obs.shape
        assert K == self.num_use_neighbor_obs, f"K mismatch: {K} vs {self.num_use_neighbor_obs}"
        assert Dn == self.neighbor_obs_dim, f"neighbor_obs_dim mismatch: {Dn} vs {self.neighbor_obs_dim}"

        # --------- mask 处理 ---------
        if neighbor_mask is None:
            neighbor_mask = torch.ones(B, K, device=neighbor_obs.device, dtype=torch.bool)
        else:
            neighbor_mask = neighbor_mask.to(device=neighbor_obs.device).bool()

        mask_f = neighbor_mask.to(dtype=neighbor_obs.dtype).unsqueeze(-1)  # [B,K,1]

        # --------- 1) e_i: 每个邻居 token embedding ---------
        # =============== NEW: self 条件化融合（neighbor 为主，self 不盖过） ===============
        # neighbor base token: [B,K,H]
        n0 = self.neigh_proj(neighbor_obs.reshape(B * K, Dn)).reshape(B, K, -1)

        # self condition: [B,H]
        s = self.self_proj(self_obs)  # self 对所有邻居相同，但只用来调制，不直接拼接进 token 主体

        # FiLM 参数: [B,H] -> expand 到 [B,K,H]
        gb = self.film(s)                                # [B,2H]
        gamma, beta = gb.chunk(2, dim=-1)                # [B,H], [B,H]
        gamma = self.gamma_scale * torch.tanh(gamma)     # 小幅度
        beta  = self.beta_scale  * torch.tanh(beta)

        gamma = gamma.unsqueeze(1).expand(B, K, -1)       # [B,K,H]
        beta  = beta.unsqueeze(1).expand(B, K, -1)        # [B,K,H]

        # neighbor 被 self 调制，但用 residual blend 控制强度（alpha 初始小）
        n1 = n0 * (1.0 + gamma) + beta                    # [B,K,H]
        e_i = n0 + self.alpha * (n1 - n0)                 # [B,K,H]

        # padding 的 token 直接清零（避免后续任何泄漏）
        e_i = e_i * mask_f

        # 再过一个 token MLP，提升表达能力
        e_i = self.embedding_mlp(e_i.reshape(B * K, -1)).reshape(B, K, -1)
        e_i = e_i * mask_f

        # 最后对 token 做一次 LN（再 mask 回 0）
        e_i = self.token_ln(e_i) * mask_f

        # --------- 2) h_i: value ---------
        h_i = self.neighbor_value_mlp(e_i.reshape(B * K, -1)).reshape(B, K, -1)  # [B,K,H]
        h_i = h_i * mask_f

        # --------- 3) e_m: 有效邻居均值 embedding ---------
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1e-6)                     # [B,1,1]
        e_m = (e_i * mask_f).sum(dim=1, keepdim=True) / denom                       # [B,1,H]
        e_m_repeat = e_m.expand(B, K, self.neighbor_hidden_size)                    # [B,K,H]

        # --------- 4) scores -> masked-softmax 权重 ---------
        att_in = torch.cat([e_i, e_m_repeat], dim=-1)                               # [B,K,2H]
        scores = self.attention_mlp(att_in.reshape(B * K, -1)).reshape(B, K)        # [B,K]

        # 只对有效邻居做 softmax（M=0 时 weights 全 0）
        attn_weights = self._masked_softmax(scores, neighbor_mask, dim=1)           # [B,K]
        attn_weights = attn_weights * neighbor_mask.to(attn_weights.dtype)          # [B,K] 再保险

        # --------- 5) 加权聚合得到全局邻居特征 ---------
        neigh_global = (h_i * attn_weights.unsqueeze(-1)).sum(dim=1)                # [B,H]

        if return_tokens:
            neigh_tokens = e_i                                                     # [B,K,H]
            return neigh_global, neigh_tokens, attn_weights
        else:
            return neigh_global

class EnvEncoder(nn.Module):
    """
    输出 4 路特征拼接后的 env embedding:

      self_feat            : self_obs 编码
      obs_global           : SDF 全局 embedding
      neigh_global         : 邻居聚合（邻居本身信息）
      neigh_delta_global   : 邻居从障碍中检索到的“增量修正”(纯 ctx 聚合)

    最终 out: [B, env_output_feature]
    """
    def __init__(
        self,
        self_obs_dim=4,
        self_obs_embed_dim=64,
        neighbor_obs_dim=4,
        neighbor_hidden_size=128,
        neighbor_base_channels=64,
        obstacle_embedding_dim=128,
        n_head=4,
        num_use_neighbor_obs=3,
        env_output_feature=256,
        dropout=0.1,
        use_sdf_grad: bool = True,
        sdf_grad_scale: float = 8.0,
        sdf_grad_clip: float | None = 1.0,
        use_ctx_ffn: bool = True,
        use_neigh_self_attn: bool = True,
    ):
        super().__init__()
        assert n_head in (4, 8), "建议只用 n_head=4 或 8"
        assert self_obs_dim % 2 == 0, "self_obs_dim 必须是偶数"

        self.self_obs_dim = self_obs_dim
        self.self_obs_embed_dim = self_obs_embed_dim
        self.half_dim = self_obs_dim // 2

        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs

        self.obstacle_embedding_dim = obstacle_embedding_dim
        self.neighbor_base_channels = neighbor_base_channels
        self.use_ctx_ffn = use_ctx_ffn
        self.use_neigh_self_attn = use_neigh_self_attn

        # 1) self 编码
        self.embed_encoder = nn.Sequential(
            nn.Linear(self_obs_dim, self_obs_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self_obs_embed_dim, self_obs_embed_dim),
        )
        self.self_ln = nn.LayerNorm(self_obs_embed_dim)

        # 2) obstacle 编码
        self.obstacle_encoder = ObstacleEncoder(
            embedding_dim=obstacle_embedding_dim,
            base_channels=neighbor_base_channels,
            use_sdf_grad=use_sdf_grad,
            sdf_grad_scale=sdf_grad_scale,
            sdf_grad_clip=sdf_grad_clip,
        )
        self.obs_global_ln = nn.LayerNorm(obstacle_embedding_dim)

        # obstacle tokens: [B,M,C] -> [B,M,d_model]
        self.obstacle_token_proj = nn.Sequential(
            nn.Linear(neighbor_base_channels, neighbor_hidden_size),
            nn.LayerNorm(neighbor_hidden_size),
        )

        # 3) neighbor 编码
        self.neighbor_encoder = NeighborEncoder(
            self_obs_dim=self_obs_dim,
            neighbor_obs_dim=neighbor_obs_dim,
            neighbor_hidden_size=neighbor_hidden_size,
            num_use_neighbor_obs=num_use_neighbor_obs,
        )
        self.neigh_global_ln = nn.LayerNorm(neighbor_hidden_size)

        # 4) CrossAttention
        assert neighbor_hidden_size % n_head == 0, "neighbor_hidden_size 必须能整除 n_head"
        head_dim = neighbor_hidden_size // n_head

        self.cross_attention = CrossAttention(
            n_head=n_head,
            d_model=neighbor_hidden_size,
            d_k=head_dim,
            d_v=head_dim,
            dropout=dropout,
        )

        # 4.5) neighbor self-attention（让邻居先交流）
        if self.use_neigh_self_attn:
            self.neigh_self_attn = nn.MultiheadAttention(
                embed_dim=neighbor_hidden_size,
                num_heads=n_head,
                dropout=dropout,
                batch_first=True,
            )
            self.neigh_self_ln = nn.LayerNorm(neighbor_hidden_size)
            self.neigh_self_alpha = nn.Parameter(torch.tensor(0.1))

        # 4.6) ctx_tokens FFN 残差块（纯增量语义不变）
        if self.use_ctx_ffn:
            self.ctx_ffn = nn.Sequential(
                nn.LayerNorm(neighbor_hidden_size),
                nn.Linear(neighbor_hidden_size, 2 * neighbor_hidden_size),
                nn.SiLU(),
                nn.Linear(2 * neighbor_hidden_size, neighbor_hidden_size),
                nn.Dropout(dropout),
            )
            self.ctx_alpha = nn.Parameter(torch.tensor(0.1))

        # 5) delta LN + 可学习缩放（你想永远保留增量也没问题，默认 1）
        self.delta_alpha = nn.Parameter(torch.tensor(1.0))
        self.delta_ln = nn.LayerNorm(neighbor_hidden_size)

        # 6) fusion
        fusion_in = (
            self_obs_embed_dim
            + obstacle_embedding_dim
            + neighbor_hidden_size
            + neighbor_hidden_size
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 2 * env_output_feature),
            nn.ReLU(inplace=True),
            nn.Linear(2 * env_output_feature, env_output_feature),
            nn.ReLU(inplace=True),
            nn.Linear(env_output_feature, env_output_feature),
        )

    def forward(self, self_obs, neighbor_obs, sdf_input, neighbor_mask):
        """
        self_obs      : [B, self_obs_dim]
        neighbor_obs  : [B, K, neighbor_obs_dim]
        sdf_input     : [B, 1, H, W]   (SDF 已经归一化到[0,1])
        neighbor_mask : [B, K] bool / 0-1
        """
        B = self_obs.size(0)
        K = neighbor_obs.size(1)

        # 0) self_obs -> start + goal_rel
        start = self_obs[:, :self.half_dim]
        goal  = self_obs[:, self.half_dim:]
        start_rel = start - start
        goal_rel = goal - start
        self_obs_proc = torch.cat([start_rel, goal_rel], dim=-1)

        # 1) self_feat
        self_feat = self.self_ln(self.embed_encoder(self_obs_proc))  # [B, Ds_emb]

        # 2) obstacle global + tokens
        obs_global, obs_tokens = self.obstacle_encoder(sdf_input)    # obs_tokens: [B,M,C]
        obs_global = self.obs_global_ln(obs_global)
        obs_tokens_proj = self.obstacle_token_proj(obs_tokens)       # [B,M,d_model]

        # 3) neighbor global + tokens + weights
        neigh_global, neigh_tokens, neigh_w = self.neighbor_encoder(
            self_obs=self_obs_proc,
            neighbor_obs=neighbor_obs,
            neighbor_mask=neighbor_mask,
            return_tokens=True,
        )
        neigh_global = self.neigh_global_ln(neigh_global)
        # neigh_tokens: [B,K,d_model] padding 已经清零
        # neigh_w     : [B,K] padding=0，有效邻居时 sum=1；无邻居时全 0

        # qmask（只用于防止 padding query 被写入非 0）
        if neighbor_mask is None:
            qmask_f = torch.ones(B, K, device=self_obs.device, dtype=neigh_tokens.dtype)
        else:
            qmask_f = neighbor_mask.to(device=self_obs.device, dtype=neigh_tokens.dtype)

        # 3.5) 邻居 self-attention（不改变 tokens 形状）
        if self.use_neigh_self_attn:
            pad_mask = None
            if neighbor_mask is not None:
                pad_mask = ~neighbor_mask  # True 表示 padding
            attn_out, _ = self.neigh_self_attn(
                neigh_tokens, neigh_tokens, neigh_tokens,
                key_padding_mask=pad_mask,
                need_weights=False,
            )
            # 如果整行都是 padding，强制置零避免 NaN
            if neighbor_mask is not None:
                valid = neighbor_mask.any(dim=1)
                if not valid.all():
                    attn_out = attn_out.clone()
                    attn_out[~valid] = 0.0
            neigh_tokens = neigh_tokens + self.neigh_self_alpha * attn_out
            neigh_tokens = self.neigh_self_ln(neigh_tokens)
            neigh_tokens = neigh_tokens * qmask_f.unsqueeze(-1)

        # 4) neighbor_tokens -> query obstacle_tokens
        # ✅ 取回纯 ctx（纯增量项）
        _, _, ctx_tokens = self.cross_attention(
            query=neigh_tokens,          # [B,K,d_model]
            key=obs_tokens_proj,         # [B,M,d_model]
            value=obs_tokens_proj,       # [B,M,d_model]
            mask=None,
            return_ctx=True,
        )  # ctx_tokens: [B,K,d_model]

        # padding query 强制清零（否则 query=0 也会拿到均值 ctx）
        ctx_tokens = ctx_tokens * qmask_f.unsqueeze(-1)

        # ctx_tokens FFN 残差（保持“纯增量”语义）
        if self.use_ctx_ffn:
            ctx_tokens = ctx_tokens + self.ctx_alpha * self.ctx_ffn(ctx_tokens)
            ctx_tokens = ctx_tokens * qmask_f.unsqueeze(-1)

        # --- 5) 聚合 “障碍感知的增量修正” ---
        ctx_strength = ctx_tokens.norm(p=2, dim=-1)                      # [B, K]
        w_obst = neigh_w * (ctx_strength + 1e-3)                         # [B, K]  障碍感知权重
        w_obst = w_obst / w_obst.sum(dim=1, keepdim=True).clamp_min(1e-6)

        neigh_delta_global = (ctx_tokens * w_obst.unsqueeze(-1)).sum(dim=1)  # [B, d_model]
        neigh_delta_global = self.delta_ln(self.delta_alpha * neigh_delta_global)

        # 6) 4 路融合
        fused = torch.cat(
            [self_feat, obs_global, neigh_global, neigh_delta_global],
            dim=-1,
        )
        out = self.fusion_mlp(fused)
        return out

class FiLM1d(nn.Module):
    """
    对向量特征 z: [N, D] 做 FiLM: y = (1 + gamma) ⊙ z + beta
    - gamma/beta 由 cond 生成，线性层零初始化，tanh 限幅并乘小尺度
    """
    def __init__(self, feat_dim: int, cond_dim: int, gamma_scale: float = 0.1, beta_scale: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale

    def forward(self, z: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        gb = self.proj(cond_emb)                # [N, 2D]
        gamma, beta = gb.chunk(2, dim=1)        # [N, D], [N, D]
        gamma = self.gamma_scale * torch.tanh(gamma)
        beta  = self.beta_scale  * torch.tanh(beta)
        return z * (1 + gamma) + beta

class LateFusionHead(nn.Module):
    """
    mode:
      - 'film'
      - 'film_no_res'
      - 'gated'
      - 'sigmoid'
      - 'concat'
      - 'concat_ln'   # NEW: 分别 LN 后再 concat
    """
    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        out_dim: int,
        mode: str = "film",
        gate_bias_init: float = 2.0,
        gate_hidden: int = 256,
    ):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in (
            "film", "gated", "concat", "film_no_res", "sigmoid", "concat_ln"
        ), f"未知融合模式: {self.mode}"

        if self.mode == "film":
            self.film = FiLM1d(z_dim, c_dim, gamma_scale=0.1, beta_scale=0.1)
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.post = nn.Sequential(
                nn.LayerNorm(z_dim), nn.ReLU(inplace=True),
                nn.Linear(z_dim, out_dim), nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)
            )

        elif self.mode == "film_no_res":
            self.film = FiLM1d(z_dim, c_dim, gamma_scale=0.1, beta_scale=0.1)
            self.post = nn.Sequential(
                nn.LayerNorm(z_dim), nn.ReLU(inplace=True),
                nn.Linear(z_dim, out_dim), nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)
            )

        elif self.mode == "gated":
            self.gate = nn.Linear(c_dim, z_dim)
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.post = nn.Sequential(
                nn.LayerNorm(z_dim), nn.ReLU(inplace=True),
                nn.Linear(z_dim, out_dim), nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)
            )

        elif self.mode == "sigmoid":
            gate_in = z_dim + c_dim
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, z_dim),
            )
            nn.init.constant_(self.gate[-1].bias, gate_bias_init)

            self.c_proj = nn.Linear(c_dim, z_dim)
            nn.init.xavier_uniform_(self.c_proj.weight)
            nn.init.zeros_(self.c_proj.bias)

            self.post = nn.Sequential(
                nn.LayerNorm(z_dim), nn.ReLU(inplace=True),
                nn.Linear(z_dim, out_dim), nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)
            )

        elif self.mode == "concat_ln":
            # 分别对两路做 LayerNorm，再 concat
            self.z_ln = nn.LayerNorm(z_dim)
            self.c_ln = nn.LayerNorm(c_dim)
            self.post = nn.Sequential(
                nn.Linear(z_dim + c_dim, 2 * out_dim), nn.LayerNorm(2 * out_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * out_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )

        else:  # concat
            self.post = nn.Sequential(
                nn.Linear(z_dim + c_dim, 2 * out_dim), nn.LayerNorm(2 * out_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * out_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.mode == "film":
            z_tilde = self.film(z, c)
            z = z + self.alpha * (z_tilde - z)
            return self.post(z)

        elif self.mode == "film_no_res":
            z_tilde = self.film(z, c)
            return self.post(z_tilde)

        elif self.mode == "gated":
            g = torch.tanh(self.gate(c)) * 0.1
            z_tilde = z * (1 + g)
            z = z + self.alpha * (z_tilde - z)
            return self.post(z)

        elif self.mode == "sigmoid":
            g = torch.sigmoid(self.gate(torch.cat([z, c], dim=1)))  # [N, z_dim]
            c_z = self.c_proj(c)                                    # [N, z_dim]
            z_mix = g * z + (1.0 - g) * c_z
            return self.post(z_mix)

        elif self.mode == "concat_ln":
            # 如果 c 来自冻结先验，这里可选：c = c.detach()
            z = self.z_ln(z)
            c = self.c_ln(c)
            return self.post(torch.cat([z, c], dim=1))

        else:  # concat
            return self.post(torch.cat([z, c], dim=1))

class RLFields(nn.Module):
    def __init__(
        self, 
        self_obs_dim: int = 4,
        self_obs_embed_dim: int = 64, 
        neighbor_obs_dim: int = 4, 
        neighbor_hidden_size: int = 128, 
        neighbor_base_channels: int = 64,
        obstacle_embedding_dim: int = 128, 
        n_head: int = 4, 
        num_use_neighbor_obs: int = 3,
        env_output_feature: int = 256,
        fusion_mode: str = "film",
        # -------- SDF/Attention 增强 --------
        use_sdf_grad: bool = True,
        sdf_grad_scale: float = 8.0,
        sdf_grad_clip: float | None = 1.0,
        use_ctx_ffn: bool = True,
        use_neigh_self_attn: bool = True,
        # -------- NTFields 配置 --------
        nt_dim: int = 2,
        nt_vmax: float = 1.0,
        nt_h_size: int = 256,
        nt_enc_d: int = 7,
        nt_dec_d: int = 7,
        nt_full_coords_enc: bool = False,
        nt_basic_block: str = "resnet",
        nt_use_unet: bool = False,
        nt_symm_op_type: str = "min_max",
        nt_pretrained_path: str = None,
        device: str = "cuda"):

        super(RLFields, self).__init__()
        # RLFields 自身配置
        self.self_obs_dim = self_obs_dim
        self.self_obs_embed_dim = self_obs_embed_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.neighbor_base_channels = neighbor_base_channels
        self.obstacle_embedding_dim = obstacle_embedding_dim
        self.n_head = n_head
        self.num_use_neighbor_obs = num_use_neighbor_obs
        self.env_output_feature = env_output_feature
        self.fusion_mode = fusion_mode
        self.use_sdf_grad = use_sdf_grad
        self.sdf_grad_scale = sdf_grad_scale
        self.sdf_grad_clip = sdf_grad_clip
        self.use_ctx_ffn = use_ctx_ffn
        self.use_neigh_self_attn = use_neigh_self_attn

        # NTFields config
        self.nt_dim = nt_dim
        self.nt_vmax = nt_vmax
        self.nt_h_size = nt_h_size
        self.nt_enc_d = nt_enc_d
        self.nt_dec_d = nt_dec_d
        self.nt_full_coords_enc = nt_full_coords_enc
        self.nt_basic_block = nt_basic_block
        self.nt_use_unet = nt_use_unet
        self.nt_symm_op_type = nt_symm_op_type

        self.device = torch.device(device)

        # -------- 环境编码器 --------
        self.env_encoder = EnvEncoder(
            self_obs_dim=self.self_obs_dim,
            self_obs_embed_dim=self.self_obs_embed_dim,
            neighbor_obs_dim=self.neighbor_obs_dim,
            neighbor_hidden_size=self.neighbor_hidden_size,
            neighbor_base_channels=self.neighbor_base_channels,
            obstacle_embedding_dim=self.obstacle_embedding_dim,
            n_head=self.n_head,
            num_use_neighbor_obs=self.num_use_neighbor_obs,
            env_output_feature=self.env_output_feature,
            use_sdf_grad=self.use_sdf_grad,
            sdf_grad_scale=self.sdf_grad_scale,
            sdf_grad_clip=self.sdf_grad_clip,
            use_ctx_ffn=self.use_ctx_ffn,
            use_neigh_self_attn=self.use_neigh_self_attn,
        )

        # -------- NTFields 主干 --------
        self.ntfields_nn = NN(
            device=self.device,
            dim=self.nt_dim,
            vmax=self.nt_vmax,
            h_size=self.nt_h_size,
            enc_d=self.nt_enc_d,
            dec_d=self.nt_dec_d,
            full_coords_enc=self.nt_full_coords_enc,
            basic_block=self.nt_basic_block,
            use_unet=self.nt_use_unet,
            symm_op_type=self.nt_symm_op_type,
        )

        # NTFields 对称后的特征维度
        # 对于 symm_op_type='min_max'，out_feature 的维度是 2 * h_size
        if self.ntfields_nn.symm_op_type == 'f':
            self.nt_feat_dim = self.nt_h_size
        else:
            self.nt_feat_dim = 2 * self.nt_h_size  # min_max
        
        # 加载预训练权重
        if nt_pretrained_path is not None:
            ckpt = torch.load(nt_pretrained_path, map_location=self.device)
            self.ntfields_nn.load_state_dict(ckpt["model_state_dict"], strict=True)
        
        # 冻结 NTFields 参数
        for p in self.ntfields_nn.parameters():
            p.requires_grad_(False)
        self.ntfields_nn.eval()    # 里面如果有 BN/Dropout，就固定住

        self.ntfields_feat_proj = nn.Sequential(
            nn.Linear(self.nt_feat_dim, self.env_output_feature),
            nn.ReLU(),
            nn.Linear(self.env_output_feature, self.env_output_feature),
        )

        # 门控fuser
        self.late_fusion = LateFusionHead(
            z_dim=self.env_output_feature,
            c_dim=self.env_output_feature,
            out_dim=self.env_output_feature,
            mode=self.fusion_mode
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        # -------- 1) 解析 self_obs --------
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

        idx = torch.arange(self.num_use_neighbor_obs, device=observations.device).unsqueeze(0)  # [1, K]
        idx = idx.expand(B, -1)                                  # [B, K]
        neighbor_mask = idx < neighbor_valid_num.unsqueeze(-1)   # [B, K] bool

        # -------- 3) 解析 SDF + prior_dir --------
        # obs 的最后 nt_dim 维是 env 预先计算好的 prior_dir，避免在 model 内重复计算。
        prior_dir = observations[:, -self.nt_dim:]               # [B, nt_dim]
        sdf_flat = observations[:, offset:-self.nt_dim]          # [B, H*H]
        sdf_len = sdf_flat.shape[1]
        H = int(math.sqrt(sdf_len))
        assert H * H == sdf_len, f"sdf_input 展平长度 {sdf_len} 不能组成正方形图像"
        sdf_input = sdf_flat.view(B, 1, H, H)                    # [B, 1, H, H]

        # -------- 4) 环境编码 --------
        env_embedding = self.env_encoder(
            self_obs=self_obs,
            neighbor_obs=neighbor_obs,
            sdf_input=sdf_input,
            neighbor_mask=neighbor_mask,
        )                                                        # [B, env_output_feature]

        # -------- 5) NTFields 特征（用 self_obs 作为 coords） --------
        nt_feat, _ = self.ntfields_nn.out_feature(self_obs)      # [B, nt_feat_dim]
        nt_feat = self.ntfields_feat_proj(nt_feat)

        # -------- 6) LateFusionHead 门控融合 --------
        fused_feat = self.late_fusion(env_embedding, nt_feat)    # [B, nt_feat_dim]

        return fused_feat, prior_dir
    
# PC_Planner网络
class NN(nn.Module):
    """
    NTFields 主网络:
    - 支持 encoder + generator 或 U-Net 两种结构
    - 对输入 coords 做对称操作 (x0, x1) ↔ (x1, x0)，并用 symm_op_type 指定组合方式
    """
    def __init__(
        self,
        device,
        dim: int,
        vmax: float,
        h_size: int,
        enc_d: int,
        dec_d: int,
        full_coords_enc: bool,
        basic_block: str,
        use_unet: bool,
        symm_op_type: str,
    ) -> None:
        """
        参数拆开说明:
            device          : 设备 (cpu / cuda)
            dim             : 单个坐标维度 (例如 2 或 3)
            vmax            : 输出 tau 的最大值（用在 sigmoid 缩放）
            h_size          : 主隐藏维度 (原 args.h_size)
            enc_d           : encoder 深度 (原 args.enc_d)
            dec_d           : decoder / generator 深度 (原 args.dec_d)
            full_coords_enc : 是否使用 full_coords_enc (原 args.full_coords_enc)
            basic_block     : 'resnext' / 'resnet' / 'unet' (原 args.basic_block)
            use_unet        : 是否使用 U-Net 结构 (原 args.use_unet)
            symm_op_type    : 'min_max' / 'mean' / 'f' 等 (原 args.symm_op_type)
        """
        super().__init__()

        self.dim = dim
        self.vmax = vmax
        self.full_coords_enc = full_coords_enc
        self.device = device

        self.scale = 10
        self.act = nn.ELU()

        # 保留原来的名字 nl1 / nl2
        self.nl1 = enc_d           # encoder 深度
        self.nl2 = dec_d           # decoder / generator 深度

        # 记录结构类型（原本在 args 里）
        self.basic_block_type = basic_block
        self.use_unet = use_unet
        self.symm_op_type = symm_op_type

        self.h_size = h_size

        # 坐标编码维度
        if self.full_coords_enc:
            p_dim = dim * 2
        else:
            p_dim = dim

        # 选择基本 block 类型
        self.basic_blocks = {
            'resnext': C_encoder_block_next,
            'resnet': C_encoder_block,
            'unet': UnetBaiscBlock,
        }
        block_cls = self.basic_blocks[self.basic_block_type]

        # ---------------- encoder ----------------
        if self.use_unet:
            # U-Net 风格 encoder
            self.enc0 = nn.Sequential(
                nn.Linear(p_dim, h_size),
                self.act
            )
            self.encoder_blocks = nn.ModuleList([
                UnetBaiscBlock(h_size, 32, 32),
                UnetBaiscBlock(32, 64, 64),
                UnetBaiscBlock(64, 128, 128),
                UnetBaiscBlock(128, 256, 256),
            ])
        else:
            # 普通 encoder: 多个 residual blocks
            self.encoder = nn.Sequential(
                nn.Linear(p_dim, h_size),
                self.act,
                *[block_cls(h_size, h_size, h_size) for _ in range(self.nl1 - 1)],
                nn.Linear(h_size, h_size),
            )

        # ---------------- decoder / generator ----------------
        if self.use_unet:
            # U-Net decoder
            self.dec0 = nn.Sequential(
                nn.Linear(2 * h_size, 2 * h_size),
                self.act,
            )
            self.decoder_blocks = nn.ModuleList([
                UnetBaiscBlock(1024, 256, 256),
                UnetBaiscBlock(512, 128, 128),
                UnetBaiscBlock(256, 64, 64),
                UnetBaiscBlock(128, 32, 32),
            ])
            self.dec1 = nn.Linear(32, 1)
        else:
            # 普通 generator
            if self.symm_op_type == 'f':
                gen0_in_size = h_size
            else:
                gen0_in_size = 2 * h_size

            self.gen0 = nn.Sequential(
                nn.Linear(gen0_in_size, 2 * h_size),
                self.act,
            )
            self.generator = nn.Sequential(
                *[block_cls(2 * h_size, 2 * h_size, 2 * h_size) for _ in range(self.nl2 - 1)],
                nn.Linear(2 * h_size, h_size),
                self.act,
                nn.Linear(h_size, 1),
            )

    def init_weights(self, m: nn.Module) -> None:
        """
        自定义初始化：仅作用于 nn.Linear
        在外部调用: model.apply(model.init_weights)
        """
        if isinstance(m, nn.Linear):
            stdv = (1.0 / math.sqrt(m.weight.size(1)) / 1.0) * 2.0
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)

    def symm_op(self, x: torch.Tensor) -> torch.Tensor:
        """
        对 2*size 个样本做对称操作:
        - 前一半视为 x0，后一半视为 x1
        - 输出 cat(max(x0,x1), min(x0,x1)) 作为对称特征
        输入:  [2*size, C]
        输出:  [size, 2*C]
        """
        size = x.shape[0] // 2
        x0 = x[:size, ...]
        x1 = x[size:, ...]

        x_0 = torch.max(x0, x1)
        x_1 = torch.min(x0, x1)

        y = torch.cat((x_0, x_1), dim=1)
        return y

    def _encode_and_symm(
        self,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """
        通用的 encode + 对称组合流程，供 out / out_feature 共用。
        返回:
            x             : 对称后的特征
            coords        : 
            skip_connections : 仅在 use_unet=True 时非 None
        """
        size = coords.shape[0]

        # 构造 (x0, x1) 对
        if self.full_coords_enc:
            x0 = coords
            x1 = torch.cat((coords[:, self.dim:], coords[:, :self.dim]), dim=1)
        else:
            x0 = coords[:, :self.dim]
            x1 = coords[:, self.dim:]

        x = torch.vstack((x0, x1))

        # encoder
        skip_connections = None
        if self.use_unet:
            x = self.enc0(x)
            skip_connections = []
            for enc_block in self.encoder_blocks:
                x = enc_block(x)
                skip_connections.append(self.symm_op(x))
        else:
            x = self.encoder(x)

        # 拆回两半
        x0 = x[:size, ...]
        x1 = x[size:, ...]

        # 对称组合
        if self.symm_op_type == 'min_max':
            x_0 = torch.max(x0, x1)
            x_1 = torch.min(x0, x1)
            x = torch.cat((x_0, x_1), 1)
        elif self.symm_op_type == 'mean':
            x = (torch.cat((x0, x1), 1) + torch.cat((x1, x0), 1)) / 2.0
        elif self.symm_op_type == 'f':
            # f 可以换成更复杂的对称函数，这里用简单平均
            f = lambda a, b: (a + b) / 2.0
            x = f(x0, x1)
        else:
            # 未知类型时回退到 min_max 逻辑
            x_0 = torch.max(x0, x1)
            x_1 = torch.min(x0, x1)
            x = torch.cat((x_0, x_1), 1)

        return x, coords, skip_connections

    def out(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算标量输出（tau），并保留 coords 的梯度:
        返回:
            x      : [N, 1]
            coords : [N, ...] (requires_grad=True)
        """
        x, coords, skip_connections = self._encode_and_symm(coords)

        if self.use_unet:
            x = self.dec0(x)
            for dec_block in self.decoder_blocks:
                skip_x = skip_connections.pop()
                x = torch.cat((x, skip_x), dim=-1)
                x = dec_block(x)
            x = self.dec1(x)
        else:
            x = self.gen0(x)
            x = self.generator(x)

        x = self.vmax * torch.sigmoid(0.1 * x)
        return x, coords

    def gradient(self, y, x, create_graph=True):                                                               
        grad_y = torch.ones_like(y)                                                                 
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        return grad_x
    
    def calculate_gradient(self, Xp: torch.Tensor) -> torch.Tensor:
        """
        Xp: [B, 2*dim] = [start, goal]
        返回: [B, 2*dim] = [dT/dx_start(dim), dT/dx_goal(dim)]
        仅作为先验使用，不让 RL 反向传播到 NTFields。
        """
        dim = self.dim
        eps_T0   = 1e-8    # 距离下限
        eps_tau  = 1e-4    # tau 下限
        eps_norm = 1e-6    # 梯度范数下限

        with torch.enable_grad():  # 只对这里建图
            # 只对输入建图，不让外部 graph 传进来
            Xp_in = Xp.clone().detach().requires_grad_(True)  # [B, 2*dim]

            # tau: [B,1]
            tau, Xp_in = self.out(Xp_in)
            # 防止 tau 太小
            tau_safe = tau.clamp_min(eps_tau)
            T3 = tau_safe ** 2  # [B,1]

            # d tau / d Xp
            dtau = self.gradient(tau_safe, Xp_in, create_graph=False)  # [B, 2*dim]

            # 几何量
            D  = Xp_in[:, dim:] - Xp_in[:, :dim]        # [B, dim] = goal - start
            T0 = torch.sqrt((D * D).sum(dim=1, keepdim=True) + eps_T0)  # [B,1]

            # ----- dT/d(goal) 相关 -----
            V0 = D                        # [B,dim]
            V1 = dtau[:, dim:]            # [B,dim]

            Y1 = V0 / (T0 * tau_safe)     # [B,dim]
            Y2 = (T0 / T3) * V1           # [B,dim]

            Ypred1 = -(Y1 - Y2)           # [B,dim]
            Spred1 = Ypred1.norm(dim=1, keepdim=True).clamp_min(eps_norm)
            Ypred1 = Ypred1 / (Spred1 ** 2)

            # ----- dT/d(start) 相关 -----
            V0 = -D                       # [B,dim]
            V1 = dtau[:, :dim]            # [B,dim]

            Y1 = V0 / (T0 * tau_safe)
            Y2 = (T0 / T3) * V1

            Ypred0 = -(Y1 - Y2)           # [B,dim]
            Spred0 = Ypred0.norm(dim=1, keepdim=True).clamp_min(eps_norm)
            Ypred0 = Ypred0 / (Spred0 ** 2)

            grad_all = torch.cat([Ypred0, Ypred1], dim=1)  # [B, 2*dim]

        # 兜底：把 NaN/Inf 全部替换成有限数
        grad_all = torch.nan_to_num(
            grad_all,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        )

        return grad_all.detach()

    def out_feature(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        只提取中间对称后的特征，不走解码器
        返回:
            x      : 对称特征
            coords : 原始 coords
        """
        x, coords, _ = self._encode_and_symm(coords)
        return x, coords

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch 标准接口，等价于调用 out():
        返回:
            output : [N,1]
            coords : [N,...] (requires_grad=True)
        """
        output, coords = self.out(coords)
        return output, coords
