import torch
from torch import nn
import math
from .model_blocks import C_encoder_block, C_encoder_block_next, CrossAttention, UnetBaiscBlock

class ObstacleEncoder(nn.Module):
    """
    单通道 SDF 编码器（InstanceNorm2d，batch=1 友好）。
    输入的esdf需要归一化到[0,1]
    Conv2d(bias=False) → IN → SiLU；两次 stride=2 下采样 + 一次膨胀卷积。
    输出: [B, embedding_dim]
    """
    def __init__(self, 
                 embedding_dim: int = 128, 
                 base_channels: int = 32)-> None:
        super().__init__()
        C = base_channels
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
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

    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        if sdf.dim() == 3:  # [B,H,W] -> [B,1,H,W]
            sdf = sdf.unsqueeze(1)
        feats = self.feature_extractor(sdf)                 # [B, C, 20, 20]
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
        num_use_neighbor_obs: int = 6,
    ) -> None:
        super().__init__()
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs

        H = neighbor_hidden_size

        # e_i = MLP([self_obs, neighbor_obs_i])  -> 邻居 token embedding
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self_obs_dim + neighbor_obs_dim, H),
            nn.SiLU(),
            nn.Linear(H, H),
            nn.SiLU(),
        )

        # h_i = MLP(e_i) -> 用于做加权求和的 value
        self.neighbor_value_mlp = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, H),
            nn.SiLU(),
        )

        # score([e_i, e_m]) -> 标量打分（注意力 logit）
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.SiLU(),
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, 1),
        )

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
        self_obs_expanded = self_obs.unsqueeze(1).expand(B, K, self.self_obs_dim)   # [B,K,Ds]
        mlp_input = torch.cat([self_obs_expanded, neighbor_obs], dim=-1)            # [B,K,Ds+Dn]
        e_i = self.embedding_mlp(mlp_input.view(B * K, -1)).view(B, K, -1)          # [B,K,H]

        # padding 的 token 直接清零（避免后续任何泄漏）
        e_i = e_i * mask_f

        # --------- 2) h_i: value ---------
        h_i = self.neighbor_value_mlp(e_i.view(B * K, -1)).view(B, K, -1)           # [B,K,H]
        h_i = h_i * mask_f

        # --------- 3) e_m: 有效邻居均值 embedding ---------
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1e-6)                     # [B,1,1]
        e_m = (e_i * mask_f).sum(dim=1, keepdim=True) / denom                       # [B,1,H]
        e_m_repeat = e_m.expand(B, K, self.neighbor_hidden_size)                    # [B,K,H]

        # --------- 4) scores -> masked-softmax 权重 ---------
        att_in = torch.cat([e_i, e_m_repeat], dim=-1)                               # [B,K,2H]
        scores = self.attention_mlp(att_in.view(B * K, -1)).view(B, K)              # [B,K]

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
        self_obs_embed_dim=128,
        neighbor_obs_dim=4,
        neighbor_hidden_size=128,
        neighbor_base_channels=32,
        obstacle_embedding_dim=128,
        n_head=4,
        num_use_neighbor_obs=6,
        env_output_feature=128,
        dropout=0.1,
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

        # 1) self 编码
        self.embed_encoder = nn.Sequential(
            nn.Linear(self_obs_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self_obs_embed_dim),
        )
        self.self_ln = nn.LayerNorm(self_obs_embed_dim)

        # 2) obstacle 编码
        self.obstacle_encoder = ObstacleEncoder(
            embedding_dim=obstacle_embedding_dim,
            base_channels=neighbor_base_channels,
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
            nn.Linear(fusion_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, env_output_feature),
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
        goal_rel = goal - start
        self_obs_proc = torch.cat([start, goal_rel], dim=-1)

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

        # 4) neighbor_tokens -> query obstacle_tokens
        # ✅ 取回纯 ctx（纯增量项）
        _, attn, ctx_tokens = self.cross_attention(
            query=neigh_tokens,          # [B,K,d_model]
            key=obs_tokens_proj,         # [B,M,d_model]
            value=obs_tokens_proj,       # [B,M,d_model]
            mask=None,
            return_ctx=True,
        )  # ctx_tokens: [B,K,d_model]
        
        ctx_tokens = ctx_tokens * qmask_f.unsqueeze(-1)

        # --- 5) 聚合 “障碍感知的增量修正” ---
        ctx_strength = ctx_tokens.norm(p=2, dim=-1)                      # [B, K]
        w_obst = neigh_w * (ctx_strength + 1e-3)                         # [B, K]  障碍感知权重
        w_obst = w_obst / w_obst.sum(dim=1, keepdim=True).clamp_min(1e-6)

        neigh_delta_global = (ctx_tokens * w_obst.unsqueeze(-1)).sum(dim=1)  # [B, d_model]
        neigh_delta_global = self.delta_ln(self.delta_alpha * neigh_delta_global)
        
        from utils.eval_attention import CrossAttnSDFVis, analyze_cross_attn_vs_sdf, analyze_attn_similarity,debug_query_attn_ctx_similarity, analyze_query_attn_ctx_invariance, debug_attn_entropy_and_logits
        vis = CrossAttnSDFVis(sdf_scale=4.0, attn_alpha=0.55)
    
        # # 聚合 attention（并列SDF）
        # vis.show_aggregated(
        #     sdf_input=sdf_input,             # ✅ 直接传网络输入的SDF
        #     attn_weights=attn,
        #     neighbor_mask=neighbor_mask,     # 建议传
        #     b=0,
        #     agg="mean_valid",
        #     extent=None,                     # 你说不要坐标的话就None
        #     fname="logs/attn_agg_pair.png",
        # )

        # # 单邻居 attention（强烈建议逐个看）
        # vis.show_single_neighbor(
        #     sdf_input=sdf_input,
        #     attn_weights=attn,
        #     b=0,
        #     k=2,
        #     head=None,
        #     extent=None,
        #     fname="logs/attn_k2_pair.png",
        # )

        # vis.show_valid_neighbors_heads_grid(
        #     sdf_input=sdf_input,
        #     attn_weights=attn,
        #     neighbor_mask=neighbor_mask,
        #     b=0,
        #     extent=None,
        #     out_dir="logs",
        #     prefix="attn_grid",
        # )

        # analyze_attn_similarity(
        #     attn_weights=attn,                 # [B,H,K,M]
        #     neigh_tokens=neigh_tokens,         # [B,K,d_model]
        #     ctx_tokens=ctx_tokens,             # [B,K,d_model]
        #     neighbor_mask=neighbor_mask,       # [B,K] bool
        #     cross_attn_module=self.cross_attention,
        #     b=0,
        #     topk=30,
        # )

        # import time
        # time.sleep(2)

        # debug_query_attn_ctx_similarity(
        #     neigh_tokens=neigh_tokens.detach(),
        #     attn=attn.detach(),
        #     ctx_tokens=ctx_tokens.detach(),
        #     neighbor_mask=neighbor_mask,
        #     b=0,
        #     pair_mode="all_pairs",  # 或 "vs_first"
        #     print_top=30,
        # )

        # analyze_query_attn_ctx_invariance(
        #     cross_attn=self.cross_attention,
        #     neigh_tokens=neigh_tokens,
        #     obs_tokens_proj=obs_tokens_proj,
        #     attn_weights=attn,
        #     ctx_tokens=ctx_tokens,
        #     neighbor_mask=neighbor_mask,
        #     b=0,
        # )

        # debug_attn_entropy_and_logits(self.cross_attention, neigh_tokens, obs_tokens_proj, attn, neighbor_mask, b=0)


        # # neigh_w 聚合
        # analyze_cross_attn_vs_sdf(
        #     sdf_input=sdf_input,
        #     attn_weights=attn,
        #     neighbor_mask=neighbor_mask,
        #     neighbor_weights=neigh_w.detach(),
        #     b=0,
        # )

        # # w_obst 聚合（更贴近你最终 delta 聚合逻辑）
        # analyze_cross_attn_vs_sdf(
        #     sdf_input=sdf_input,
        #     attn_weights=attn,
        #     neighbor_mask=neighbor_mask,
        #     neighbor_weights=w_obst.detach(),
        #     b=0,
        # )
        # print('tokens', neigh_tokens)
        # print('weight', attn)
        # debug_projected_q_cos(self.cross_attention, neigh_tokens, neighbor_mask, b=0, k0=0, k1=1)
        # debug_value_token_diversity(obs_tokens_proj, b=0)

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
                nn.Linear(z_dim + c_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
                nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )

        else:  # concat
            self.post = nn.Sequential(
                nn.Linear(z_dim + c_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
                nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True),
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
        self_obs_embed_dim: int = 128, 
        neighbor_obs_dim: int = 4, 
        neighbor_hidden_size: int = 128, 
        neighbor_base_channels: int = 32,
        obstacle_embedding_dim: int = 128, 
        n_head: int = 4, 
        num_use_neighbor_obs: int = 6,
        env_output_feature: int = 128,
        fusion_mode: str = "film",
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
            env_output_feature=self.env_output_feature
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

        # 门控fuser
        self.late_fusion = LateFusionHead(
            z_dim=self.env_output_feature,
            c_dim=self.nt_feat_dim,
            out_dim=self.nt_feat_dim,
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
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==== 配置，需和 RLFields 里保持一致 ====
    self_obs_dim = 4              # [start(2), goal(2)]
    neighbor_obs_dim = 4
    num_use_neighbor_obs = 6
    H = 80                        # SDF 分辨率 HxH，这里用 80x80

    # nt_dim 要和 self_obs_dim 对应：self_obs_dim == 2 * nt_dim
    nt_dim = 2

    # ==== 构建模型 ====
    model = RLFields(
        self_obs_dim=self_obs_dim,
        self_obs_embed_dim=128,
        neighbor_obs_dim=neighbor_obs_dim,
        neighbor_hidden_size=128,
        neighbor_base_channels=32,
        obstacle_embedding_dim=128,
        n_head=6,
        num_use_neighbor_obs=num_use_neighbor_obs,
        env_output_feature=128,
        fusion_mode="film",
        nt_dim=nt_dim,
        nt_vmax=1.0,
        nt_h_size=256,
        nt_enc_d=7,
        nt_dec_d=7,
        nt_full_coords_enc=False,
        nt_basic_block="resnet",
        nt_use_unet=False,
        nt_symm_op_type="min_max",
        device=str(device),
    ).to(device)

    model.train()

    # ==== 构造假输入 ====
    batch_size = 3

    neigh_flat_dim = num_use_neighbor_obs * neighbor_obs_dim
    sdf_flat_dim = H * H
    obs_dim_total = self_obs_dim + neigh_flat_dim + 1 + sdf_flat_dim

    # 全部先置零
    observations = torch.zeros(batch_size, obs_dim_total, device=device)

    # 1) self_obs: [start, goal]
    # 这里随便造一点在 [-1, 1] 的值
    offset = 0
    observations[:, offset: offset + self_obs_dim] = (
        torch.rand(batch_size, self_obs_dim, device=device) * 2.0 - 1.0
    )
    offset += self_obs_dim

    # 2) neighbor_obs_flat: [B, K * neighbor_obs_dim]
    observations[:, offset: offset + neigh_flat_dim] = torch.randn(
        batch_size, neigh_flat_dim, device=device
    )
    offset += neigh_flat_dim

    # 3) neighbor_valid_num: [B]，0~K 之间的整数
    neighbor_valid_num = torch.randint(
        low=0,
        high=num_use_neighbor_obs + 1,
        size=(batch_size,),
        device=device,
    ).float()
    observations[:, offset] = neighbor_valid_num
    offset += 1

    # 4) sdf_flat: [B, H*H]，在 [0,1] 范围
    observations[:, offset:] = torch.rand(
        batch_size, sdf_flat_dim, device=device
    )

    observations.requires_grad_(True)

    # ==== 前向 ====
    tau = model(observations)
    print("tau shape:", tau.shape)
    print("tau:", tau)

    # ==== 反向，看下梯度是否能正常传播 ====
    loss = tau.mean()
    loss.backward()

    # 打印一部分 grad 确认不是 None
    print("observations.grad[0, :10]:", observations.grad[0, :10])

    total_grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item()
    print("total grad norm over parameters:", total_grad_norm)