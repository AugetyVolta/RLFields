import math
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CrossAttnSDFVis:
    """
    并列可视化（左：SDF米制；右：SDF + Attention）

    直接传 sdf_input（网络输入的归一化SDF）:
      sdf_input: [B,1,H,W] 或 [B,H,W]，范围 [0,1]

    sdf_meter = sdf_input * sdf_scale（你说用4恢复到米制）
    """

    def __init__(
        self,
        sdf_scale=4.0,          # ✅ 归一化反变换：meter = norm * 4
        sdf_cmap="plasma",
        attn_cmap="viridis",
        attn_alpha=0.55,
        upsample_mode="bilinear",
        dpi=160,
    ):
        self.sdf_scale = float(sdf_scale)
        self.sdf_cmap = sdf_cmap
        self.attn_cmap = attn_cmap
        self.attn_alpha = float(attn_alpha)
        self.upsample_mode = upsample_mode
        self.dpi = int(dpi)

    # -------------------- utils --------------------
    @staticmethod
    def _infer_grid_hw(M: int):
        g = int(math.sqrt(M))
        assert g * g == M, f"M={M} 不是完全平方数，无法reshape为网格"
        return g, g

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    @torch.no_grad()
    def _get_sdf_meter(self, sdf_input, b=0):
        """
        sdf_input: [B,1,H,W] or [B,H,W]
        return: sdf_meter [H,W] (torch)
        """
        if sdf_input.dim() == 4:
            sdf_norm = sdf_input[b, 0]          # [H,W]
        elif sdf_input.dim() == 3:
            sdf_norm = sdf_input[b]             # [H,W]
        else:
            raise ValueError("sdf_input must be [B,1,H,W] or [B,H,W]")

        return sdf_norm * self.sdf_scale

    @torch.no_grad()
    def _attn_1d_single(self, attn_weights, b=0, k=0, head=None):
        """
        attn_weights: [B, head, K, M]
        return: [M]
        """
        attn_k = attn_weights[b, :, k, :]  # [head, M]
        if head is None:
            return attn_k.mean(dim=0)      # [M]
        else:
            return attn_k[head]            # [M]

    @torch.no_grad()
    def _attn_1d_agg(self, attn_weights, neighbor_mask=None, b=0, agg="mean_valid"):
        """
        聚合所有有效邻居attention：
          head均值 -> [K,M]
          padding清零
        return: [M]
        """
        attn_km = attn_weights[b].mean(dim=0)  # [K,M]

        if neighbor_mask is None:
            mask = torch.ones(attn_km.size(0), device=attn_km.device)
        else:
            mask = neighbor_mask[b].float()    # [K]

        attn_km = attn_km * mask[:, None]
        denom = mask.sum().clamp_min(1.0)

        if agg == "mean_valid":
            return attn_km.sum(dim=0) / denom
        elif agg == "sum_valid":
            return attn_km.sum(dim=0)
        else:
            raise ValueError("agg must be 'mean_valid' or 'sum_valid'")

    @torch.no_grad()
    def _attn_1d_to_heat(self, attn_1d, H, W):
        """
        [M] -> [g,g] -> upsample -> [H,W]
        """
        M = attn_1d.numel()
        gH, gW = self._infer_grid_hw(M)
        heat_g = attn_1d.view(gH, gW)

        heat = F.interpolate(
            heat_g[None, None],
            size=(H, W),
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in ("bilinear", "bicubic") else None,
        )[0, 0]
        return heat

    # -------------------- plotting --------------------
    def _plot_pair(self, sdf_meter, attn_heat, extent=None, fname=None, title_right="SDF + Attention"):
        """
        左：米制SDF
        右：SDF底图 + attention叠加
        """
        sdf_np = self._to_numpy(sdf_meter)
        heat_np = self._to_numpy(attn_heat)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：SDF（米制）
        im0 = axes[0].imshow(sdf_np, cmap=self.sdf_cmap, origin="lower", extent=extent)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04,
                     label="Distance to nearest obstacle [m]")
        axes[0].set_title("Signed Distance Field (SDF)")
        axes[0].set_xlabel("X [m]" if extent is not None else "X [px]")
        axes[0].set_ylabel("Y [m]" if extent is not None else "Y [px]")

        # 右图：SDF + attention
        axes[1].imshow(sdf_np, cmap=self.sdf_cmap, origin="lower", extent=extent)
        im1 = axes[1].imshow(heat_np, cmap=self.attn_cmap, origin="lower",
                             extent=extent, alpha=self.attn_alpha)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Attention weight")
        axes[1].set_title(title_right)
        axes[1].set_xlabel("X [m]" if extent is not None else "X [px]")
        axes[1].set_ylabel("Y [m]" if extent is not None else "Y [px]")

        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", pad_inches=0.1, dpi=self.dpi)
            plt.close()
        else:
            plt.show()

    # -------------------- public APIs --------------------
    @torch.no_grad()
    def show_single_neighbor(
        self,
        sdf_input,            # ✅ [B,1,H,W] 或 [B,H,W]，归一化
        attn_weights,         # [B, head, K, M]
        b=0,
        k=0,
        head=None,            # None=对head平均
        extent=None,
        fname=None,
    ):
        sdf_meter = self._get_sdf_meter(sdf_input, b=b)  # [H,W]
        H, W = sdf_meter.shape

        attn_1d = self._attn_1d_single(attn_weights, b=b, k=k, head=head)
        heat = self._attn_1d_to_heat(attn_1d, H, W)

        head_str = "mean(head)" if head is None else f"head={head}"
        self._plot_pair(
            sdf_meter=sdf_meter,
            attn_heat=heat,
            extent=extent,
            fname=fname,
            title_right=f"SDF + Attention | single neighbor k={k} | {head_str}",
        )

    @torch.no_grad()
    def show_aggregated(
        self,
        sdf_input,            # ✅ [B,1,H,W] 或 [B,H,W]，归一化
        attn_weights,         # [B, head, K, M]
        neighbor_mask=None,   # [B,K]
        b=0,
        agg="mean_valid",
        extent=None,
        fname=None,
    ):
        sdf_meter = self._get_sdf_meter(sdf_input, b=b)  # [H,W]
        H, W = sdf_meter.shape

        attn_1d = self._attn_1d_agg(attn_weights, neighbor_mask=neighbor_mask, b=b, agg=agg)
        heat = self._attn_1d_to_heat(attn_1d, H, W)

        self._plot_pair(
            sdf_meter=sdf_meter,
            attn_heat=heat,
            extent=extent,
            fname=fname,
            title_right=f"Aggregated Attention | agg={agg}",
        )
    
    @staticmethod
    def _head_grid_shape(n_head: int):
        """
        head=4 -> (2,2)
        head=8 -> (2,4)
        其他 -> 尽量接近方形
        """
        if n_head == 4:
            return 2, 2
        if n_head == 8:
            return 2, 4
        r = int(math.floor(math.sqrt(n_head)))
        c = int(math.ceil(n_head / max(r, 1)))
        return r, c

    @staticmethod
    def _valid_ks_from_mask(neighbor_mask, b=0):
        """
        neighbor_mask: [B,K] bool/0-1
        return: valid ks list
        """
        if neighbor_mask is None:
            return None
        m = neighbor_mask[b].bool()
        return m.nonzero(as_tuple=False).squeeze(-1).tolist()

    # ---------- 1) 单个邻居：一张图显示所有 head ----------
    @torch.no_grad()
    def show_single_neighbor_heads_grid(
        self,
        sdf_input,
        attn_weights,      # [B, head, K, M]
        b=0,
        k=0,
        extent=None,
        fname=None,
        show_colorbar=True,
    ):
        """
        一张图：左 SDF；右侧用网格显示 head0~headN 的 attention overlay
        """
        sdf_meter = self._get_sdf_meter(sdf_input, b=b)  # [H,W]
        H, W = sdf_meter.shape
        sdf_np = self._to_numpy(sdf_meter)

        n_head = attn_weights.size(1)
        r, c = self._head_grid_shape(n_head)

        fig = plt.figure(figsize=(3.8 * (c + 1), 3.5 * r))
        gs = fig.add_gridspec(nrows=r, ncols=c + 1, width_ratios=[1.05] + [1] * c)

        # 左侧：SDF（占满 r 行）
        ax_sdf = fig.add_subplot(gs[:, 0])
        im0 = ax_sdf.imshow(sdf_np, cmap=self.sdf_cmap, origin="lower", extent=extent)
        ax_sdf.set_title("SDF (meter)")
        ax_sdf.set_xlabel("X [px]" if extent is None else "X [m]")
        ax_sdf.set_ylabel("Y [px]" if extent is None else "Y [m]")
        if show_colorbar:
            fig.colorbar(im0, ax=ax_sdf, fraction=0.046, pad=0.04, label="Distance [m]")

        # 右侧：每个 head 一个格子
        heat_ims = []
        for h in range(n_head):
            rr = h // c
            cc = h % c
            ax = fig.add_subplot(gs[rr, cc + 1])

            ax.imshow(sdf_np, cmap=self.sdf_cmap, origin="lower", extent=extent)

            attn_1d = self._attn_1d_single(attn_weights, b=b, k=k, head=h)  # [M]
            heat = self._attn_1d_to_heat(attn_1d, H, W)
            heat_np = self._to_numpy(heat)

            imh = ax.imshow(heat_np, cmap=self.attn_cmap, origin="lower",
                            extent=extent, alpha=self.attn_alpha)
            heat_ims.append(imh)

            ax.set_title(f"head {h}")
            ax.set_xlabel("X [px]" if extent is None else "X [m]")
            ax.set_ylabel("Y [px]" if extent is None else "Y [m]")

        fig.suptitle(f"SDF + Attn | b={b}, single neighbor k={k}", y=1.02)
        plt.tight_layout()

        # colorbar（attention）可选：用最后一个 head 的 im
        if show_colorbar and len(heat_ims) > 0:
            fig.colorbar(heat_ims[-1], ax=fig.axes, fraction=0.015, pad=0.01, label="Attention weight")

        if fname is not None:
            os.makedirs(os.path.dirname(fname), exist_ok=True) if "/" in fname else None
            plt.savefig(fname, bbox_inches="tight", pad_inches=0.1, dpi=self.dpi)
            plt.close()
        else:
            plt.show()

    # ---------- 2) 单个邻居：逐 head 单独保存（你之前那种） ----------
    @torch.no_grad()
    def show_single_neighbor_all_heads(
        self,
        sdf_input,
        attn_weights,
        b=0,
        k=0,
        extent=None,
        out_dir="logs",
        prefix="attn_single",
    ):
        """
        输出：
          logs/attn_single_b0_k2_head0.png
          logs/attn_single_b0_k2_head1.png ...
        """
        os.makedirs(out_dir, exist_ok=True)
        n_head = attn_weights.size(1)
        for h in range(n_head):
            fname = f"{out_dir}/{prefix}_b{b}_k{k}_head{h}.png"
            self.show_single_neighbor(
                sdf_input=sdf_input,
                attn_weights=attn_weights,
                b=b,
                k=k,
                head=h,
                extent=extent,
                fname=fname,
            )

    # ---------- 3) 所有有效邻居：每个邻居都画一个“head网格大图” ----------
    @torch.no_grad()
    def show_valid_neighbors_heads_grid(
        self,
        sdf_input,
        attn_weights,
        neighbor_mask,      # [B,K] 必须传，才能找 valid ks
        b=0,
        extent=None,
        out_dir="logs",
        prefix="attn_grid",
        show_colorbar=True,
    ):
        """
        输出：
          logs/attn_grid_b0_k0_heads.png
          logs/attn_grid_b0_k1_heads.png ...
        """
        os.makedirs(out_dir, exist_ok=True)
        valid_ks = self._valid_ks_from_mask(neighbor_mask, b=b)
        if valid_ks is None or len(valid_ks) == 0:
            print("[Vis] No valid neighbors found.")
            return

        for k in valid_ks:
            fname = f"{out_dir}/{prefix}_b{b}_k{k}_heads.png"
            self.show_single_neighbor_heads_grid(
                sdf_input=sdf_input,
                attn_weights=attn_weights,
                b=b,
                k=k,
                extent=extent,
                fname=fname,
                show_colorbar=show_colorbar,
            )

    # ---------- 4) 所有有效邻居：每个邻居都逐 head 单图保存（文件最多） ----------
    @torch.no_grad()
    def show_valid_neighbors_all_heads_files(
        self,
        sdf_input,
        attn_weights,
        neighbor_mask,
        b=0,
        extent=None,
        out_dir="logs",
        prefix="attn_single",
    ):
        """
        输出很多文件：
          logs/attn_single_b0_k0_head0.png
          logs/attn_single_b0_k0_head1.png
          ...
          logs/attn_single_b0_k1_head0.png ...
        """
        os.makedirs(out_dir, exist_ok=True)
        valid_ks = self._valid_ks_from_mask(neighbor_mask, b=b)
        if valid_ks is None or len(valid_ks) == 0:
            print("[Vis] No valid neighbors found.")
            return

        for k in valid_ks:
            self.show_single_neighbor_all_heads(
                sdf_input=sdf_input,
                attn_weights=attn_weights,
                b=b,
                k=k,
                extent=extent,
                out_dir=out_dir,
                prefix=prefix,
            )

def _safe_prob(p: torch.Tensor, eps: float = 1e-8):
    # p: [..., M] already softmaxed
    return (p + eps) / (p + eps).sum(dim=-1, keepdim=True)

def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    # p,q: [..., M] probability
    p = _safe_prob(p, eps)
    q = _safe_prob(q, eps)
    return (p * (p.log() - q.log())).sum(dim=-1)

def _js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    # Jensen-Shannon divergence
    p = _safe_prob(p, eps)
    q = _safe_prob(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps)

@torch.no_grad()
def debug_query_attn_ctx_similarity(
    neigh_tokens: torch.Tensor,   # [B, K, D]
    attn: torch.Tensor,           # [B, H, K, M]
    ctx_tokens: torch.Tensor,     # [B, K, D]
    neighbor_mask: torch.Tensor,  # [B, K] bool
    b: int = 0,
    pair_mode: str = "all_pairs", # "all_pairs" 或 "vs_first"
    print_top: int = 50,
):
    """
    打印：
      - query cos
      - ctx  cos
      - 每个 head 的 attn cos / mean|diff| / KL / JS
    """
    B, K, D = neigh_tokens.shape
    _, H, _, M = attn.shape
    assert b < B

    valid_ks = torch.nonzero(neighbor_mask[b]).flatten().tolist()
    if len(valid_ks) <= 1:
        print(f"[debug] batch {b}: valid neighbors <= 1, skip")
        return

    def one_pair(k0, k1):
        q0 = neigh_tokens[b, k0]
        q1 = neigh_tokens[b, k1]
        c0 = ctx_tokens[b, k0]
        c1 = ctx_tokens[b, k1]

        q_cos = F.cosine_similarity(q0, q1, dim=0).item()
        c_cos = F.cosine_similarity(c0, c1, dim=0).item()
        c_l2  = (c0 - c1).pow(2).mean().sqrt().item()

        # 每个 head 的 attn 对比
        head_stats = []
        for h in range(H):
            a0 = attn[b, h, k0]  # [M]
            a1 = attn[b, h, k1]  # [M]
            a_cos = F.cosine_similarity(a0, a1, dim=0).item()
            a_l1  = (a0 - a1).abs().mean().item()
            a_kl  = _kl(a0, a1).item()
            a_js  = _js(a0, a1).item()
            head_stats.append((a_cos, a_l1, a_kl, a_js))

        # 汇总 head 平均（更容易看）
        mean_cos = sum(x[0] for x in head_stats) / H
        mean_l1  = sum(x[1] for x in head_stats) / H
        mean_kl  = sum(x[2] for x in head_stats) / H
        mean_js  = sum(x[3] for x in head_stats) / H

        print(f"\n[k={k0} vs k={k1}] query_cos={q_cos:.4f} | ctx_cos={c_cos:.4f} ctx_L2={c_l2:.3e}")
        print(f"  attn(head-mean): cos={mean_cos:.6f} | L1={mean_l1:.3e} | KL={mean_kl:.3e} | JS={mean_js:.3e}")
        for h, (a_cos, a_l1, a_kl, a_js) in enumerate(head_stats):
            print(f"    head {h}: attn_cos={a_cos:.6f} | L1={a_l1:.3e} | KL={a_kl:.3e} | JS={a_js:.3e}")

    # 选择对比方式
    if pair_mode == "vs_first":
        k0 = valid_ks[0]
        for k1 in valid_ks[1:]:
            one_pair(k0, k1)
    else:
        # all_pairs：所有有效 neighbor 两两对比（可能输出很多）
        cnt = 0
        for i in range(len(valid_ks)):
            for j in range(i + 1, len(valid_ks)):
                one_pair(valid_ks[i], valid_ks[j])
                cnt += 1
                if cnt >= print_top:
                    print(f"\n[debug] stop at {print_top} pairs")
                    return

@torch.no_grad()
def debug_attn_variance_over_k(attn: torch.Tensor, neighbor_mask: torch.Tensor, b: int = 0):
    """
    统计：同一个 head 内，valid neighbor 的 attention map 方差大小
    attn: [B,H,K,M]
    """
    B, H, K, M = attn.shape
    valid = neighbor_mask[b]  # [K]
    idx = torch.nonzero(valid).flatten()
    if idx.numel() <= 1:
        print("[attn_var] valid_k <= 1, skip")
        return

    a = attn[b, :, idx, :]  # [H, Kv, M]
    # 对 K 维度求方差，再对 M 平均
    var = a.var(dim=1).mean(dim=-1)  # [H]
    print("\n[attn_var over k] per-head mean(var over tokens):")
    for h in range(H):
        print(f"  head {h}: {var[h].item():.6e}")

@torch.no_grad()
def analyze_cross_attn_vs_sdf(
    sdf_input,               # [B,1,H,W] or [B,H,W], normalized [0,1]
    attn_weights,            # [B,head,K,M]
    neighbor_mask=None,      # [B,K] optional (0/1 or bool)
    neighbor_weights=None,   # [B,K] optional (neigh_w 或 w_obst). None则均匀聚合有效邻居
    b=0,
    ks=None,                 # None -> 自动用 neighbor_mask 找有效邻居
    heads=None,              # None -> 全部 head
    sdf_scale=4.0,           # meter = sdf_norm * sdf_scale
    topk=10,
    eps=1e-3,
    upsample_mode="bilinear",
    print_per_k=True,
    print_agg=True,
):
    """
    输出指标（每个k、每个head）：
      1) max/mean, topk_mass, entropy_norm        (聚焦程度)
      2) topk SDF(m) min/mean/max                 (关注区域离障碍远近)
      3) topk_mean - global_mean                  (相对全图偏空旷/偏障碍)
      4) corr(attn, 1/sdf)                        (越大越偏向障碍)

    另外输出 Aggregated（按 neighbor_weights 聚合所有有效 k）的一套同样指标。

    返回：
      results_per_k[(k,head)]
      results_agg[("agg",head)] 以及 ("agg","mean_head")
    """

    # ---- sdf meter ----
    if sdf_input.dim() == 4:
        sdf_meter = sdf_input[b, 0] * sdf_scale  # [H,W]
    elif sdf_input.dim() == 3:
        sdf_meter = sdf_input[b] * sdf_scale     # [H,W]
    else:
        raise ValueError("sdf_input must be [B,1,H,W] or [B,H,W]")
    H, W = sdf_meter.shape

    # ---- basic shapes ----
    B, n_head, K, M = attn_weights.shape
    g = int(math.sqrt(M))
    assert g * g == M, f"M={M} not a square, cannot reshape to grid"
    logM = math.log(M)
    global_mean_sdf = float(sdf_meter.mean().item())

    # ---- valid ks ----
    if neighbor_mask is None:
        valid_ks = list(range(K))
        mask_k = torch.ones(K, device=attn_weights.device, dtype=torch.float32)
    else:
        m = neighbor_mask[b].bool()
        valid_ks = m.nonzero(as_tuple=False).squeeze(-1).tolist()
        mask_k = m.float()  # [K]

    if ks is None:
        ks = valid_ks
    else:
        ks = list(ks)

    if heads is None:
        heads = list(range(n_head))
    else:
        heads = list(heads)

    def attn1d_to_map(attn_1d):
        a2d = attn_1d.view(g, g)
        a_up = F.interpolate(
            a2d[None, None],
            size=(H, W),
            mode=upsample_mode,
            align_corners=False if upsample_mode in ("bilinear", "bicubic") else None,
        )[0, 0]  # [H,W]
        return a_up

    def compute_metrics(attn_1d):
        # attn_1d: [M] already sums to 1
        mean = float(attn_1d.mean().item())
        mx = float(attn_1d.max().item())
        ratio = mx / (mean + 1e-12)

        v_top, _ = torch.topk(attn_1d, min(topk, M))
        topk_mass = float(v_top.sum().item())

        ent = float((-(attn_1d * attn_1d.clamp_min(1e-12).log())).sum().item())
        ent_norm = ent / (logM + 1e-12)

        # map-based topk sdf
        a_map = attn1d_to_map(attn_1d)
        vals, idx = torch.topk(a_map.flatten(), min(topk, H * W))
        ys = idx // W
        xs = idx % W
        sdf_top = sdf_meter[ys, xs]

        sdf_min = float(sdf_top.min().item())
        sdf_mean = float(sdf_top.mean().item())
        sdf_max = float(sdf_top.max().item())

        topk_minus_global = sdf_mean - global_mean_sdf

        inv = (1.0 / (sdf_meter + eps)).flatten()
        a_flat = a_map.flatten()
        a_flat = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-12)
        inv = (inv - inv.mean()) / (inv.std() + 1e-12)
        corr = float((a_flat * inv).mean().item())

        return dict(
            M=M,
            uniform=1.0 / M,
            max_over_mean=ratio,
            topk_mass=topk_mass,
            topk_mass_uniform=(min(topk, M) / M),
            entropy=ent,
            entropy_max=logM,
            entropy_norm=ent_norm,
            topk_sdf_min=sdf_min,
            topk_sdf_mean=sdf_mean,
            topk_sdf_max=sdf_max,
            global_sdf_mean=global_mean_sdf,
            topk_minus_global=topk_minus_global,
            corr_attn_inv_sdf=corr,
        )

    # ===================== (A) per-k metrics =====================
    results_per_k = {}
    if print_per_k:
        print(f"valid ks = {valid_ks}\n")

    for k in ks:
        for head in heads:
            a = attn_weights[b, head, k].clone()
            a = a / a.sum().clamp_min(1e-12)

            met = compute_metrics(a)
            results_per_k[(k, head)] = met

            if print_per_k:
                print(f"[b={b}, k={k}, head={head}]")
                print(f"  focus: max/mean={met['max_over_mean']:.3f}, "
                      f"top{topk}_mass={met['topk_mass']:.4f} (uniform≈{met['topk_mass_uniform']:.4f}), "
                      f"entropy_norm={met['entropy_norm']:.3f}")
                print(f"  top{topk} SDF(m): min={met['topk_sdf_min']:.3f}, mean={met['topk_sdf_mean']:.3f}, "
                      f"max={met['topk_sdf_max']:.3f} | global_mean={met['global_sdf_mean']:.3f} | "
                      f"topk-global={met['topk_minus_global']:+.3f}")
                print(f"  corr(attn,1/sdf)={met['corr_attn_inv_sdf']:.4f}\n")

    # ===================== (B) aggregated metrics =====================
    results_agg = {}
    if print_agg:
        # weights over K
        if neighbor_weights is None:
            w = mask_k.clone()  # 有效邻居均匀
        else:
            w = neighbor_weights[b].float().to(attn_weights.device) * mask_k

        w = w / w.sum().clamp_min(1e-12)  # 归一化到 sum=1

        print("---- Aggregated over valid neighbors ----")
        print(f"agg weights (valid only, normalized): {w.detach().cpu().numpy()}\n")

        # per head agg
        agg_attn_heads = []
        for head in heads:
            # Σ_k w_k * attn[b,head,k,:]
            a = (attn_weights[b, head] * w[:, None]).sum(dim=0)  # [M]
            a = a / a.sum().clamp_min(1e-12)
            met = compute_metrics(a)
            results_agg[("agg", head)] = met
            agg_attn_heads.append(a)

            print(f"[AGG, head={head}]")
            print(f"  focus: max/mean={met['max_over_mean']:.3f}, "
                  f"top{topk}_mass={met['topk_mass']:.4f} (uniform≈{met['topk_mass_uniform']:.4f}), "
                  f"entropy_norm={met['entropy_norm']:.3f}")
            print(f"  top{topk} SDF(m): min={met['topk_sdf_min']:.3f}, mean={met['topk_sdf_mean']:.3f}, "
                  f"max={met['topk_sdf_max']:.3f} | global_mean={met['global_sdf_mean']:.3f} | "
                  f"topk-global={met['topk_minus_global']:+.3f}")
            print(f"  corr(attn,1/sdf)={met['corr_attn_inv_sdf']:.4f}\n")

        # mean-head agg
        if len(agg_attn_heads) > 0:
            a_mean = torch.stack(agg_attn_heads, dim=0).mean(dim=0)  # [M]
            a_mean = a_mean / a_mean.sum().clamp_min(1e-12)
            met = compute_metrics(a_mean)
            results_agg[("agg", "mean_head")] = met

            print(f"[AGG, mean_head]")
            print(f"  focus: max/mean={met['max_over_mean']:.3f}, "
                  f"top{topk}_mass={met['topk_mass']:.4f} (uniform≈{met['topk_mass_uniform']:.4f}), "
                  f"entropy_norm={met['entropy_norm']:.3f}")
            print(f"  top{topk} SDF(m): min={met['topk_sdf_min']:.3f}, mean={met['topk_sdf_mean']:.3f}, "
                  f"max={met['topk_sdf_max']:.3f} | global_mean={met['global_sdf_mean']:.3f} | "
                  f"topk-global={met['topk_minus_global']:+.3f}")
            print(f"  corr(attn,1/sdf)={met['corr_attn_inv_sdf']:.4f}\n")

    return results_per_k, results_agg

import torch
import torch.nn.functional as F
import math
from itertools import combinations

@torch.no_grad()
def analyze_query_attn_ctx_invariance(
    cross_attn,              # 你的 CrossAttention module
    neigh_tokens,            # [B,K,D]
    obs_tokens_proj,         # [B,M,D]
    attn_weights,            # [B,H,K,M]
    ctx_tokens,              # [B,K,D]  (return_ctx=True 时的 ctx)
    neighbor_mask=None,      # [B,K] bool
    b=0,
    max_pairs=30,
):
    """
    输出：
      - raw query 相似度
      - projected query (per-head) 相似度
      - attention (per-head) 相似度 + KL/JS/L1
      - ctx 相似度
    """
    device = neigh_tokens.device
    B, K, D = neigh_tokens.shape
    _, H, _, M = attn_weights.shape
    dk = cross_attn.d_k

    if neighbor_mask is None:
        valid_idx = list(range(K))
    else:
        valid_idx = neighbor_mask[b].bool().nonzero(as_tuple=False).squeeze(-1).tolist()

    if len(valid_idx) < 2:
        print("[analyze] valid neighbors < 2, skip.")
        return

    # -------- projected query per head --------
    # q: [B,H,K,dk]
    q = cross_attn.w_qs(neigh_tokens).view(B, K, H, dk).permute(0, 2, 1, 3).contiguous()

    def _kl(p, q, eps=1e-12):
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        return (p * (p.log() - q.log())).sum().item()

    def _js(p, q, eps=1e-12):
        m = 0.5 * (p + q)
        return 0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps)

    pairs = list(combinations(valid_idx, 2))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    print(f"\n[analyze] b={b} valid_neighbors={valid_idx} total_pairs={len(pairs)}")

    for (k0, k1) in pairs:
        # ---- raw query ----
        q0_raw = neigh_tokens[b, k0]
        q1_raw = neigh_tokens[b, k1]
        raw_cos = F.cosine_similarity(q0_raw, q1_raw, dim=0).item()

        # ---- ctx ----
        c0 = ctx_tokens[b, k0]
        c1 = ctx_tokens[b, k1]
        ctx_cos = F.cosine_similarity(c0, c1, dim=0).item()
        ctx_l2  = (c0 - c1).norm(p=2).item()

        # ---- attention head-mean ----
        a0_mean = attn_weights[b, :, k0, :].mean(dim=0)  # [M]
        a1_mean = attn_weights[b, :, k1, :].mean(dim=0)  # [M]
        attn_mean_cos = F.cosine_similarity(a0_mean, a1_mean, dim=0).item()
        attn_mean_l1  = (a0_mean - a1_mean).abs().mean().item()
        attn_mean_kl  = _kl(a0_mean, a1_mean)
        attn_mean_js  = _js(a0_mean, a1_mean)

        print(f"\n[k={k0} vs k={k1}] raw_query_cos={raw_cos:.4f} | ctx_cos={ctx_cos:.4f} ctx_L2={ctx_l2:.3e}")
        print(f"  attn(head-mean): cos={attn_mean_cos:.6f} | L1={attn_mean_l1:.3e} | KL={attn_mean_kl:.3e} | JS={attn_mean_js:.3e}")

        # ---- per head detail ----
        for h in range(H):
            # projected query cos
            pq0 = q[b, h, k0]   # [dk]
            pq1 = q[b, h, k1]
            proj_q_cos = F.cosine_similarity(pq0, pq1, dim=0).item()

            # attn cos
            a0 = attn_weights[b, h, k0, :]  # [M]
            a1 = attn_weights[b, h, k1, :]
            attn_cos = F.cosine_similarity(a0, a1, dim=0).item()
            attn_l1  = (a0 - a1).abs().mean().item()
            attn_kl  = _kl(a0, a1)
            attn_js  = _js(a0, a1)

            print(f"    head {h}: proj_q_cos={proj_q_cos:.6f} | attn_cos={attn_cos:.6f} | L1={attn_l1:.3e} | KL={attn_kl:.3e} | JS={attn_js:.3e}")

import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def debug_attn_entropy_and_logits(self_cross_attn, neigh_tokens, obs_tokens_proj, attn, neighbor_mask=None, b=0):
    """
    attn: [B, H, K, M]
    """
    B, H, K, M = attn.shape
    dk = self_cross_attn.d_k
    temp = dk ** 0.5

    # 1) attention entropy（越接近 log(M) 越均匀）
    p = attn.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)  # [B,H,K]
    ent_mean = ent[b].mean().item()
    ent_max_uniform = math.log(M)
    print(f"[attn entropy] mean={ent_mean:.4f} | uniform_logM={ent_max_uniform:.4f} | ratio={ent_mean/ent_max_uniform:.4f}")

    # 2) logits std：如果 std 很小，softmax 必然接近均匀
    q = self_cross_attn.w_qs(neigh_tokens).view(B, K, H, dk).permute(0,2,1,3)  # [B,H,K,dk]
    k = self_cross_attn.w_ks(obs_tokens_proj).view(B, M, H, dk).permute(0,2,1,3) # [B,H,M,dk]

    logits = torch.matmul(q, k.transpose(-2,-1)) / temp  # [B,H,K,M]
    logits_std = logits[b].std(dim=-1).mean().item()     # mean over (H,K)
    logits_absmax = logits[b].abs().max().item()
    print(f"[logits] std(mean over H,K)={logits_std:.6f} | absmax={logits_absmax:.6f}")

    # 3) 每个 head 的 top1 是否一致（如果总是同一个位置，也说明 query 没起作用）
    top1 = attn[b].argmax(dim=-1)  # [H,K]
    print(f"[top1 unique] unique_count={len(torch.unique(top1))} / (H*K={H*K})")

# utils/attn_debug.py
import math
import torch
import torch.nn.functional as F

def _cos(a, b, dim=-1):
    return F.cosine_similarity(a, b, dim=dim)

def _kl(p, q, eps=1e-12):
    # p,q: [..., M], 已经是概率分布
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)

def _js(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps=eps) + 0.5 * _kl(q, m, eps=eps)

def _l1_mean(p, q):
    return (p - q).abs().mean(dim=-1)

def _entropy(p, eps=1e-12):
    p = p.clamp_min(eps)
    return (-(p * p.log())).sum(dim=-1)

def _topk_overlap(p, q, k=30):
    # 返回 overlap_ratio = |A∩B|/k
    k = min(k, p.numel())
    idx_p = torch.topk(p, k).indices
    idx_q = torch.topk(q, k).indices
    inter = torch.isin(idx_p, idx_q).float().sum()
    return (inter / k).item()

@torch.no_grad()
def analyze_attn_similarity(
    attn_weights,          # [B, H, K, M]
    neigh_tokens,          # [B, K, d_model]  (raw query tokens)
    ctx_tokens=None,       # [B, K, d_model]  (可选)
    neighbor_mask=None,    # [B, K] bool
    cross_attn_module=None,# 你的 CrossAttention 实例，用来算 proj_q
    b=0,
    topk=30,
    print_pairs=True,
):
    """
    打印所有有效邻居 query 两两之间的相似度：
      raw_query_cos / proj_q_cos / attn_cos / L1 / KL / JS / topk_overlap
    """
    device = attn_weights.device
    B, H, K, M = attn_weights.shape

    if neighbor_mask is None:
        valid = torch.arange(K, device=device)
    else:
        valid = neighbor_mask[b].nonzero(as_tuple=False).squeeze(-1)

    valid = valid.tolist()
    if len(valid) <= 1:
        print(f"[analyze] b={b} valid_neighbors={valid} (skip: need >=2)")
        return

    print(f"[analyze] b={b} valid_neighbors={valid} total_pairs={len(valid)*(len(valid)-1)//2}\n")

    # 预计算 proj_q（每个 head 的投影 query）
    # proj_q: [H, K, d_k]
    proj_q = None
    if cross_attn_module is not None:
        # neigh_tokens[b]: [K, d_model]
        q = cross_attn_module.w_qs(neigh_tokens[b])  # [K, H*d_k]
        d_k = cross_attn_module.d_k
        proj_q = q.view(K, H, d_k).transpose(0, 1).contiguous()  # [H, K, d_k]

    # 统计注意力的“尖锐程度”，看看是不是都接近 uniform
    ent = _entropy(attn_weights[b].reshape(H*K, M)).mean().item()
    uniform_logM = math.log(M)
    print(f"[attn entropy] mean={ent:.4f} | uniform_logM={uniform_logM:.4f} | ratio={ent/uniform_logM:.4f}")

    # top1 unique：不同 head、不同 query 的 argmax 是否多样
    top1 = attn_weights[b].argmax(dim=-1).reshape(-1)  # [H*K]
    uniq = torch.unique(top1).numel()
    print(f"[top1 unique] unique_count={uniq} / (H*K={H*K})\n")

    # 两两比较
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            k0, k1 = valid[i], valid[j]

            q0 = neigh_tokens[b, k0]
            q1 = neigh_tokens[b, k1]
            raw_q_cos = _cos(q0, q1, dim=0).item()

            # ctx 相似
            if ctx_tokens is not None:
                c0 = ctx_tokens[b, k0]
                c1 = ctx_tokens[b, k1]
                ctx_cos = _cos(c0, c1, dim=0).item()
                ctx_l2  = (c0 - c1).norm(p=2).item()
            else:
                ctx_cos, ctx_l2 = None, None

            # head-mean attention
            a0_mean = attn_weights[b, :, k0, :].mean(dim=0)
            a1_mean = attn_weights[b, :, k1, :].mean(dim=0)

            # 保证归一（正常 softmax 已经是）
            a0_mean = a0_mean / a0_mean.sum().clamp_min(1e-12)
            a1_mean = a1_mean / a1_mean.sum().clamp_min(1e-12)

            attn_cos_mean = _cos(a0_mean, a1_mean, dim=0).item()
            l1_mean = _l1_mean(a0_mean, a1_mean).item()
            kl_mean = _kl(a0_mean, a1_mean).item()
            js_mean = _js(a0_mean, a1_mean).item()
            overlap_mean = _topk_overlap(a0_mean, a1_mean, k=topk)

            if print_pairs:
                if ctx_cos is None:
                    print(f"[k={k0} vs k={k1}] raw_query_cos={raw_q_cos:.4f}")
                else:
                    print(f"[k={k0} vs k={k1}] raw_query_cos={raw_q_cos:.4f} | ctx_cos={ctx_cos:.4f} ctx_L2={ctx_l2:.3e}")

                print(f"  attn(head-mean): cos={attn_cos_mean:.6f} | L1={l1_mean:.3e} | KL={kl_mean:.3e} | JS={js_mean:.3e} | top{topk}_overlap={overlap_mean:.3f}")

            # per-head
            for h in range(H):
                a0 = attn_weights[b, h, k0, :]
                a1 = attn_weights[b, h, k1, :]
                a0 = a0 / a0.sum().clamp_min(1e-12)
                a1 = a1 / a1.sum().clamp_min(1e-12)

                attn_cos = _cos(a0, a1, dim=0).item()
                l1 = _l1_mean(a0, a1).item()
                kl = _kl(a0, a1).item()
                js = _js(a0, a1).item()
                overlap = _topk_overlap(a0, a1, k=topk)

                if proj_q is not None:
                    pq_cos = _cos(proj_q[h, k0], proj_q[h, k1], dim=0).item()
                    print(f"    head {h}: proj_q_cos={pq_cos:.6f} | attn_cos={attn_cos:.6f} | L1={l1:.3e} | KL={kl:.3e} | JS={js:.3e} | top{topk}_overlap={overlap:.3f}")
                else:
                    print(f"    head {h}: attn_cos={attn_cos:.6f} | L1={l1:.3e} | KL={kl:.3e} | JS={js:.3e} | top{topk}_overlap={overlap:.3f}")

            print("")


