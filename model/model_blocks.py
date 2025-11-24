import torch
import torch.nn as nn
import torch.nn.functional as F

class C_encoder_block_next(nn.Module):
    """
    ResNeXt 风格的全连接残差块:
    - 输入: [N, in_channel]
    - 输出: [N, out_channel]，其中 out_channel 必须与 in_channel 一致才能残差相加
    """
    def __init__(
        self,
        in_channel: int,
        hidden_size: int,
        out_channel: int,
        cardinality: int = 32,
    ) -> None:
        super().__init__()
        self.act = nn.ELU()
        self.cardinality = cardinality

        # 每个分支的隐藏维度
        hidden_card_dim = hidden_size // cardinality

        # 多分支 MLP
        self.enc_branches = nn.ModuleList()
        for _ in range(cardinality):
            branch = nn.Sequential(
                nn.Linear(in_channel, hidden_card_dim),
                nn.ELU(),
                nn.Linear(hidden_card_dim, out_channel),
            )
            self.enc_branches.append(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C]
        branch_outputs = [branch(x) for branch in self.enc_branches]  # list of [N,C]
        y = torch.stack(branch_outputs, dim=1)                        # [N,card,C]
        y = torch.sum(y, dim=1)                                       # [N,C]

        # 残差
        y = y + x
        x = self.act(y)
        return x


class C_encoder_block(nn.Module):
    """
    简单的 MLP 残差块:
    - Linear -> ELU -> Linear + 残差，再过一次 ELU
    """
    def __init__(self, in_channel: int, hidden_size: int, out_channel: int) -> None:
        super().__init__()
        self.act = nn.ELU()
        self.enc = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            self.act,
            nn.Linear(hidden_size, out_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.enc(x)
        y = y + x
        x = self.act(y)
        return x


class UnetBaiscBlock(nn.Module):
    """
    简单的两层全连接“U-Net” block:
    - Linear -> ELU -> Linear -> ELU
    """
    def __init__(self, in_channel: int, hidden_size: int, out_channel: int) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, out_channel),
            nn.ELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    """ 计算 Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # 计算注意力得分
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature  # [B, n_head, len_q, len_k]
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # 应用 mask，避免计算填充部分的注意力

        attn_weights = torch.softmax(attn, dim=-1)  # 归一化得分，得到注意力权重
        output = torch.matmul(attn_weights, v)  # 加权求和值
        return output, attn_weights

class CrossAttention(nn.Module):
    """ Cross-Attention 模块 """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(CrossAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 线性变换层，分别生成不同头的 Query、Key 和 Value
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # 注意力模块
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask=None):
        """
        query: 来自邻居信息的输入，形状为 [B, K, d_model]
        key: 来自SDF特征的输入，形状为 [B, M, d_model]
        value: 来自SDF特征的输入，形状为 [B, M, d_model]
        mask: 可选的掩码，形状为 [B, K]，用于指定哪些邻居是有效的
        """
        B, K, _ = query.size()  # B 为批量大小，K 为邻居数量

        # 将 Query, Key 和 Value 投影到多个头上
        q = self.w_qs(query).view(B, K, self.n_head, self.d_k).transpose(1, 2)  # [B, n_head, K, d_k]
        k = self.w_ks(key).view(B, -1, self.n_head, self.d_k).transpose(1, 2)    # [B, n_head, M, d_k]
        v = self.w_vs(value).view(B, -1, self.n_head, self.d_v).transpose(1, 2)  # [B, n_head, M, d_v]

        # 使用 Scaled Dot-Product Attention 计算加权求和
        output, attn_weights = self.attention(q, k, v, mask=mask)  # [B, n_head, K, d_v]

        # 合并所有头的输出
        output = output.transpose(1, 2).contiguous().view(B, K, -1)  # [B, K, n_head * d_v]
        output = self.dropout(self.fc(output))  # 通过线性层处理合并后的输出
        output = output + query  # 残差连接
        output = self.layer_norm(output)  # 层归一化

        return output, attn_weights

