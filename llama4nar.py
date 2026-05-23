from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch import LongTensor, Tensor
from torch.nn import functional as F
from typing import Union
from typing import Optional
 

class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 1024, base: int = 10000):
        r"""Rotary position embedding.

        [1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary 
        position embedding." Neurocomputing, 2024

        h: head_dim
        l: seq_len
        """
        super().__init__()

        self.head_dim = head_dim

        # Calculate θ = 1 / 10000**(2i/h)
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))  # (h/2,)

        # Matrix pθ
        pos_theta = torch.outer(torch.arange(max_len), theta).float()  # (l, h/2)

        # Rotation matrix
        w = torch.stack([torch.cos(pos_theta), torch.sin(pos_theta)], dim=-1)  # (l, h/2, 2)
        self.register_buffer(name="w", tensor=w)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply RoPE.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h)

        Outputs:
            out: (b, l, n, h)
        """
        L = x.shape[1]
        x = rearrange(x, 'b l n (h c) -> b l n h c', c=2)  # (b, l, n, h/2, 2)
        w = self.w[0 : L][None, :, None, :, :]  # (1, l, 1, h/2, 2)
        x = self.rotate(x, w)  # (b, l, n, h/2, 2)
        x = rearrange(x, 'b l n h c -> b l n (h c)')  # (b, l, n, h)
        
        return x

    def rotate(self, x: Tensor, w: Tensor) -> Tensor:
        r"""Rotate x.

        x0 = cos(θp)·x0 - sin(θp)·x1
        x1 = sin(θp)·x0 + cos(θp)·x1

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h/2, 2)
            w: (1, l, 1, h/2, 2)

        Outputs:
            out: (b, l, n, h/2, 2)
        """

        out = torch.stack([
            w[..., 0] * x[..., 0] - w[..., 1] * x[..., 1],
            w[..., 0] * x[..., 1] + w[..., 1] * x[..., 0]
            ],
            dim=-1,
        )  # (b, l, n, h/2, 2)

        return out

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply Nd RoPE with sparse positions.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim
        k: data dim

        Args:
            x: (b, l, n, h)
            pos: (l, k)
            n_dim: int

        Outputs:
            out: (b, l, n, h)
        """
        
        B, L, N, H = x.shape
        K = pos.shape[1]  # rope_dim
        assert H == K * self.head_dim

        x = rearrange(x, 'b l n (k h c) -> k b l n h c', k=K, c=2)  # (k, b, l, n, h/2, 2)
        x = x.contiguous()

        for i in range(K):
            p = pos[:, i]  # (l,)
            w = self.w[p][None, :, None, :, :]  # (1, l, 1, h/2, 2)
            x[i] = self.rotate(x[i], w).clone()  # x: (k, b, l, n, h/2, 2)

        out = rearrange(x, 'k b l n h c -> b l n (k h c)')  # (b, l, n, h)
        
        return out

 
 

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn  = SelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp   = MLP(n_embd, n_head)

    # def forward(self, x: Tensor, rope: RoPE | None, attn_mask: Tensor | None) -> Tensor:
    def forward(self, x: Tensor, rope: Union[RoPE, None], attn_mask: Union[Tensor, None]) -> Tensor:

        x = x + self.attn(self.norm1(x), rope, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output

 
class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    # def forward(
    #     self,
    #     x: Tensor,               # (B, L, D)
    #     rope: RoPE | None, 
    #     attn_mask: Tensor | None # 形状见下文
    # ) -> Tensor:
    def forward(
        self, 
        x: Tensor, 
        rope: Optional[RoPE], 
        attn_mask: Optional[Tensor]
    ) -> Tensor:

        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # (B, L, D)
        q = rearrange(q, 'b l (n h) -> b n l h', n=self.n_head, h=self.head_dim)
        k = rearrange(k, 'b l (n h) -> b n l h', n=self.n_head, h=self.head_dim)
        v = rearrange(v, 'b l (n h) -> b n l h', n=self.n_head, h=self.head_dim)

        if rope is not None:
            # 你的 RoPE 实现是 (B, L, N, H) 形状，先换回去应用再换回
            q = rearrange(rope(rearrange(q, 'b n l h -> b l n h')), 'b l n h -> b n l h')
            k = rearrange(rope(rearrange(k, 'b n l h -> b l n h')), 'b l n h -> b n l h')

        # 关键点：这里没有“上三角因果 mask”，允许全双向注意力
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,     # 可为 None；用于遮住 padding（见下）
            dropout_p=0.0
        )  # (B, N, L, H)
        y = rearrange(y, 'b n l h -> b l (n h)')
        return self.proj(y)


class MLP(nn.Module):
    r"""Ref: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        n_hidden = int(8 * n_embd / 3)
        self.fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        r"""Causal self attention.

        Args:
            x: (b, l, d)
           
        Outputs:
            x: (b, l, d)
        """

        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.proj(x)
        return x




@dataclass
class DecoderConfig:
    seq_len: int = 50       # 序列长度
    feat_dim: int = 100      # 输入 / 输出特征维度
    n_layer: int = 4
    n_head: int = 4
    hidden_size: int = 256

    # vocab_size: int = 4096  # 输出词表大小


class PatchEncoder(nn.Module):
    """Patch Encoder: compress P frames into 1 token via CLS pooling.

    Input:  (B, T, P, D)   — T patches, each P frames of D-dim features
    Output: (B, T, hidden_size)  — one vector per patch

    Uses a small bidirectional Transformer (reusing Block/RoPE from this file)
    with a learnable CLS token prepended to each patch.
    """

    def __init__(self, input_dim: int, hidden_size: int, n_layer: int = 4, n_head: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_dim, hidden_size, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, hidden_size) * 0.02)

        head_dim = hidden_size // n_head
        self.rope = RoPE(head_dim)
        self.blocks = nn.ModuleList([Block(hidden_size, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, P, D)
        returns: (B, T, hidden_size)
        """
        B, T, P, D = x.shape
        x = self.input_proj(x)                                          # (B, T, P, H)
        cls = self.cls_token.expand(B, T, 1, -1)                       # (B, T, 1, H)
        x = torch.cat([cls, x], dim=2)                                 # (B, T, P+1, H)
        x = rearrange(x, 'b t p h -> (b t) p h')                      # (B*T, P+1, H)

        for blk in self.blocks:
            x = blk(x, self.rope, None)                                # bidirectional
        x = self.norm(x)

        cls_out = x[:, 0, :]                                           # (B*T, H)
        return rearrange(cls_out, '(b t) h -> b t h', b=B)             # (B, T, H)


class ContinuousDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size

        # 一个线性层把输入 feat_dim 映射到 hidden_size
        self.input_proj = nn.Linear(cfg.feat_dim, cfg.hidden_size)
        self.blocks = nn.ModuleList([
            Block(cfg.hidden_size, cfg.n_head) for _ in range(cfg.n_layer)
        ])
        self.norm = RMSNorm(cfg.hidden_size)
        # 输出一个连续特征，恢复到 feat_dim
        # self.output_proj = nn.Linear(cfg.hidden_size, cfg.feat_dim)
        # self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # RoPE
        # 这里 head_dim 是 hidden_size / n_head
        head_dim = cfg.hidden_size // cfg.n_head
        # self.rope = None 
        self.rope = RoPE(head_dim) #, max_len=cfg.seq_len)

    # def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    def forward(
        self, 
        x: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        x: (B, L, feat_dim) 连续特征输入
        attention_mask: (B, L) bool mask, True 表示有效 (非 padding)
        返回: preds: (B, L, feat_dim)
        """
        B, L, feat = x.shape
        # assert L == self.cfg.seq_len, f"输入长度 {L} 要等于 config.seq_len {self.cfg.seq_len}"

        # 投影到 hidden
        h = self.input_proj(x)  # (B, L, hidden_size)

        sdpa_mask = None
        if attention_mask is not None:
            # 构造适合 scaled_dot_product_attention 的 mask
            # mask 的 shape 是 (B, 1, 1, L)，padding位置为 True 或 False 的 convention 看实现
            # 这里我们把 padding=0, valid=1 的 mask 变成 additive mask
            pad = (~attention_mask).to(torch.float32) * -1e9  # (B, L)
            sdpa_mask = pad.view(B, 1, 1, L)  # broadcast

        # Transformer blocks
        for blk in self.blocks:
            h = blk(h, self.rope, sdpa_mask)

        h = self.norm(h)  # (B, L, hidden_size)
        # # preds = self.output_proj(h)  # (B, L, feat_dim)
        # logits = self.lm_head(h)   # (B, L, V) —— 并行逐位置预测
        return h

 
def model_size_info(model: nn.Module) -> dict:
    """
    返回模型大小信息，包括参数总数 (所有参数, 不区分是否 trainable)，
    以及 float32 存储所需字节大小估算（MB）。
    """

    total_params = sum(p.numel() for p in model.parameters())
    
    # 包含 buffers（non-parameter 状态）的大小
    total_size_bytes = 0
    for p in model.parameters():
        total_size_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_size_bytes += b.numel() * b.element_size()
    
    size_all_MB = total_size_bytes / (1024 ** 2)
    
    # 格式化参数量为人类易读形式
    def human_params(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n/1_000_000_000:.2f}B"
        elif n >= 1_000_000:
            return f"{n/1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n/1_000:.2f}K"
        else:
            return str(n)
    
    pretty = f"Params: {human_params(total_params)}; Memory for all params+buffers (float32): ~{size_all_MB:.2f} MB"
    
    return {
        "total_params": total_params,
        "size_all_MB": size_all_MB,
        "pretty": pretty
    }


if __name__ == "__main__":
    # 超参
    batch_size = 8
    seq_len = 50
    feat_dim = 100

    cfg = DecoderConfig(seq_len=seq_len, feat_dim=feat_dim, n_layer=4, n_head=8, hidden_size=1024)
    model = ContinuousDecoder(cfg)
    model_size_info(model)
    # 随机输入 /目标
    x = torch.randn(batch_size, seq_len, feat_dim)   # 输入连续特征
    target = torch.randn(batch_size, seq_len, feat_dim)  # 目标特征

    # mask （例如某些位置是 padding，不算 loss）
    # 这里我们假设不 padding，mask 全是 True
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    preds = model(x, attention_mask=attention_mask)  # (B, L, feat_dim)

    # loss = continuous_loss(preds, target, mask=attention_mask)

    print("preds.shape:", preds.shape)
    # print("loss:", loss.item())

    # 测试 backward
    loss.backward()
    print("backward OK")