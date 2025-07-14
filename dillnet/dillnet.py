import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
import torch.utils.checkpoint as checkpoint

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + x_rot * sin * self.scale

class Retention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = math.sqrt(self.head_dim)
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.gamma_param = nn.Parameter(torch.randn(heads))
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.talking_heads_scores_proj = nn.Linear(heads, heads)
        self.talking_heads_values_proj = nn.Linear(heads, heads)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.w_q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        Q = self.rotary(Q)
        K = self.rotary(K)
        gamma = torch.sigmoid(self.gamma_param)
        gamma = torch.clamp(gamma, min=0.9, max=0.999)
        idx = torch.arange(T, device=x.device)
        diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
        t_w = torch.exp(-gamma.view(self.heads, 1, 1) * diff)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores = scores.permute(0, 2, 3, 1)
        scores = self.talking_heads_scores_proj(scores)
        scores = scores.permute(0, 3, 1, 2)
        weighted_scores = scores * t_w.unsqueeze(0)
        out = torch.matmul(weighted_scores, V)
        out = out.permute(0, 2, 3, 1)
        out = self.talking_heads_values_proj(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.transpose(1, 2).reshape(B, T, -1)
        gate_out = F.silu(self.gate(x))
        out = self.dropout(gate_out * out)
        return out

@torch.jit.script
def binary_operator(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    A_i, Bu_i = q_i
    A_j, Bu_j = q_j
    return A_j * A_i, torch.addcmul(Bu_j, A_j, Bu_i)

@torch.jit.script
def associative_scan(elems: Tuple[torch.Tensor, torch.Tensor]):
    scanned_elems = (elems[0].clone(), elems[1].clone())
    num_elems = scanned_elems[0].shape[0]
    if num_elems <= 1:
        return scanned_elems
    stride = 1
    while stride < num_elems:
        A_a, Bu_a = scanned_elems[0][:-stride], scanned_elems[1][:-stride]
        A_b, Bu_b = scanned_elems[0][stride:],  scanned_elems[1][stride:]
        A_res, Bu_res = binary_operator((A_a, Bu_a), (A_b, Bu_b))
        new_A = torch.cat((scanned_elems[0][:stride], A_res), dim=0)
        new_Bu = torch.cat((scanned_elems[1][:stride], Bu_res), dim=0)
        scanned_elems = (new_A, new_Bu)
        stride *= 2
    stride = stride // 2
    while stride > 0:
        A_prev, Bu_prev = scanned_elems[0][stride-1:-stride], scanned_elems[1][stride-1:-stride]
        A_curr, Bu_curr = scanned_elems[0][2*stride-1:],       scanned_elems[1][2*stride-1:]
        A_res, Bu_res = binary_operator((A_prev, Bu_prev), (A_curr, Bu_curr))
        new_A = torch.cat((scanned_elems[0][:2*stride-1], A_res), dim=0)
        new_Bu = torch.cat((scanned_elems[1][:2*stride-1], Bu_res), dim=0)
        scanned_elems = (new_A, new_Bu)
        stride = stride // 2
    return scanned_elems

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, d_state=32, d_conv=4, dt_rank='auto'):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(embed_dim / 8) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=embed_dim // 4
        )
        self.x_proj = nn.Linear(embed_dim, self.dt_rank + self.d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, embed_dim)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _forward_logic(self, x):
        B, T, D = x.shape
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        delta, B_param, C_param = torch.split(
            x_ssm, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x_conv, delta, B_param, C_param)
        y = y + x_conv * self.D
        y = self.norm(y)
        y = y * F.silu(res)
        return self.out_proj(y)

    def selective_scan(self, u, delta, B, C):
        B_size, T_size, D_size = u.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = (delta.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1)
        deltaA = deltaA.permute(1, 0, 2, 3).contiguous()
        deltaB_u = deltaB_u.permute(1, 0, 2, 3).contiguous()
        flat_A = deltaA.view(T_size, -1, N)
        flat_B_u = deltaB_u.view(T_size, -1, N)
        _, ys = associative_scan((flat_A, flat_B_u))
        ys = ys.view(T_size, B_size, D_size, N)
        ys = ys.permute(1, 0, 2, 3).contiguous()
        y = (ys * C.unsqueeze(2)).sum(-1)
        return y
    
    def forward(self, x):
        forward_pass = checkpoint.checkpoint(self._forward_logic, x, use_reentrant=False)
        backward_pass = checkpoint.checkpoint(self._forward_logic, torch.flip(x, dims=[1]), use_reentrant=False)
        return forward_pass + torch.flip(backward_pass, dims=[1])

class DillNetTokenOperator(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1, use_mamba=False):
        super().__init__()
        self.use_mamba = use_mamba
        if self.use_mamba:
            self.op = MambaBlock(embed_dim=embed_dim)
        else:
            self.op = Retention(embed_dim=embed_dim, heads=heads, dropout=dropout)
        self.norm = RMSNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.op(x))

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DillNetMixerBlock(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, drop_path=0.0, token_operator=None):
        super().__init__()
        self.token_operator = token_operator
        self.channel_mixer = FeedForward(embed_dim, dropout=dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.drop_path = drop_path

    def forward(self, x):
        def stoch_depth(y):
            if self.training and torch.rand(()).item() < self.drop_path:
                return y
            return y
        x = x + stoch_depth(self.token_operator(self.norm1(x)))
        x = x + stoch_depth(self.channel_mixer(self.norm2(x)))
        return x

class DillNet(nn.Module):
    def __init__(self, embed_dim, depth, heads):
        super().__init__()
        self.blocks = nn.ModuleList()
        sched = ["ret","mamba","ret","ret","mamba","mamba"] * ((depth//6)+1)
        for i,kind in enumerate(sched[:depth]):
            use_mamba = (kind=="mamba")
            token_op = DillNetTokenOperator(embed_dim, heads, 0.1, use_mamba=use_mamba)
            mixer_block = DillNetMixerBlock(embed_dim=embed_dim, dropout=0.1, drop_path=0.1, token_operator=token_op)
            self.blocks.append(mixer_block)
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x