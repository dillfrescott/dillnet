import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
import torch.utils.checkpoint as checkpoint
from titans_pytorch import NeuralMemory

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
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        seq_len = x.shape[-2]
        device = x.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(1)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + x_rot * sin * self.scale

class TalkingHeadsAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()
        assert embed_dim % heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.pre_softmax_proj = nn.Linear(heads, heads, bias=False)
        self.post_softmax_proj = nn.Linear(heads, heads, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope):
        B, T, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        q = rope(q)
        k = rope(k)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * (self.head_dim ** -0.5)
        dots = self.pre_softmax_proj(dots.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = dots.softmax(dim=-1)
        attn = self.post_softmax_proj(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.dropout(self.to_out(out))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()
        self.attn = TalkingHeadsAttention(embed_dim, heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.ffn(self.norm2(x))
        return x

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
        A_b, Bu_b = scanned_elems[0][stride:], scanned_elems[1][stride:]
        A_res, Bu_res = binary_operator((A_a, Bu_a), (A_b, Bu_b))
        new_A = torch.cat((scanned_elems[0][:stride], A_res), dim=0)
        new_Bu = torch.cat((scanned_elems[1][:stride], Bu_res), dim=0)
        scanned_elems = (new_A, new_Bu)
        
        stride *= 2
    
    stride = stride // 2
    while stride > 0:
        A_prev, Bu_prev = scanned_elems[0][stride-1:-stride], scanned_elems[1][stride-1:-stride]
        A_curr, Bu_curr = scanned_elems[0][2*stride-1:], scanned_elems[1][2*stride-1:]
        A_res, Bu_res = binary_operator((A_prev, Bu_prev), (A_curr, Bu_curr))
        new_A = torch.cat((scanned_elems[0][:2*stride-1], A_res), dim=0)
        new_Bu = torch.cat((scanned_elems[1][:2*stride-1], Bu_res), dim=0)
        scanned_elems = (new_A, new_Bu)
        
        stride = stride // 2

    return scanned_elems

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, d_state=16, d_conv=4, dt_rank='auto'):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(embed_dim / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=embed_dim
        )
        self.x_proj = nn.Linear(embed_dim, self.dt_rank + self.d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, embed_dim)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.skip_proj = nn.Linear(embed_dim, embed_dim)

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
        res = F.silu(res)
        y = y * res
        skip = self.skip_proj(x)
        return self.out_proj(y) + skip

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
        return checkpoint.checkpoint(self._forward_logic, x, use_reentrant=False)

class MambaLayer(nn.Module):
    def __init__(self, embed_dim, d_state=16, d_conv=4, dt_rank='auto'):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.mamba = MambaBlock(embed_dim, d_state, d_conv, dt_rank)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

class DillNet(nn.Module):
    def __init__(self, embed_dim, depth, heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        head_dim = embed_dim // heads
        
        self.rope = RotaryEmbedding(dim=head_dim)
        self.mem = NeuralMemory(
            dim = embed_dim,
            chunk_size = 64
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_mamba = (i % 2 != 0)
            if use_mamba:
                self.blocks.append(MambaLayer(embed_dim=embed_dim))
            else:
                self.blocks.append(TransformerBlock(embed_dim=embed_dim, heads=heads, dropout=dropout))
        
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.mem(x)

        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                x = block(x, rope=self.rope)
            else:
                x = block(x)
        
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x