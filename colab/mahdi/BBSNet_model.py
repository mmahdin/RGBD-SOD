import math
import torch.nn.functional as F
from typing import Tuple
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from models.ResNet import ResNet50


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel,
                        kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
            * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
            * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, attention, x1, x2, x3):
        # Note that there is an error in the manuscript. In the paper, the refinement strategy is depicted as ""f'=f*S1"", it should be ""f'=f+f*S1"".
        x1 = x1+torch.mul(x1, self.upsample2(attention))
        x2 = x2+torch.mul(x2, self.upsample2(attention))
        x3 = x3+torch.mul(x3, attention)

        return x1, x2, x3


# =========================
# ViT Fusion Blocks
# =========================
class PatchEmbedConv(nn.Module):
    """
    Patchify via a conv layer: conv(in_ch -> embed_dim, kernel_size=patch_size, stride=patch_size)
    Input: (B, C, H, W)
    Output tokens: (L, B, embed_dim) where L = (H/ps)*(W/ps)
    Also returns spatial size (H', W') for unpatchify.
    """

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int):
        super().__init__()
        assert patch_size >= 1
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize pos_embed later because L = H/ps * W/ps is unknown until forward
        self.pos_embed = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H ({H}) and W ({W}) must be divisible by patch_size ({self.patch_size})"

        x_p = self.proj(x)  # (B, embed_dim, H', W')
        B, E, Hp, Wp = x_p.shape
        tokens = x_p.flatten(2).transpose(1, 2)  # (B, L, E), L = Hp*Wp

        # --- Add learnable positional embeddings ---
        L = tokens.shape[1]
        if self.pos_embed is None or self.pos_embed.shape[1] != L:
            # Create learnable pos embedding per patch location
            self.pos_embed = nn.Parameter(
                torch.zeros(1, L, E, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        tokens = tokens + self.pos_embed  # (B, L, E)

        tokens = self.norm(tokens)
        tokens = tokens.transpose(0, 1)  # (L, B, E)
        return tokens, (Hp, Wp)


class PatchUnembedConv(nn.Module):
    """
    Project token embeddings back to spatial feature map and upsample to original HxW.
    - in_tokens: (L, B, embed_dim)
    - returns: (B, out_ch, H, W) that can be added to original Ri
    """

    def __init__(self, embed_dim: int, out_ch: int, patch_size: int, Hp: int, Wp: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_ch = out_ch
        self.patch_size = patch_size
        # project embed_dim -> out_ch (for each low-res spatial location)
        self.to_spatial = nn.Linear(embed_dim, out_ch)
        self.Hp = Hp
        self.Wp = Wp

    def forward(self, tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # tokens: (L, B, E)
        L, B, E = tokens.shape
        Hp = self.Hp
        Wp = self.Wp
        assert L == Hp * Wp, "token length doesn't match Hp*Wp"

        # to (B, L, out_ch)
        t = tokens.transpose(0, 1)  # (B, L, E)
        t = self.to_spatial(t)      # (B, L, out_ch)
        # reshape to (B, out_ch, Hp, Wp)
        t = t.transpose(1, 2).contiguous().view(B, self.out_ch, Hp, Wp)
        # upsample to original H, W
        H, W = target_hw
        if (Hp * self.patch_size, Wp * self.patch_size) != (H, W):
            # just in case target_hw provided doesn't match, compute desired scale
            t = F.interpolate(t, size=(H, W), mode='bilinear',
                              align_corners=False)
        else:
            t = F.interpolate(t, scale_factor=self.patch_size,
                              mode='bilinear', align_corners=False)
        return t


class CrossMultiHeadAttention(nn.Module):
    """
    Wrapper for cross-attention using nn.MultiheadAttention.
    - query: (Lq, B, E)
    - key/value: (Lk, B, E)
    Returns: attn_out shape (Lq, B, E)
    """

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=False)
        self.dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Identity()  # placeholder if you want an output linear

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None):
        # query: (Lq, B, E), key/value: (Lk, B, E)
        # nn.MultiheadAttention expects (L, B, E) when batch_first=False
        attn_out, _ = self.mha(query, key, value, attn_mask=attn_mask)
        attn_out = self.dropout(attn_out)
        return self.proj(attn_out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (L, B, E)
        # apply LN over last dim: easier to transpose to (B,L,E)
        x_t = x.transpose(0, 1)   # (B, L, E)
        y = self.norm(x_t)
        y = self.fc1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y.transpose(0, 1)     # (L, B, E)
        return y


class RGBDViTBlock(nn.Module):
    """
    Single-level fusion block using:
      - self-attention (per modality)
      - cross-attention (query=RGB, key/value=Depth and vice-versa)
      - MLP (token-wise)
      - residual connections back to original Ri/Ti feature maps.

    Input:
      Ri: (B, C_r, H, W)  - RGB features from ResNet
      Ti: (B, C_t, H, W)  - Depth features from separate ResNet (same spatial size)
    Output:
      Fr: (B, C_r, H, W)  - fused RGB features
      Ft: (B, C_t, H, W)  - fused Depth features

    Notes:
      - patch_size must divide H and W.
      - embed_dim: dimension used inside attention tokens.
    """

    def __init__(self,
                 in_ch_r: int,
                 in_ch_t: int,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 patch_size: int = 4,
                 mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        # Patch embedders for each modality (conv-based)
        self.rgb_patch = PatchEmbedConv(in_ch_r, embed_dim, patch_size)
        self.dep_patch = PatchEmbedConv(in_ch_t, embed_dim, patch_size)

        # Self-attention for each modality
        self.rgb_self_attn = CrossMultiHeadAttention(
            embed_dim, num_heads, attn_dropout)
        self.dep_self_attn = CrossMultiHeadAttention(
            embed_dim, num_heads, attn_dropout)

        # Cross-attention: rgb queries, depth keys/vals and vice versa
        self.rgb_cross_attn = CrossMultiHeadAttention(
            embed_dim, num_heads, attn_dropout)
        self.dep_cross_attn = CrossMultiHeadAttention(
            embed_dim, num_heads, attn_dropout)

        # MLPs
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(embed_dim, hidden_dim, dropout)

        # Project back to original channel counts
        # We'll use a small linear inside PatchUnembed but need to create it after we know Hp/Wp - so build later dynamically
        self._out_proj = None  # placeholder

    def _ensure_unembed(self, Hp, Wp, in_ch_r, in_ch_t, device):
        if (self._out_proj is None) or (self._out_proj.Hp != Hp):
            self._out_proj = PatchUnembedConv(embed_dim=self.rgb_patch.proj.out_channels,
                                              out_ch=in_ch_r,
                                              patch_size=self.patch_size,
                                              Hp=Hp, Wp=Wp)
            # move to device
            self._out_proj.to(device)

    def forward(self, Ri: torch.Tensor, Ti: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ri: (B, C_r, H, W)
        Ti: (B, C_t, H, W)
        returns: Fr, Ft same shapes as Ri and Ti
        """
        B, Cr, H, W = Ri.shape
        B2, Ct, H2, W2 = Ti.shape
        assert B == B2 and H == H2 and W == W2, "Inputs must have same batch and spatial dims"
        device = Ri.device

        # Patchify both
        # [1, 64, 88, 88] --> [484, 1, 32]
        rgb_tokens, (Hp, Wp) = self.rgb_patch(Ri)  # (L, B, E)

        # [1, 64, 88, 88] --> [484, 1, 32]
        dep_tokens, _ = self.dep_patch(Ti)

        # L = rgb_tokens.shape[0]
        # ensure unembed modules created
        self._ensure_unembed(Hp, Wp, Cr, Ct, device)

        # --- RGB stream ---
        # Self-attn (pre-LN style already applied in PatchEmbedConv)
        rgb_self = self.rgb_self_attn(rgb_tokens, rgb_tokens, rgb_tokens)
        dep_self = self.dep_self_attn(dep_tokens, dep_tokens, dep_tokens)
        self_attention = rgb_self + dep_self

        # Cross-attn: rgb queries, depth key/value
        rgb_cross = self.rgb_cross_attn(rgb_tokens, dep_tokens, dep_tokens)
        dep_cross = self.dep_cross_attn(dep_tokens, rgb_tokens, rgb_tokens)
        cross_attention = rgb_cross + dep_cross

        o = self_attention + cross_attention + rgb_tokens

        m = self.mlp(o)

        f = self._out_proj(m, (H, W)) + Ri

        return f


# =========================
# ViT Fusion Blocks
# =========================

def get_2d_sincos_pos_embed(h, w, dim, device):
    """
    2D sine-cos positional embedding like ViT (no cls token).
    Returns [H*W, dim]
    """
    assert dim % 4 == 0, "pos dim must be multiple of 4"
    grid_y = torch.arange(h, device=device, dtype=torch.float32)
    grid_x = torch.arange(w, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [H,W]
    gy = gy.reshape(-1)  # [HW]
    gx = gx.reshape(-1)

    def pe(pos, d_half):
        # pos: [HW], d_half: int
        div_term = torch.exp(torch.arange(
            0, d_half, 2, device=device).float() * (-math.log(10000.0) / d_half))
        sin = torch.sin(pos.unsqueeze(1) * div_term)      # [HW, d_half/2]
        cos = torch.cos(pos.unsqueeze(1) * div_term)      # [HW, d_half/2]
        return torch.cat([sin, cos], dim=1)               # [HW, d_half]

    d_quarter = dim // 4
    emb_y = pe(gy, 2 * d_quarter)   # [HW, dim/2]
    emb_x = pe(gx, 2 * d_quarter)   # [HW, dim/2]
    return torch.cat([emb_y, emb_x], dim=1)  # [HW, dim]


class PatchEmbed(nn.Module):
    """
    Conv patchifying that also handles channel lift/reduce to d_model.
    """

    def __init__(self, in_ch, d_model, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size,
                              stride=patch_size, padding=0, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B,C,H,W] -> tokens [B, HW/P^2, d_model], spatial (H', W')
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, d_model, H', W']
        Hp, Wp = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d_model]
        x = self.norm(x)
        return x, (Hp, Wp)


class PatchUnembed(nn.Module):
    """
    Reverse of PatchEmbed with a 1x1 to map back to C and a conv upsample if patch_size>1.
    """

    def __init__(self, out_ch, d_model, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.reproj = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Conv2d(
            d_model, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, tokens, spatial_hw):
        # tokens: [B, N, d_model]; spatial_hw = (H', W')
        B, N, D = tokens.shape
        Hp, Wp = spatial_hw
        # x = self.reproj(tokens)              # [B,N,D]
        x = tokens
        x = x.transpose(1, 2).reshape(B, D, Hp, Wp)  # [B,D,H',W']
        x = self.to_out(x)                   # [B,C,H',W']
        if self.patch_size > 1:
            x = F.interpolate(x, size=(Hp * self.patch_size,
                              Wp * self.patch_size), mode="nearest")
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttnBlock(nn.Module):
    """
    PreNorm Self-Attention + MLP
    """

    def __init__(self, dim, num_heads=8, drop=0.0, attn_drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        # self.drop = nn.Dropout(drop)
        # self.norm2 = nn.LayerNorm(dim)
        # self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        # x: [B,N,D]
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        # x = x + self.drop(attn_out)
        # x = x + self.mlp(self.norm2(x))
        return attn_out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=6, num_heads=8, drop=0.0, attn_drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttnBlock(dim, num_heads, drop, attn_drop, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttnBlock(nn.Module):
    """
    Cross-attention Transformer block:
    - q from x, kv from y
    - residual + MLP
    """

    def __init__(self, dim, num_heads=8, drop=0.0, attn_drop=0.0, mlp_ratio=4.0):
        super().__init__()
        # Norms
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Cross Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        # self.drop = nn.Dropout(drop)

        # Feedforward
        # self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x, y):
        """
        Cross-attention: x attends to y (q=x, kv=y).
        Args:
            x: [B, Nx, D]
            y: [B, Ny, D]
        Returns:
            Updated x
        """
        # Cross-attention
        q = self.q_norm(x)
        kv = self.kv_norm(y)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        # x = x + self.drop(attn_out)

        # Feedforward
        # x = x + self.mlp(self.norm2(x))
        return attn_out


class CrossAttnEncoder(nn.Module):
    def __init__(self, dim, depth=6, num_heads=8, drop=0.0, attn_drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnBlock(dim, num_heads, drop, attn_drop, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x, y)
        return x


class XModalFusionBlock(nn.Module):
    """
    One stage:
      1) Self-attn on rgb and depth separately
      2) Cross-attn both directions
      3) Fuse outputs (concat + linear) or gated sum
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0,
                 attn_drop=0.0, self_depth=1, cross_depth=1):
        super().__init__()
        self.rgb_self = TransformerEncoder(
            dim, self_depth, num_heads, drop, attn_drop, mlp_ratio)
        self.dep_self = TransformerEncoder(
            dim, self_depth, num_heads, drop, attn_drop, mlp_ratio)
        self.rgb_from_dep = CrossAttnEncoder(
            dim, cross_depth, num_heads, drop, attn_drop, mlp_ratio)
        self.dep_from_rgb = CrossAttnEncoder(
            dim, cross_depth, num_heads, drop, attn_drop, mlp_ratio)

        # gates (scalar version, could also be vector)
        self.gate_rgb = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_dep = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        # norm + MLP
        # self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop=drop)

    def forward(self, rgb, dep):
        # self-attention
        s_rgb = self.rgb_self(rgb)
        s_dep = self.dep_self(dep)

        # cross attention
        c_rgb = self.rgb_from_dep(s_rgb, s_dep)
        c_dep = self.dep_from_rgb(s_dep, s_rgb)

        g_rgb = self.gate_rgb(c_rgb)  # Now [B, N, D]
        g_dep = self.gate_dep(c_dep)

        # gated fusion
        z = s_rgb + s_dep + g_rgb * c_rgb + g_dep * c_dep

        # norm + mlp (residual style)
        z = z + self.mlp(z)
        return z


class RGBDTransformerFusion(nn.Module):
    """
    Transformer-based RGB-D fusion that keeps input spatial size & channels.

    Args:
        in_ch (int): channels of RGB/Depth feature maps (must match)
        d_model (int): token embedding dim (e.g., 256/384)
        patch_size (int): patchify stride (1 keeps per-pixel tokens)
        num_heads (int)
        depth (int): number of fusion layers (stacked)
        mlp_ratio (float)
        drop (float)
        attn_drop (float)
        fuse (str): 'gated' or 'concat'
    """

    def __init__(self,
                 in_ch,
                 d_model=384,
                 patch_size=1,
                 num_heads=8,
                 depth=2,
                 self_depth=1,
                 cross_depth=1,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_rgb = PatchEmbed(in_ch, d_model, patch_size)
        self.embed_dep = PatchEmbed(in_ch, d_model, patch_size)
        self.blocks = nn.ModuleList([
            XModalFusionBlock(d_model, num_heads, mlp_ratio,
                              drop, attn_drop, self_depth=self_depth,
                              cross_depth=cross_depth)
            for _ in range(depth)
        ])
        self.unembed = PatchUnembed(in_ch, d_model, patch_size)
        self.pos_drop = nn.Dropout(drop)

    def forward(self, feat_rgb, feat_dep):
        """
        feat_rgb, feat_dep: [B, C, H, W] (same shape)
        returns: fused [B, C, H, W]
        """
        assert feat_rgb.shape == feat_dep.shape, "RGB and Depth features must have same shape"
        B, C, H, W = feat_rgb.shape
        device = feat_rgb.device

        # patchify to tokens
        rgb_tok, (Hp, Wp) = self.embed_rgb(feat_rgb)  # [B,N,D]
        dep_tok, _ = self.embed_dep(feat_dep)  # [B,N,D]
        N = Hp * Wp

        # add 2D sine-cos positional embeddings
        pos = get_2d_sincos_pos_embed(
            Hp, Wp, rgb_tok.size(-1), device)  # [N,D]
        pos = pos.unsqueeze(0).expand(B, N, -1)  # [B,N,D]
        rgb_tok = rgb_tok + pos
        dep_tok = dep_tok + pos
        # rgb_tok = self.pos_drop(rgb_tok)
        # dep_tok = self.pos_drop(dep_tok)

        # stacked fusion layers
        z = None
        for blk in self.blocks:
            z = blk(rgb_tok, dep_tok)  # fused tokens [B,N,D]
            # update streams with fused info (residual guidance)
            rgb_tok = rgb_tok + 0.5 * z
            dep_tok = dep_tok + 0.5 * z

        # unpatchify back to original channel/size
        fused = self.unembed(z, (Hp, Wp))  # [B,C,H',W']

        if fused.shape[-2:] != (H, W):
            # Fallback (should match when patch_size divides H,W; if not, interpolate)
            fused = F.interpolate(fused, size=(
                H, W), mode="bilinear", align_corners=False)
        return fused + feat_rgb


# =========================
# Swin Fusion Blocks
# =========================
class RGBDFusion4Swin(nn.Module):
    """
    Transformer-based RGB-D fusion that keeps input spatial size & channels.

    Args:
        in_ch (int): channels of RGB/Depth feature maps (must match)
        d_model (int): token embedding dim (e.g., 256/384)
        patch_size (int): patchify stride (1 keeps per-pixel tokens)
        num_heads (int)
        depth (int): number of fusion layers (stacked)
        mlp_ratio (float)
        drop (float)
        attn_drop (float)
        fuse (str): 'gated' or 'concat'
    """

    def __init__(self,
                 in_ch,
                 d_model=384,
                 patch_size=1,
                 num_heads=8,
                 depth=2,
                 self_depth=1,
                 cross_depth=1,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_rgb = PatchEmbed(in_ch, d_model, patch_size)
        self.embed_dep = PatchEmbed(in_ch, d_model, patch_size)
        self.blocks = nn.ModuleList([
            XModalFusionBlock(d_model, num_heads, mlp_ratio,
                              drop, attn_drop, self_depth=self_depth,
                              cross_depth=cross_depth)
            for _ in range(depth)
        ])
        self.unembed = PatchUnembed(in_ch, d_model, patch_size)
        self.pos_drop = nn.Dropout(drop)

    def forward(self, feat_rgb, feat_dep):
        """
        feat_rgb, feat_dep: [B, C, H, W] (same shape)
        returns: fused [B, C, H, W]
        """
        assert feat_rgb.shape == feat_dep.shape, "RGB and Depth features must have same shape"
        B, C, H, W = feat_rgb.shape
        device = feat_rgb.device

        # patchify to tokens
        rgb_tok, (Hp, Wp) = self.embed_rgb(feat_rgb)  # [B,N,D]
        dep_tok, _ = self.embed_dep(feat_dep)  # [B,N,D]
        N = Hp * Wp

        # add 2D sine-cos positional embeddings
        pos = get_2d_sincos_pos_embed(
            Hp, Wp, rgb_tok.size(-1), device)  # [N,D]
        pos = pos.unsqueeze(0).expand(B, N, -1)  # [B,N,D]
        rgb_tok = rgb_tok + pos
        dep_tok = dep_tok + pos
        # rgb_tok = self.pos_drop(rgb_tok)
        # dep_tok = self.pos_drop(dep_tok)

        # stacked fusion layers
        z = None
        for blk in self.blocks:
            z = blk(rgb_tok, dep_tok)  # fused tokens [B,N,D]
            # update streams with fused info (residual guidance)
            rgb_tok = rgb_tok + 0.5 * z
            dep_tok = dep_tok + 0.5 * z

        # unpatchify back to original channel/size
        fused = self.unembed(z, (Hp, Wp))  # [B,C,H',W']

        if fused.shape[-2:] != (H, W):
            # Fallback (should match when patch_size divides H,W; if not, interpolate)
            fused = F.interpolate(fused, size=(
                H, W), mode="bilinear", align_corners=False)
        return fused + feat_rgb


class RefineSwin(nn.Module):
    def __init__(self):
        super(RefineSwin, self).__init__()
        self.mode = 'bilinear'
        self.align_corners = True

    def forward(self, attention, x1, x2, x3):
        # interpolate to match spatial sizes (broadcast will handle channels)
        att1 = F.interpolate(
            attention, size=x1.shape[2:], mode=self.mode, align_corners=self.align_corners)
        att2 = F.interpolate(
            attention, size=x2.shape[2:], mode=self.mode, align_corners=self.align_corners)
        att3 = F.interpolate(
            attention, size=x3.shape[2:], mode=self.mode, align_corners=self.align_corners)

        # refinement: f' = f + f * S  (attention S is single-channel, broadcast across C)
        x1 = x1 + x1 * att1
        x2 = x2 + x2 * att2
        x3 = x3 + x3 * att3

        return x1, x2, x3


class aggregation_final_swin(nn.Module):

    def __init__(self, channel):
        super(aggregation_final_swin, self).__init__()
        self.relu = nn.ReLU(True)

        self.mode = 'bilinear'
        self.align_corners = True

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        # x1: [1, 32, 24, 24]
        # x2: [1, 32, 48, 48]
        # x3: [1, 32, 96, 96]

        x1u = F.interpolate(
            x1, size=x3.shape[2:], mode=self.mode, align_corners=self.align_corners)
        x2u = F.interpolate(
            x2, size=x3.shape[2:], mode=self.mode, align_corners=self.align_corners)
        x2_1 = self.conv_upsample1(x1u) * x2u

        x3_1 = self.conv_upsample2(x1u) \
            * self.conv_upsample3(x2u) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1u)), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2


# ==================================================================


class BaseModel(nn.Module):
    def __init__(self, channel=32):
        super().__init__()

        # Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')

        # Decoder 1
        self.rfb2_1 = GCM(512, channel)
        self.rfb3_1 = GCM(1024, channel)
        self.rfb4_1 = GCM(2048, channel)
        self.agg1 = aggregation_init(channel)

        # Decoder 2
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(256, channel)
        self.rfb5_2 = GCM(512, channel)
        self.agg2 = aggregation_final(channel)

        # upsample function
        self.upsample = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # Refinement flow
        self.HA = Refine()

        # Components of PTM module
        self.inplanes = 32*2
        self.deconv1 = self._make_transpose(TransBasicBlock, 32*2, 3, stride=2)
        self.inplanes = 32
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32*3, 32*2)
        self.agant2 = self._make_agant_layer(32*2, 32)
        self.out0_conv = nn.Conv2d(32*3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(32*2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(32*1, 1, kernel_size=1, stride=1, bias=True)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    # initialize the weights
    def initialize_weights(self):
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(
            self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)


class BBSNetChannelSpatialAttention(BaseModel):
    def __init__(self):
        super().__init__()

        # Components of DEM module
        self.atten_depth_channel_0 = ChannelAttention(64)
        self.atten_depth_channel_1 = ChannelAttention(256)
        self.atten_depth_channel_2 = ChannelAttention(512)
        self.atten_depth_channel_3_1 = ChannelAttention(1024)
        self.atten_depth_channel_4_1 = ChannelAttention(2048)

        self.atten_depth_spatial_0 = SpatialAttention()
        self.atten_depth_spatial_1 = SpatialAttention()
        self.atten_depth_spatial_2 = SpatialAttention()
        self.atten_depth_spatial_3_1 = SpatialAttention()
        self.atten_depth_spatial_4_1 = SpatialAttention()

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        # layer0 merge
        temp = x_depth.mul(self.atten_depth_channel_0(x_depth))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        x = x + temp
        # layer0 merge end

        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)

        # layer1 merge
        temp = x1_depth.mul(self.atten_depth_channel_1(x1_depth))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        x1 = x1 + temp
        # layer1 merge end

        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        # layer2 merge
        temp = x2_depth.mul(self.atten_depth_channel_2(x2_depth))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        x2 = x2 + temp
        # layer2 merge end

        x2_1 = x2

        x3_1 = self.resnet.layer3_1(x2_1)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        # layer3_1 merge
        temp = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))
        temp = temp.mul(self.atten_depth_spatial_3_1(temp))
        x3_1 = x3_1 + temp
        # layer3_1 merge end

        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        # layer4_1 merge
        temp = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))
        temp = temp.mul(self.atten_depth_spatial_4_1(temp))
        x4_1 = x4_1 + temp
        # layer4_1 merge end

        # produce initial saliency map by decoder1
        x2_1 = self.rfb2_1(x2_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        attention_map = self.agg1(x4_1, x3_1, x2_1)

        # Refine low-layer features by initial map
        x, x1, x5 = self.HA(attention_map.sigmoid(), x, x1, x2)

        # produce final saliency map by decoder2
        x0_2 = self.rfb0_2(x)
        x1_2 = self.rfb1_2(x1)
        x5_2 = self.rfb5_2(x5)

        y = self.agg2(x5_2, x1_2, x0_2)

        # PTM module
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)

        return self.upsample(attention_map), y


class BBSNetTransformerAttention(BaseModel):
    def __init__(self):
        super().__init__()

        heads = (4, 4, 8, 8, 8)
        embed_dim = (32, 64, 128, 128, 128)  # embed_dim for each stage
        depth = [1]*5
        self_depth = [1]*5
        cross_depth = [2]*5
        mlp_ratio = [4]*5
        attn_drop = [0.1]*5
        proj_drop = [0.1]*5
        C = [64, 256, 512, 1024, 2048]           # channel dims from ResNet

        # you can also set per stage (smaller patch for early layers)
        patch_size = [8, 8, 4, 2, 1]

        # Replace FusionBlock2D with RGBDViTBlock
        self.fuse0 = RGBDTransformerFusion(
            in_ch=C[0], d_model=embed_dim[0], patch_size=patch_size[0], num_heads=heads[0],
            depth=depth[0], mlp_ratio=mlp_ratio[0], drop=proj_drop[0], attn_drop=attn_drop[0],
            self_depth=self_depth[0], cross_depth=cross_depth[0]
        )
        self.fuse1 = RGBDTransformerFusion(
            in_ch=C[1], d_model=embed_dim[1], patch_size=patch_size[1], num_heads=heads[1],
            depth=depth[1], mlp_ratio=mlp_ratio[1], drop=proj_drop[1], attn_drop=attn_drop[1],
            self_depth=self_depth[1], cross_depth=cross_depth[1]
        )
        self.fuse2 = RGBDTransformerFusion(
            in_ch=C[2], d_model=embed_dim[2], patch_size=patch_size[2], num_heads=heads[2],
            depth=depth[2], mlp_ratio=mlp_ratio[2], drop=proj_drop[2], attn_drop=attn_drop[2],
            self_depth=self_depth[2], cross_depth=cross_depth[2]
        )
        self.fuse3_1 = RGBDTransformerFusion(
            in_ch=C[3], d_model=embed_dim[3], patch_size=patch_size[3], num_heads=heads[3],
            depth=depth[3], mlp_ratio=mlp_ratio[3], drop=proj_drop[3], attn_drop=attn_drop[3],
            self_depth=self_depth[3], cross_depth=cross_depth[3]
        )
        self.fuse4_1 = RGBDTransformerFusion(
            in_ch=C[4], d_model=embed_dim[4], patch_size=patch_size[4], num_heads=heads[4],
            depth=depth[4], mlp_ratio=mlp_ratio[4], drop=proj_drop[4], attn_drop=attn_drop[4],
            self_depth=self_depth[4], cross_depth=cross_depth[4]
        )

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        # stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # [1, 64, 88, 88]

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)  # [1, 64, 88, 88]
        # x_depth = x_depth.mul(self.atten_depth_channel_0(x_depth))

        # ---- layer0 fusion (64ch) ----
        x = self.fuse0(x, x_depth)  # [1, 64, 88, 88]

        # layer1
        x1 = self.resnet.layer1(x)  # [1, 256, 88, 88]
        x1_depth = self.resnet_depth.layer1(x_depth)  # [1, 256, 88, 88]
        # x1_depth = x1_depth.mul(self.atten_depth_channel_1(x1_depth))

        # ---- layer1 fusion (256ch) ----
        x1 = self.fuse1(x1, x1_depth)  # [1, 256, 88, 88]

        # layer2
        x2 = self.resnet.layer2(x1)  # [1, 512, 44, 44]
        x2_depth = self.resnet_depth.layer2(x1_depth)  # [1, 512, 44, 44]
        # x2_depth = x2_depth.mul(self.atten_depth_channel_2(x2_depth))

        # ---- layer2 fusion (512ch) ----
        x2 = self.fuse2(x2, x2_depth)  # [1, 512, 44, 44]

        # layer3_1
        x3_1 = self.resnet.layer3_1(x2)  # [1, 1024, 22, 22]
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)  # [1, 1024, 22, 22]
        # x3_1_depth = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))

        # ---- layer3_1 fusion (1024ch) ----
        x3_1 = self.fuse3_1(x3_1, x3_1_depth)  # [1, 1024, 22, 22]

        # layer4_1
        x4_1 = self.resnet.layer4_1(x3_1)  # [1, 2048, 11, 11]
        x4_1_depth = self.resnet_depth.layer4_1(
            x3_1_depth)  # [1, 2048, 11, 11]
        # x4_1_depth = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))

        # ---- layer4_1 fusion (2048ch) ----
        x4_1 = self.fuse4_1(x4_1, x4_1_depth)  # [1, 2048, 11, 11]

        # produce initial saliency map by decoder1
        x2_1 = self.rfb2_1(x2)  # [1, 32, 44, 44]
        x3_1 = self.rfb3_1(x3_1)  # [1, 32, 22, 22]
        x4_1 = self.rfb4_1(x4_1)  # [1, 32, 11, 11]

        attention_map = self.agg1(x4_1, x3_1, x2_1)  # [1, 1, 44, 44]

        # Refine low-layer features by initial map
        # [1, 64, 88, 88], [1, 256, 88, 88], [1, 512, 44, 44]
        x, x1, x5 = self.HA(attention_map.sigmoid(), x, x1, x2)

        # produce final saliency map by decoder2
        x0_2 = self.rfb0_2(x)  # [1, 32, 88, 88]
        x1_2 = self.rfb1_2(x1)  # [1, 32, 88, 88]
        x5_2 = self.rfb5_2(x5)  # [1, 32, 44, 44]

        y = self.agg2(x5_2, x1_2, x0_2)  # [1, 96, 88, 88]

        # PTM module
        y = self.agant1(y)  # [1, 64, 88, 88]
        y = self.deconv1(y)  # [1, 64, 176, 176]
        y = self.agant2(y)  # [1, 32, 176, 176]
        y = self.deconv2(y)  # [1, 32, 352, 352]
        y = self.out2_conv(y)  # [1, 1, 352, 352]

        return self.upsample(attention_map), y
