import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from models.ResNet import ResNet50
from typing import Tuple
from models.swin_transformer import BasicLayer


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
# Attention Blocks
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

        rgb_tokens, (Hp, Wp) = self.rgb_patch(Ri)  # (L, B, E)
        dep_tokens, _ = self.dep_patch(Ti)

        self._ensure_unembed(Hp, Wp, Cr, Ct, device)

        rgb_self = self.rgb_self_attn(rgb_tokens, rgb_tokens, rgb_tokens)
        dep_self = self.dep_self_attn(dep_tokens, dep_tokens, dep_tokens)
        self_attention = rgb_self + dep_self

        rgb_cross = self.rgb_cross_attn(rgb_tokens, dep_tokens, dep_tokens)
        dep_cross = self.dep_cross_attn(dep_tokens, rgb_tokens, rgb_tokens)
        cross_attention = rgb_cross + dep_cross

        o = self_attention + cross_attention + rgb_tokens

        m = self.mlp(o)

        f = self._out_proj(m, (H, W)) + Ri

        return f, torch.permute(m + rgb_tokens, (1, 0, 2))


# ==================================================================

class TransformerGCM(nn.Module):
    """
    Transformer-based reimplementation of GCM with multi-branch token mixers,
    residual fusion, and optional unpatchify back to a feature map.

    Inputs:
      x_tokens: [B, N, D] (e.g., [1, 121, 128])
      grid_size: (H, W) tokens (e.g., (11, 11))
      If return_tokens=False, will unpatchify to [B, C_out, H*p, W*p] using patch_size=p.

    Args:
      dim: token embedding dim D
      heads: attention heads per branch
      out_channels: channels for the returned image-like tensor
      patch_size: ViT patch size used before fusion (so 4 when 44/11=4)
      mlp_ratio: MLP expansion
    """

    def __init__(
        self,
        dim: int = 128,
        input_resolution=(11, 11),
        window_size=1,
        heads: int = 4,
        depth=1,
        out_channels: int = 32,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        output_shape=(44, 44),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.input_resolution = input_resolution
        self.output_shape = output_shape

        self.branch1_attn = BasicLayer(
            dim=dim, input_resolution=input_resolution,
            depth=depth, num_heads=heads, window_size=window_size, mlp_ratio=mlp_ratio
        )

        self._out_proj = PatchUnembedConv(embed_dim=dim,
                                          out_ch=out_channels,
                                          patch_size=self.patch_size,
                                          Hp=input_resolution[0], Wp=input_resolution[0])

    def forward(
        self,
        x_tokens: torch.Tensor,
        fmap=False
    ):
        """
        x_tokens: [B, N, D]
        grid_size: (H, W) token grid (e.g., (11,11))
        """
        H, W = self.input_resolution
        b1 = self.branch1_attn(x_tokens)
        if fmap:
            b1 = b1.permute(1, 0, 2)
            return self._out_proj(b1, self.output_shape)
        return b1


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


class MultiScaleAttentionRefiner(nn.Module):
    def __init__(self, dims_in, dims_refine, out_dim=128, H=24, W=24):
        """
        Args:
            dims_in: list of channel dimensions of [fusion2, fusion3, fusion4] (e.g. [128, 128, 128])
            dims_refine: list of channel dimensions of [fusion0, fusion1, fusion2] (e.g. [32, 64, 128])
            out_dim: hidden embedding size for attention fusion
            H, W: spatial size of the highest resolution (e.g. 24x24 → 576 tokens)
        """
        super().__init__()
        self.H, self.W = H, W

        # Projection layers for input tokens
        self.proj_in = nn.ModuleList([
            nn.Linear(d, out_dim) for d in dims_in
        ])

        # Projection layers for refinement tokens
        self.proj_refine = nn.ModuleList([
            nn.Linear(d, out_dim) for d in dims_refine
        ])

        # Attention fusion
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=4, batch_first=True)

        # Output projections for refined tokens
        self.proj_out = nn.ModuleList([
            nn.Linear(out_dim, d) for d in dims_refine
        ])

    def forward(self, fusion0, fusion1, fusion2, fusion3, fusion4):
        """
        fusion0: [N, 576, 32]  → corresponds to [32,24,24]
        fusion1: [N, 576, 64]
        fusion2: [N, 144, 128] → corresponds to [128,12,12]
        fusion3: [N, 144, 128] → corresponds to [128,6,6]
        fusion4: [N, 576, 128] → corresponds to [128,3,3] but reshaped to tokens
        """

        B = fusion0.size(0)

        # --- Step 1: Project and Upsample fusion2, fusion3, fusion4 to [N, 576, out_dim] ---
        inputs = []
        for i, (f, proj) in enumerate(zip([fusion2, fusion3, fusion4], self.proj_in)):
            x = proj(f)  # [N, L, out_dim]
            L = x.size(1)
            spatial_size = int(L ** 0.5)  # assume square
            x = x.transpose(1, 2).reshape(B, -1, spatial_size, spatial_size)
            x = F.interpolate(x, size=(self.H, self.W),
                              mode="bilinear", align_corners=False)
            x = x.flatten(2).transpose(1, 2)  # [N, 576, out_dim]
            inputs.append(x)

        fused = torch.stack(inputs, dim=0).mean(
            0)  # average fusion → [N, 576, out_dim]

        # --- Step 2: Self-attention to get attention map ---
        # [N, 576, out_dim], [N, 576, 576]
        attn_out, attn_weights = self.attn(fused, fused, fused)

        # reshape attention map to [N, H, W]
        attn_map = attn_weights.mean(1).reshape(
            B, self.H, self.W)  # average across queries

        # --- Step 3: Refine fusion0, fusion1, fusion2 ---
        refined = []
        for f, proj_in, proj_out in zip([fusion0, fusion1, fusion2], self.proj_refine, self.proj_out):
            # [N, 576, out_dim] (note: fusion2 will be upsampled too)
            x = proj_in(f)
            if x.size(1) != self.H * self.W:
                # Upsample to 24x24
                L = x.size(1)
                spatial_size = int(L ** 0.5)
                x = x.transpose(1, 2).reshape(
                    B, -1, spatial_size, spatial_size)
                x = F.interpolate(x, size=(self.H, self.W),
                                  mode="bilinear", align_corners=False)
                x = x.flatten(2).transpose(1, 2)

            # apply attention (elementwise modulation)
            attn_factor = attn_map.flatten(1).unsqueeze(-1)  # [N, 576, 1]
            x = x * attn_factor
            refined.append(proj_out(x))  # project back to original dim

        return refined, attn_map.unsqueeze(1)


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

        dropout = 0.1

        heads_stage = (4, 4, 8, 8, 8)
        embed_dim = (32, 64, 128, 128, 128)  # embed_dim for each stage
        C = [64, 256, 512, 1024, 2048]           # channel dims from ResNet
        S = [96, 96, 48, 24, 12]

        # you can also set per stage (smaller patch for early layers)
        P = [4, 4, 2, 2, 1]

        # Replace FusionBlock2D with RGBDViTBlock
        self.fuse0 = RGBDViTBlock(
            in_ch_r=C[0], in_ch_t=C[0],
            embed_dim=embed_dim[0], num_heads=heads_stage[0],
            patch_size=P[0], dropout=dropout, attn_dropout=dropout
        )
        self.fuse1 = RGBDViTBlock(
            in_ch_r=C[1], in_ch_t=C[1],
            embed_dim=embed_dim[1], num_heads=heads_stage[1],
            patch_size=P[1], dropout=dropout, attn_dropout=dropout
        )
        self.fuse2 = RGBDViTBlock(
            in_ch_r=C[2], in_ch_t=C[2],
            embed_dim=embed_dim[2], num_heads=heads_stage[2],
            patch_size=P[2], dropout=dropout, attn_dropout=dropout
        )
        self.fuse3_1 = RGBDViTBlock(
            in_ch_r=C[3], in_ch_t=C[3],
            embed_dim=embed_dim[3], num_heads=heads_stage[3],
            patch_size=P[3], dropout=dropout, attn_dropout=dropout
        )
        self.fuse4_1 = RGBDViTBlock(
            in_ch_r=C[4], in_ch_t=C[4],
            embed_dim=embed_dim[4], num_heads=heads_stage[4],
            patch_size=P[4], dropout=dropout, attn_dropout=dropout
        )

        W = [8, 8, 8, 4, 4]
        self.rfb2_1 = TransformerGCM(
            dim=embed_dim[2], input_resolution=(S[2]//P[2], S[2]//P[2]), window_size=W[2],
            depth=2, out_channels=32, patch_size=P[2], output_shape=(S[2], S[2])
        )

        self.rfb3_1 = TransformerGCM(
            dim=embed_dim[3], input_resolution=(S[3]//P[3], S[3]//P[3]), window_size=W[3],
            depth=2, out_channels=32, patch_size=P[3], output_shape=(S[3], S[3])
        )

        self.rfb4_1 = TransformerGCM(
            dim=embed_dim[4], input_resolution=(S[4]//P[4], S[4]//P[4]), window_size=W[4],
            depth=2, out_channels=32, patch_size=P[4], output_shape=(S[4], S[4])
        )

        self.rfb0_2 = TransformerGCM(
            dim=embed_dim[0], input_resolution=(S[0]//P[0], S[0]//P[0]), window_size=W[0],
            depth=2, out_channels=32, patch_size=P[2], output_shape=(S[0], S[0])
        )

        self.rfb1_2 = TransformerGCM(
            dim=embed_dim[1], input_resolution=(S[1]//P[1], S[1]//P[1]), window_size=W[1],
            depth=2, out_channels=32, patch_size=P[1], output_shape=(S[1], S[1])
        )

        self.rfb5_2 = TransformerGCM(
            dim=embed_dim[2], input_resolution=(S[2]//P[2], S[2]//P[2]), window_size=W[2],
            depth=2, out_channels=32, patch_size=P[2], output_shape=(S[2], S[2])
        )

        self.refiner = MultiScaleAttentionRefiner(
            dims_in=[128, 128, 128],   # fusion2, fusion3, fusion4
            dims_refine=[32, 64, 128],  # fusion0, fusion1, fusion2
            out_dim=128,
            H=24, W=24
        )

        # input shape: [64, 96, 96] = [C1, H, W]
        # fusion0: [H/P0 * H/P0, E0] = [576, 32]
        # fusion1: [H/P1 * H/P1, E1] = [576, 64]
        # fusion2: [(H/2)/P2 * (H/2)/P2, E2] = [576, 128]
        # fusion3: [(H/4)/P3 * (H/4)/P3, E3] = [144, 128]
        # fusion4: [(H/8)/P4 * (H/8)/P4, E4] = [144, 128]

        self.agg2 = aggregation_final_swin(32)

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        # stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # [2, 64, 96, 96]

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)  # [2, 64, 96, 96]

        # ---- layer0 fusion (64ch) ----
        # [2, 64, 96, 96], [2, 576, 32]
        x, fused_tokens0 = self.fuse0(x, x_depth)

        # layer1
        x1 = self.resnet.layer1(x)  # [2, 256, 96, 96]
        x1_depth = self.resnet_depth.layer1(x_depth)  # [2, 256, 96, 96]

        # ---- layer1 fusion (256ch) ----
        # [2, 256, 96, 96], [2, 576, 64]
        x1, fused_tokens1 = self.fuse1(x1, x1_depth)

        # layer2
        x2 = self.resnet.layer2(x1)  # [2, 512, 48, 48]
        x2_depth = self.resnet_depth.layer2(x1_depth)  # [2, 512, 48, 48]

        # ---- layer2 fusion (512ch) ----
        # [2, 512, 48, 48], [2, 576, 128]
        x2, fused_tokens2 = self.fuse2(x2, x2_depth)

        # layer3_1
        x3_1 = self.resnet.layer3_1(x2)  # [2, 1024, 24, 24]
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)  # [2, 1024, 24, 24]

        # ---- layer3_1 fusion (1024ch) ----
        x3_1, fused_tokens3 = self.fuse3_1(
            x3_1, x3_1_depth)  # [2, 1024, 24, 24], [2, 144, 128]

        # layer4_1
        x4_1 = self.resnet.layer4_1(x3_1)  # [2, 2048, 12, 12]
        x4_1_depth = self.resnet_depth.layer4_1(
            x3_1_depth)  # [2, 2048, 12, 12]

        # ---- layer4_1 fusion (2048ch) ----
        x4_1, fused_tokens4 = self.fuse4_1(
            x4_1, x4_1_depth)  # [2, 2048, 12, 12], [2, 144, 128]

        # produce initial saliency map by decoder1
        x2_1 = self.rfb2_1(fused_tokens2)  # → [1, 32, 48, 48]
        x3_1 = self.rfb3_1(fused_tokens3)  # → [1, 32, 48, 48]
        x4_1 = self.rfb4_1(fused_tokens4)  # → [1, 32, 12, 12]

        (x, x1, x5), attention_map = self.refiner(
            fused_tokens0, fused_tokens1, x2_1, x3_1, x4_1
        )

        # produce final saliency map by decoder2
        x0_2 = self.rfb0_2(x, fmap=True)  # [2, 32, 96, 96]
        x1_2 = self.rfb1_2(x1, fmap=True)  # [2, 32, 96, 96]
        x5_2 = self.rfb5_2(x5, fmap=True)  # [2, 32, 48, 48]

        y = self.agg2(x5_2, x1_2, x0_2)  # [1, 96, 88, 88]

        # PTM module
        y = self.agant1(y)  # [1, 64, 88, 88]
        y = self.deconv1(y)  # [1, 64, 176, 176]
        y = self.agant2(y)  # [1, 32, 176, 176]
        y = self.deconv2(y)  # [1, 32, 352, 352]
        y = self.out2_conv(y)  # [1, 1, 352, 352]

        s1 = F.interpolate(
            attention_map, size=y.shape[2:], mode='bilinear', align_corners=True)
        return s1, y
