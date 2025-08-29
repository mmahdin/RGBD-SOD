from typing import Tuple, Optional
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from models.ResNet import ResNet50
from models.swin_cross import BasicLayer


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


class DynamicFusionGate(nn.Module):
    """
    Dynamic gating mechanism to weight the contributions of different attention outputs
    """

    def __init__(self, embed_dim: int, num_modalities: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, *inputs):
        # inputs: list of (L, B, E) tensors
        stacked = torch.stack(inputs, dim=2)  # (L, B, num_modalities, E)
        B, L, num_mod, E = stacked.shape[1], stacked.shape[0], stacked.shape[2], stacked.shape[3]

        # Compute gating weights
        flat = stacked.reshape(L, B, num_mod * E)
        weights = self.gate_net(flat)  # (L, B, num_modalities)

        # Apply weights
        weighted = stacked * weights.unsqueeze(-1)  # (L, B, num_modalities, E)
        output = torch.sum(weighted, dim=2)  # (L, B, E)

        return output


class EfficientMultiHeadAttention(nn.Module):
    """
    Efficient attention with optional local windowing for high-resolution features
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: Optional[int] = None,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask=None) -> torch.Tensor:
        # query, key, value: (L, B, E)
        L, B, E = query.shape
        H = self.num_heads
        D = self.head_dim

        # Project to query, key, value
        q = self.q_proj(query).view(L, B, H, D).transpose(0, 1)  # (B, H, L, D)
        k = self.k_proj(key).view(L, B, H, D).transpose(0, 1)    # (B, H, L, D)
        v = self.v_proj(value).view(L, B, H, D).transpose(0, 1)  # (B, H, L, D)

        # Apply local window attention if window_size is specified
        if self.window_size is not None and L > self.window_size:
            # Reshape to (B, H, L//ws, ws, D) where ws is window_size
            ws = self.window_size
            pad_len = (ws - L % ws) % ws
            if pad_len > 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))

            new_L = L + pad_len
            q = q.contiguous().view(B, H, new_L // ws, ws, D)
            k = k.contiguous().view(B, H, new_L // ws, ws, D)
            v = v.contiguous().view(B, H, new_L // ws, ws, D)

            # Compute attention within each window
            attn_weights = torch.matmul(
                q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, H, L//ws, ws, ws)

            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)  # (B, H, L//ws, ws, D)

            # Reshape back to (B, H, L, D)
            attn_output = attn_output.view(B, H, new_L, D)
            if pad_len > 0:
                attn_output = attn_output[:, :, :L, :]  # remove padding
        else:
            # Standard global attention
            attn_weights = torch.matmul(
                q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, H, L, L)

            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)  # (B, H, L, D)

        # Reshape and project back
        attn_output = attn_output.transpose(
            0, 1).contiguous().view(L, B, E)  # (L, B, E)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output


class PatchEmbedConv(nn.Module):
    """
    Improved patch embedding with optional overlapping patches
    """

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, overlap: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap

        if overlap:
            # Use overlapping patches with stride = patch_size//2
            stride = max(1, patch_size // 2)
            padding = patch_size // 4  # Adjust padding to maintain size
            self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
        else:
            # Non-overlapping patches
            self.proj = nn.Conv2d(
                in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = None

    @staticmethod
    def get_output_hw(H: int, W: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> Tuple[int, int]:
        """
        Compute output H' and W' for a Conv2D layer.
        """
        H_out = (H + 2 * padding - dilation *
                 (kernel_size - 1) - 1) // stride + 1
        W_out = (W + 2 * padding - dilation *
                 (kernel_size - 1) - 1) // stride + 1
        return H_out, W_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape

        x_p = self.proj(x)  # (B, embed_dim, H', W')
        B, E, Hp, Wp = x_p.shape
        tokens = x_p.flatten(2).transpose(1, 2)  # (B, L, E), L = Hp*Wp

        # Add learnable positional embeddings
        L = tokens.shape[1]
        if self.pos_embed is None or self.pos_embed.shape[1] != L:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, L, E, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        tokens = tokens + self.pos_embed
        tokens = self.norm(tokens)
        tokens = tokens.transpose(0, 1)  # (L, B, E)

        return tokens, (Hp, Wp)


class LearnedUpsample(nn.Module):
    """
    Learned upsampling using transposed convolution instead of simple interpolation
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=scale_factor,
            stride=scale_factor, groups=min(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = self.conv(x)
        # Ensure exact target size (handles edge cases)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size,
                              mode='bilinear', align_corners=False)
        return x


class PatchUnembedConv(nn.Module):
    """
    Improved unembedding with learned upsampling
    """

    def __init__(self, embed_dim: int, out_ch: int, patch_size: int, Hp: int, Wp: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_ch = out_ch
        self.patch_size = patch_size
        self.Hp = Hp
        self.Wp = Wp

        # Project to output channels
        self.to_spatial = nn.Linear(embed_dim, out_ch)

        # Learned upsampling instead of simple interpolation
        self.upsample = LearnedUpsample(out_ch, out_ch, patch_size)

    def forward(self, tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        L, B, E = tokens.shape
        Hp, Wp = self.Hp, self.Wp
        assert L == Hp * Wp, "token length doesn't match Hp*Wp"

        # Convert tokens to spatial feature map
        t = tokens.transpose(0, 1)  # (B, L, E)
        t = self.to_spatial(t)      # (B, L, out_ch)
        t = t.transpose(1, 2).contiguous().view(
            B, self.out_ch, Hp, Wp)  # (B, out_ch, Hp, Wp)

        # Upsample to target size using learned upsampling
        t = self.upsample(t, target_hw)

        return t


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
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
    Improved fusion block with:
    - Dynamic gated fusion
    - Efficient windowed attention for high-resolution features
    - Better tokenization and reconstruction
    """

    def __init__(self,
                 in_ch_r: int,
                 in_ch_t: int,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 patch_size: int = 4,
                 mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 shape: int = 12,
                 swin_depth: int = 2,
                 window_size: Optional[int] = None,
                 use_overlap: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.shape = shape

        if use_overlap:
            input_resolution = PatchEmbedConv.get_output_hw(
                shape, shape, kernel_size=patch_size, stride=patch_size//2, padding=patch_size//4)
        else:
            input_resolution = [shape // patch_size, shape // patch_size]

        self.input_resolution = input_resolution
        if window_size is None:
            window_size = shape

        self.window_size = window_size

        # Patch embedders with optional overlapping
        self.rgb_patch = PatchEmbedConv(
            in_ch_r, embed_dim, patch_size, use_overlap)
        self.dep_patch = PatchEmbedConv(
            in_ch_t, embed_dim, patch_size, use_overlap)

        # Self-attention for each modality with optional windowing
        self.rgb_self_attn = BasicLayer(
            dim=embed_dim, input_resolution=input_resolution,
            depth=swin_depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, attn_drop=attn_dropout
        )
        # self.rgb_self_attn = EfficientMultiHeadAttention(
        #     embed_dim, num_heads, window_size, attn_dropout)
        self.dep_self_attn = BasicLayer(
            dim=embed_dim, input_resolution=input_resolution,
            depth=swin_depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, attn_drop=attn_dropout
        )

        # Cross-attention
        self.rgb_cross_attn = BasicLayer(
            dim=embed_dim, input_resolution=input_resolution,
            depth=swin_depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, attn_drop=attn_dropout, cross_attention=True
        )
        self.dep_cross_attn = BasicLayer(
            dim=embed_dim, input_resolution=input_resolution,
            depth=swin_depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, attn_drop=attn_dropout, cross_attention=True
        )

        # Dynamic fusion gate
        self.fusion_gate = DynamicFusionGate(embed_dim, num_modalities=4)

        # MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(embed_dim, hidden_dim, dropout)

        # Output projection (will be created in forward)
        self._out_proj = None

    def _ensure_unembed(self, Hp, Wp, out_ch, device):
        if self._out_proj is None or self._out_proj.Hp != Hp or self._out_proj.Wp != Wp:
            self._out_proj = PatchUnembedConv(
                embed_dim=self.rgb_patch.proj.out_channels,
                out_ch=out_ch,
                patch_size=self.patch_size,
                Hp=Hp, Wp=Wp
            ).to(device)

    def forward(self, Ri: torch.Tensor, Ti: torch.Tensor) -> torch.Tensor:
        B, Cr, H, W = Ri.shape
        B2, Ct, H2, W2 = Ti.shape
        assert B == B2 and H == H2 and W == W2, "Inputs must have same batch and spatial dims"
        device = Ri.device

        # Patchify both modalities
        rgb_tokens, (Hp, Wp) = self.rgb_patch(Ri)  # (L, B, E)
        dep_tokens, _ = self.dep_patch(Ti)

        # Ensure unembed module is created
        self._ensure_unembed(Hp, Wp, Cr, device)

        rgb_tokens = rgb_tokens.permute(1, 0, 2)
        dep_tokens = dep_tokens.permute(1, 0, 2)

        # Self-attention
        rgb_self = self.rgb_self_attn(rgb_tokens)
        dep_self = self.dep_self_attn(dep_tokens)

        # Cross-attention
        rgb_cross = self.rgb_cross_attn(rgb_tokens, dep_tokens)
        dep_cross = self.dep_cross_attn(dep_tokens, rgb_tokens)

        rgb_self = rgb_self.permute(1, 0, 2)
        dep_self = dep_self.permute(1, 0, 2)
        rgb_cross = rgb_cross.permute(1, 0, 2)
        dep_cross = dep_cross.permute(1, 0, 2)
        rgb_tokens = rgb_tokens.permute(1, 0, 2)
        dep_tokens = dep_tokens.permute(1, 0, 2)

        # Dynamic fusion instead of simple addition
        fused_tokens = self.fusion_gate(
            rgb_self, dep_self, rgb_cross, dep_cross)
        fused_tokens = fused_tokens + rgb_tokens  # Residual connection

        # MLP
        mlp_out = self.mlp(fused_tokens)
        fused_tokens = fused_tokens + mlp_out  # Another residual connection

        # Convert back to spatial feature map
        f = self._out_proj(fused_tokens, (H, W))

        # Final residual connection with original input
        return f + Ri


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

        dropout = 0.2

        heads_stage = (4, 4, 8, 8, 8)
        # Increased early embed_dims for better representation
        embed_stage = (64, 64, 128, 128, 128)
        C = [64, 256, 512, 1024, 2048]
        S = [96, 96, 48, 24, 12]

        # More appropriate patch sizes for each stage
        patch_size = [4, 4, 2, 1, 1]  # Smaller patches for early layers

        # Window sizes for efficient attention (None for global attention)
        # Local attention for early layers, global for later
        window_sizes = [8, 8, 8, None, None]

        # Whether to use overlapping patches
        # Overlap for early layers
        use_overlap = [True, True, False, False, False]

        # Replace FusionBlock2D with improved RGBDViTBlock
        self.fuse0 = RGBDViTBlock(
            in_ch_r=C[0], in_ch_t=C[0], shape=S[0],
            embed_dim=embed_stage[0], num_heads=heads_stage[0],
            patch_size=patch_size[0], dropout=dropout, attn_dropout=dropout,
            window_size=window_sizes[0], use_overlap=use_overlap[0]
        )
        self.fuse1 = RGBDViTBlock(
            in_ch_r=C[1], in_ch_t=C[1], shape=S[1],
            embed_dim=embed_stage[1], num_heads=heads_stage[1],
            patch_size=patch_size[1], dropout=dropout, attn_dropout=dropout,
            window_size=window_sizes[1], use_overlap=use_overlap[1]
        )
        self.fuse2 = RGBDViTBlock(
            in_ch_r=C[2], in_ch_t=C[2], shape=S[2],
            embed_dim=embed_stage[2], num_heads=heads_stage[2],
            patch_size=patch_size[2], dropout=dropout, attn_dropout=dropout,
            window_size=window_sizes[2], use_overlap=use_overlap[2]
        )
        self.fuse3_1 = RGBDViTBlock(
            in_ch_r=C[3], in_ch_t=C[3], shape=S[3],
            embed_dim=embed_stage[3], num_heads=heads_stage[3],
            patch_size=patch_size[3], dropout=dropout, attn_dropout=dropout,
            window_size=window_sizes[3], use_overlap=use_overlap[3]
        )
        self.fuse4_1 = RGBDViTBlock(
            in_ch_r=C[4], in_ch_t=C[4], shape=S[4],
            embed_dim=embed_stage[4], num_heads=heads_stage[4],
            patch_size=patch_size[4], dropout=dropout, attn_dropout=dropout,
            window_size=window_sizes[4], use_overlap=use_overlap[4]
        )
        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        # stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        # ---- layer0 fusion (64ch) ----
        x = self.fuse0(x, x_depth)

        # layer1
        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)

        # ---- layer1 fusion (256ch) ----
        x1 = self.fuse1(x1, x1_depth)

        # layer2
        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        # ---- layer2 fusion (512ch) ----
        x2 = self.fuse2(x2, x2_depth)

        # layer3_1
        x3_1 = self.resnet.layer3_1(x2)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        # ---- layer3_1 fusion (1024ch) ----
        x3_1 = self.fuse3_1(x3_1, x3_1_depth)

        # layer4_1
        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        # ---- layer4_1 fusion (2048ch) ----
        x4_1 = self.fuse4_1(x4_1, x4_1_depth)

        # produce initial saliency map by decoder1
        x2_1 = self.rfb2_1(x2)
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
