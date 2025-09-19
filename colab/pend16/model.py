import torch.nn.functional as F
from typing import List
from timm.models.layers import trunc_normal_
import math
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from models.ResNet import ResNet50
from torch.nn import functional as F
from typing import Tuple
from models.swin_cross import SwinTransformer


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

# ==================================================================


def get_2d_sincos_pos_embed(h, w, dim, device):
    """Create 2D sinusoidal positional embeddings (H*W, dim)."""
    assert dim % 2 == 0, "PE dim must be even"
    y_embed = torch.arange(h, device=device).float().unsqueeze(1).repeat(1, w)
    x_embed = torch.arange(w, device=device).float().unsqueeze(0).repeat(h, 1)

    div_term = torch.exp(torch.arange(0, dim // 2, 2, device=device).float()
                         * (-torch.log(torch.tensor(10000.0, device=device)) / (dim // 2)))

    pe_y = torch.zeros(h, w, dim // 2, device=device)
    pe_x = torch.zeros(h, w, dim // 2, device=device)

    pe_y[..., 0::2] = torch.sin(y_embed.unsqueeze(-1) * div_term)
    pe_y[..., 1::2] = torch.cos(y_embed.unsqueeze(-1) * div_term)

    pe_x[..., 0::2] = torch.sin(x_embed.unsqueeze(-1) * div_term)
    pe_x[..., 1::2] = torch.cos(x_embed.unsqueeze(-1) * div_term)

    pe = torch.cat([pe_y, pe_x], dim=-1)  # (H, W, dim)
    pe = pe.view(h * w, dim)
    return pe


class CoarseCrossAttention(nn.Module):
    """
    Coarse (patchified) cross-attention block.
    - Patchify both modalities with pooling to suppress noise / misalignment
    - Apply global MultiheadAttention at the coarse scale
    - (Optional) confidence gating derived from the KV modality
    - Upsample back to the original spatial size and fuse residually

    Forward(x_q, x_kv):
        x_q:  [B, C, H, W]  (query stream to be modulated)
        x_kv: [B, C, H, W]  (key/value stream providing guidance)
    Returns:
        out: [B, C, H, W]
    """

    def __init__(self, embed_dim, attn_dim=None, num_heads=8, patch=4, dropout=0.1,
                 use_confidence=True, pooling="avg"):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim or embed_dim
        self.num_heads = num_heads
        self.patch = patch
        self.use_confidence = use_confidence

        if pooling == "avg":
            self.pool = nn.AvgPool2d(
                kernel_size=patch, stride=patch, ceil_mode=False)
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=patch, stride=patch)
        else:
            # strided conv can also be used for learnable patchify
            self.pool = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=patch, stride=patch, groups=embed_dim, bias=False)

        # projections for Q, K, V
        self.q_proj = nn.Conv2d(embed_dim, self.attn_dim, 1)
        self.k_proj = nn.Conv2d(embed_dim, self.attn_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, self.attn_dim, 1)

        self.attn = nn.MultiheadAttention(
            self.attn_dim, num_heads, dropout=dropout, batch_first=False)
        self.attn_norm = nn.LayerNorm(self.attn_dim)

        # FFN at coarse scale
        self.ffn = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.attn_dim * 4, self.attn_dim),
            nn.Dropout(dropout),
        )

        # project back to feature dim
        self.out_proj = nn.Conv2d(self.attn_dim, embed_dim, 1)
        # self.out_norm = nn.BatchNorm2d(embed_dim)

        if self.use_confidence:
            # produce a confidence mask from KV stream at coarse scale
            self.conf_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, 1, 1)
            )

    def forward(self, x_q, x_kv):
        B, C, H, W = x_q.shape

        # 1) Patchify & project to attn_dim
        q = self.pool(x_q)
        kv = self.pool(x_kv)
        Hc, Wc = q.shape[-2], q.shape[-1]

        q = self.q_proj(q)   # [B, D, Hc, Wc]
        k = self.k_proj(kv)  # [B, D, Hc, Wc]
        v = self.v_proj(kv)  # [B, D, Hc, Wc]

        # 2) Flatten to sequences (L, B, D)
        q_seq = q.flatten(2).permute(2, 0, 1)
        k_seq = k.flatten(2).permute(2, 0, 1)
        v_seq = v.flatten(2).permute(2, 0, 1)

        # 3) Add fixed 2D sin-cos positional encodings (helps alignment)
        pe = get_2d_sincos_pos_embed(Hc, Wc, q.shape[1], q.device)  # (L, D)
        pe = pe.unsqueeze(1).expand(-1, B, -1)  # (L, B, D)
        q_seq = q_seq + pe
        k_seq = k_seq + pe

        # 4) Global MHA at coarse scale
        attn_out, _ = self.attn(q_seq, k_seq, v_seq)  # (L, B, D)
        attn_out = self.attn_norm(attn_out)
        attn_out = attn_out + self.ffn(attn_out)

        # 5) Back to (B, D, Hc, Wc)
        attn_out = attn_out.permute(1, 2, 0).contiguous().view(B, -1, Hc, Wc)

        # 6) Optional confidence gating from KV stream
        if self.use_confidence:
            conf = torch.sigmoid(self.conf_head(kv))  # [B,1,Hc,Wc]
            attn_out = attn_out * (1.0 + conf)

        # 7) Project to embed_dim and upsample + residual
        attn_out = self.out_proj(attn_out)  # [B, C, Hc, Wc]
        attn_out = F.interpolate(attn_out, size=(
            H, W), mode='bilinear', align_corners=False)
        # out = self.out_norm(attn_out)
        return attn_out


class Fusion(nn.Module):
    def __init__(self, in_ch, embed_dim, shape, swin_depth,
                 num_heads, window_size, mlp_ratio=4,
                 attn_dropout=0.2, dropout=0.1, patch_size=4,
                 cross_patch=8, cross_heads=None, cross_attn_dim=None,
                 cross_conf=False, cross_pool="conv"):
        super(Fusion, self).__init__()

        # Channel reduction
        self.rgb_reduce = BasicConv2d(in_ch, embed_dim, 1)

        # Self-attention for each modality (keep your original Swin branches)
        self.rgb_self = SwinTransformer(
            img_size=(shape, shape), patch_size=patch_size,
            in_chans=in_ch, embed_dim=embed_dim,
            depths=[swin_depth], num_heads=[num_heads],
            window_size=window_size, mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_dropout
        )
        self.dep_self = SwinTransformer(
            img_size=(shape, shape), patch_size=patch_size,
            in_chans=in_ch, embed_dim=embed_dim,
            depths=[swin_depth], num_heads=[num_heads],
            window_size=window_size, mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_dropout
        )

        # Coarse (patchified) cross attention modules
        # self.rgb_to_dep = CoarseCrossAttention(
        #     embed_dim=embed_dim,
        #     attn_dim=cross_attn_dim or embed_dim,
        #     num_heads=cross_heads or num_heads,
        #     patch=cross_patch,
        #     dropout=dropout,
        #     use_confidence=cross_conf,
        #     pooling=cross_pool,
        # )
        # self.dep_to_rgb = CoarseCrossAttention(
        #     embed_dim=embed_dim,
        #     attn_dim=cross_attn_dim or embed_dim,
        #     num_heads=cross_heads or num_heads,
        #     patch=cross_patch,
        #     dropout=dropout,
        #     use_confidence=cross_conf,
        #     pooling=cross_pool,
        # )

        # # Learnable fusion weights
        # self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.beta = nn.Parameter(torch.tensor(0.5))

        # Post-fusion feedforward network
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, embed_dim, 1),
            nn.Dropout(dropout)
        )

        self.out_norm = nn.BatchNorm2d(embed_dim)

    def forward(self, Ri, Ti):
        B, C, H, W = Ri.shape

        # 1) Self-attention
        rgb_self = self.rgb_self(Ri)
        dep_self = self.dep_self(Ti)

        # 2) Coarse bi-directional cross-attention
        #    (global at coarse scale, robust to noise via patchify + confidence)
        # rgb_cross = self.dep_to_rgb(rgb_self, dep_self)  # depth guides RGB
        # dep_cross = self.rgb_to_dep(dep_self, rgb_self)  # RGB guides depth

        # 3) Adaptive fusion over {rgb_self, dep_self, rgb_cross, dep_cross}
        fused_stack = torch.stack(
            [rgb_self, dep_self], dim=1)  # [B,4,C,h,w]
        context = fused_stack.mean(dim=[3, 4])          # [B,4,C]
        weights = torch.softmax(context.mean(dim=2), 1)  # [B,4]
        fused = (fused_stack * weights[:, :, None,
                 None, None]).sum(dim=1)  # [B,C,h,w]

        # 4) Post-fusion refinement (residual)
        fused = self.mlp(fused) + fused
        fused = self.out_norm(fused)

        # 5) Upsample to input size
        fused = F.interpolate(fused, size=(
            H, W), mode='bilinear', align_corners=False)

        # 6) Low-level RGB residual
        Ri_low = self.rgb_reduce(Ri)  # [B,C,H,W]
        return fused + Ri_low


# ==================================================================


class CPA(nn.Module):
    def __init__(self, dim):
        super(CPA, self).__init__()

        self.channel_attn = ChannelAttention(dim)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = x.mul(self.channel_attn(x))
        x = x.mul(self.spatial_attn(x))
        return x


class FeatureRefiner(nn.Module):
    """Refine multi-scale RGB & Depth features with channel reduction + saliency + channel attention.
       Outputs reduced channel features (no restore)."""

    def __init__(self, channels_list, reduction=4):
        super(FeatureRefiner, self).__init__()
        self.reduction = reduction

        self.rgb_reduce = nn.ModuleList([
            nn.Conv2d(c, c // reduction, 1, bias=False) for c in channels_list
        ])
        self.depth_reduce = nn.ModuleList([
            nn.Conv2d(c, c // reduction, 1, bias=False) for c in channels_list
        ])

        self.attentions = nn.ModuleList([
            ChannelAttention(c // reduction) for c in channels_list
        ])

    def forward(self, saliency_map, rgb_feats, depth_feats):
        refined_rgb, refined_depth = [], []

        for rgb, depth, rr, dr, ca in zip(
            rgb_feats, depth_feats,
            self.rgb_reduce, self.depth_reduce,
            self.attentions
        ):
            B, C, H, W = rgb.shape

            # Reduce channels
            rgb_red = rr(rgb)
            depth_red = dr(depth)

            # Resize saliency to match feature map
            saliency_resized = F.interpolate(
                saliency_map, size=(H, W),
                mode='bilinear', align_corners=False
            )
            saliency_mask = torch.sigmoid(saliency_resized)

            # Apply saliency mask
            rgb_red = rgb_red * (1 + saliency_mask)
            depth_red = depth_red * (1 + saliency_mask)

            # Channel attention
            rgb_red = rgb_red * ca(rgb_red)
            depth_red = depth_red * ca(depth_red)

            refined_rgb.append(rgb_red)
            refined_depth.append(depth_red)

        return refined_rgb, refined_depth


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

        embed_dim = [64, 128, 128, 128, 128]
        heads_stage = (4, 4, 8, 8, 8)
        C = [64, 256, 512, 1024, 2048]           # channel dims from ResNet
        S = [88, 88, 44, 22, 11]

        # you can also set per stage (smaller patch for early layers)
        P = [8, 8, 4, 2, 1]
        W = [11, 11, 11, 11, 11]

        d_ch = 4

        # Replace FusionBlock2D with RGBDViTBlock
        self.fuse0 = Fusion(
            in_ch=C[0]//d_ch, embed_dim=embed_dim[0], shape=S[0], swin_depth=1,
            num_heads=heads_stage[0], window_size=W[0], patch_size=P[0], cross_patch=1
        )
        self.fuse1 = Fusion(
            in_ch=C[1]//d_ch, embed_dim=embed_dim[1], shape=S[1], swin_depth=1,
            num_heads=heads_stage[1], window_size=W[1], patch_size=P[1], cross_patch=1
        )
        self.fuse2 = Fusion(
            in_ch=C[2]//d_ch, embed_dim=embed_dim[2], shape=S[2], swin_depth=1,
            num_heads=heads_stage[2], window_size=W[2], patch_size=P[2], cross_patch=1
        )
        self.fuse3_1 = Fusion(
            in_ch=C[3]//d_ch, embed_dim=embed_dim[3], shape=S[3], swin_depth=1,
            num_heads=heads_stage[3], window_size=W[3], patch_size=P[3], cross_patch=1
        )
        self.fuse4_1 = Fusion(
            in_ch=C[4]//d_ch, embed_dim=embed_dim[4], shape=S[4], swin_depth=1,
            num_heads=heads_stage[4], window_size=W[4], patch_size=P[4], cross_patch=1
        )

        channel = 32
        self.rfb2_1 = GCM(embed_dim[2], channel)
        self.rfb3_1 = GCM(embed_dim[3], channel)
        self.rfb4_1 = GCM(embed_dim[4], channel)

        # Decoder 2
        self.rfb0_2 = GCM(embed_dim[0], channel)
        self.rfb1_2 = GCM(embed_dim[1], channel)
        self.rfb5_2 = GCM(embed_dim[2], channel)

        # ========================================
        self.rfb2_0 = GCM(C[2], channel)
        self.rfb3_0 = GCM(C[3], channel)
        self.rfb4_0 = GCM(C[4], channel)

        self.upsample0 = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)

        self.agg0 = aggregation_init(channel)

        self.refiner = FeatureRefiner(C)

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

        # layer1
        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)

        # layer2
        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        # layer3_1
        x3_1 = self.resnet.layer3_1(x2)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        # layer4_1
        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        # =======================================================

        # produce initial saliency map by decoder1
        x2_1_ = self.rfb2_0(x2)
        x3_1_ = self.rfb3_0(x3_1)
        x4_1_ = self.rfb4_0(x4_1)

        attention_map0 = self.agg0(x4_1_, x3_1_, x2_1_)

        rgb, dep = self.refiner(attention_map0, [x, x1, x2, x3_1, x4_1], [
            x_depth, x1_depth, x2_depth, x3_1_depth, x4_1_depth])

        # =======================================================

        x = self.fuse0(rgb[0], dep[0])
        x1 = self.fuse1(rgb[1], dep[1])
        x2 = self.fuse2(rgb[2], dep[2])
        x3_1 = self.fuse3_1(rgb[3], dep[3])
        x4_1 = self.fuse4_1(rgb[4], dep[4])

        # =======================================================

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

        return self.upsample(attention_map), y, self.upsample0(attention_map0)
