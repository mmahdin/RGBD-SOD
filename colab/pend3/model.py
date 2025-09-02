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


class Fusion(nn.Module):
    def __init__(self, in_ch, embed_dim, shape, swin_depth,
                 num_heads, window_size, mlp_ratio=4,
                 attn_dropout=0.1, dropout=0.1, patch_size=4):
        super(Fusion, self).__init__()

        # Channel reduction
        self.rgb_reduce = BasicConv2d(in_ch, embed_dim, 1)
        self.dep_reduce = BasicConv2d(in_ch, embed_dim, 1)

        # Self-attention for each modality
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

        # Bi-directional cross-attention
        self.rgb_to_dep = SwinTransformer(
            img_size=(shape//patch_size, shape//patch_size), patch_size=1,
            in_chans=embed_dim, embed_dim=embed_dim,
            depths=[1], num_heads=[num_heads],
            window_size=11, mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_dropout, cross_attention=True
        )
        self.dep_to_rgb = SwinTransformer(
            img_size=(shape//patch_size, shape//patch_size), patch_size=1,
            in_chans=embed_dim, embed_dim=embed_dim,
            depths=[1], num_heads=[num_heads],
            window_size=11, mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_dropout, cross_attention=True
        )

        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # Post-fusion feedforward network
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, embed_dim, 1),
            nn.Dropout(dropout)
        )

        # self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, Ri, Ti):

        # Self-attention
        rgb_self = self.rgb_self(Ri)
        dep_self = self.dep_self(Ti)

        # Bi-directional cross-attention
        rgb_cross = self.rgb_to_dep(rgb_self, dep_self)
        dep_cross = self.dep_to_rgb(dep_self, rgb_self)

        # Weighted fusion
        fused = rgb_self + dep_self + rgb_cross + dep_cross

        # Refinement
        # fused = self.norm(fused)
        fused = self.mlp(fused)  # residual

        fused = F.interpolate(fused, size=Ri.shape[-2:], mode='bilinear',
                              align_corners=False)

        # Reduce channels
        Ri = self.rgb_reduce(Ri)

        return fused + Ri

# ==================================================================


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=4):  # <-- add strong reduction
        super(ChannelWeights, self).__init__()
        self.dim = dim
        hidden_dim = dim // reduction  # bottleneck dimension

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # bottleneck MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, hidden_dim),   # reduce first
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.dim * 2),   # then expand back
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)                # B, 2C, H, W
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)              # B, 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(
            1, 0, 2, 3, 4)                            # 2, B, C, 1, 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=4):  # <-- also reduce here
        super(SpatialWeights, self).__init__()
        self.dim = dim
        hidden_dim = dim // reduction

        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, hidden_dim, kernel_size=1),   # reduce
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),              # expand back
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)   # B, 2C, H, W
        spatial_weights = self.mlp(x).reshape(
            B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2, B, 1, H, W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * \
            channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * \
            channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class CPA(nn.Module):
    def __init__(self, dim):
        super(CPA, self).__init__()

        self.channel_attn = ChannelAttention(dim)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = x.mul(self.channel_attn(x))
        x = x.mul(self.spatial_attn(x))
        return x

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
        P = [4, 4, 2, 1, 1]
        W = [11, 11, 11, 11, 11]

        # Replace FusionBlock2D with RGBDViTBlock
        self.fuse0 = Fusion(
            in_ch=C[0], embed_dim=embed_dim[0], shape=S[0], swin_depth=2,
            num_heads=heads_stage[0], window_size=W[0], patch_size=P[0]
        )
        self.fuse1 = Fusion(
            in_ch=C[1], embed_dim=embed_dim[1], shape=S[1], swin_depth=2,
            num_heads=heads_stage[1], window_size=W[1], patch_size=P[1]
        )
        self.fuse2 = Fusion(
            in_ch=C[2], embed_dim=embed_dim[2], shape=S[2], swin_depth=2,
            num_heads=heads_stage[2], window_size=W[2], patch_size=P[2]
        )
        self.fuse3_1 = Fusion(
            in_ch=C[3], embed_dim=embed_dim[3], shape=S[3], swin_depth=2,
            num_heads=heads_stage[3], window_size=W[3], patch_size=P[3]
        )
        self.fuse4_1 = Fusion(
            in_ch=C[4], embed_dim=embed_dim[4], shape=S[4], swin_depth=1,
            num_heads=heads_stage[4], window_size=W[4], patch_size=P[4]
        )

        channel = 32
        self.rfb2_1 = GCM(embed_dim[2], channel)
        self.rfb3_1 = GCM(embed_dim[3], channel)
        self.rfb4_1 = GCM(embed_dim[4], channel)

        # Decoder 2
        self.rfb0_2 = GCM(embed_dim[0], channel)
        self.rfb1_2 = GCM(embed_dim[1], channel)
        self.rfb5_2 = GCM(embed_dim[2], channel)

        self.rectify0 = FeatureRectifyModule(C[0])
        self.rectify1 = FeatureRectifyModule(C[1])
        self.rectify2 = FeatureRectifyModule(C[2])
        self.rectify3 = FeatureRectifyModule(C[3])
        self.rectify4 = FeatureRectifyModule(C[4])

        # self.cpa0 = CPA(C[0])
        # self.cpa1 = CPA(C[1])
        # self.cpa2 = CPA(C[2])
        # self.cpa3 = CPA(C[3])
        # self.cpa4 = CPA(C[4])

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

        x, x_depth = self.rectify0(x, x_depth)
        # x = x + self.cpa0(x_depth)

        # layer1
        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)
        x1, x1_depth = self.rectify1(x1, x1_depth)
        # x1 = x1 + self.cpa1(x1_depth)

        # layer2
        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)
        x2, x2_depth = self.rectify2(x2, x2_depth)
        # x2 = x2 + self.cpa2(x2_depth)

        # layer3_1
        x3_1 = self.resnet.layer3_1(x2)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)
        x3_1, x3_1_depth = self.rectify3(x3_1, x3_1_depth)
        # x3_1 = x3_1 + self.cpa3(x3_1_depth)

        # layer4_1
        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)
        x4_1, x4_1_depth = self.rectify4(x4_1, x4_1_depth)
        # x4_1 = x4_1 + self.cpa4(x4_1_depth)

        # =======================================================

        x = self.fuse0(x, x_depth)
        x1 = self.fuse1(x1, x1_depth)
        x2 = self.fuse2(x2, x2_depth)
        x3_1 = self.fuse3_1(x3_1, x3_1_depth)
        x4_1 = self.fuse4_1(x4_1, x4_1_depth)

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

        return self.upsample(attention_map), y
