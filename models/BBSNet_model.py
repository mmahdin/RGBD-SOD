from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torchvision.models as models
from models.ResNet import ResNet50
from torch.nn import functional as F
import math


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


class TFFBlock(nn.Module):
    def __init__(self, dim):
        super(TFFBlock, self).__init__()
        # RGB projections
        self.q_rr = nn.Linear(dim, dim)
        self.q_rt = nn.Linear(dim, dim)
        self.k_r  = nn.Linear(dim, dim)
        self.v_r  = nn.Linear(dim, dim)
        
        # Depth projections
        self.q_tt = nn.Linear(dim, dim)
        self.q_tr = nn.Linear(dim, dim)
        self.k_t  = nn.Linear(dim, dim)
        self.v_t  = nn.Linear(dim, dim)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim)
        )
        
        self.scale = math.sqrt(dim)

    def _attention(self, Q, K, V):
        """
        Computes scaled dot-product attention:
        Softmax(QK^T / sqrt(d)) V
        Q, K, V: [B, N, C]
        Returns: [B, N, C]
        """
        attn = torch.softmax((Q @ K.transpose(-2, -1)) / self.scale, dim=-1)
        return attn @ V

    def forward(self, R, T):
        """
        R, T: [B, C, H, W]
        """
        B, C, H, W = R.shape
        N = H * W
        
        # Flatten spatial dims
        R_flat = R.permute(0, 2, 3, 1).reshape(B, N, C)
        T_flat = T.permute(0, 2, 3, 1).reshape(B, N, C)
        
        # Project RGB
        Q_rr = self.q_rr(R_flat)
        Q_rt = self.q_rt(R_flat)
        K_r  = self.k_r(R_flat)
        V_r  = self.v_r(R_flat)
        
        # Project Depth
        Q_tt = self.q_tt(T_flat)
        Q_tr = self.q_tr(T_flat)
        K_t  = self.k_t(T_flat)
        V_t  = self.v_t(T_flat)
        
        # Self-attention
        A_rr = self._attention(Q_rr, K_r, V_r)
        A_tt = self._attention(Q_tt, K_t, V_t)
        SA = A_rr + A_tt
        
        # Cross-attention
        A_rt = self._attention(Q_rt, K_t, V_t)
        A_tr = self._attention(Q_tr, K_r, V_r)
        CA = A_rt + A_tr
        
        # Eq. (3): O_i
        O = SA + CA + R_flat
        
        # Eq. (4): M_i
        M = self.mlp(O)
        
        # Eq. (5): F_i
        F = R_flat + M
        
        # Reshape back to [B, C, H, W]
        return F.reshape(B, H, W, C).permute(0, 3, 1, 2)



# =========================
# Attention Blocks
# =========================
class SelfAttn2d(nn.Module):
    def __init__(self, in_channels, num_heads=4, max_proj=256, dropout=0.0):
        super().__init__()
        d_model = min(in_channels, max_proj)
        self.proj_in = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=True)
        self.proj_out = nn.Conv2d(d_model, in_channels, kernel_size=1, bias=True)

        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.pos = PositionalEncoding2D(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_ffn = nn.Parameter(torch.zeros(1))

        self.mlp_spatial = ConvMLP(in_channels, hidden_ratio=2.0)  # optional local mixing

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        tokens, (H, W) = _to_tokens(x, self.proj_in, self.pos)   # (B, L, D)
        # Self Attention
        y = self.norm1(tokens)
        y, _ = self.mha(y, y, y, need_weights=False)
        tokens = tokens + self.gamma_attn * y
        # FFN
        y2 = self.norm2(tokens)
        y2 = self.ffn(y2)
        tokens = tokens + self.gamma_ffn * y2

        out = _from_tokens(tokens, H, W, self.proj_out)  # (B, C, H, W)
        out = self.mlp_spatial(out)                      # extra local mixing
        return out


class CrossAttn2d(nn.Module):
    def __init__(self, in_channels, num_heads=4, max_proj=256, dropout=0.0):
        super().__init__()
        d_model = min(in_channels, max_proj)
        self.q_proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=True)
        self.kv_proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(d_model, in_channels, kernel_size=1, bias=True)

        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.pos_q = PositionalEncoding2D(d_model)
        self.pos_kv = PositionalEncoding2D(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, query_x, context_x):
        """
        query_x attends to context_x.
        Inputs: (B, C, H, W)
        Returns updated query_x with same shape.
        """
        B, C, H, W = query_x.shape

        q_tokens, (H, W) = _to_tokens(query_x, self.q_proj, self.pos_q)      # (B, Lq, D)
        kv_tokens, _ = _to_tokens(context_x, self.kv_proj, self.pos_kv)      # (B, Lk, D)

        qn = self.norm_q(q_tokens)
        kvn = self.norm_kv(kv_tokens)

        y, _ = self.mha(qn, kvn, kvn, need_weights=False)
        q_tokens = q_tokens + self.gamma_attn * y

        y2 = self.norm_ffn(q_tokens)
        y2 = self.ffn(y2)
        q_tokens = q_tokens + self.gamma_ffn * y2

        out = _from_tokens(q_tokens, H, W, self.out_proj)  # (B, C, H, W)
        return out


class FusionBlock2D(nn.Module):
    """
    Self-attention for each modality + cross-attention both ways.
    Returns updated (rgb, depth).
    """
    def __init__(self, channels, num_heads=4, max_proj=256, dropout=0.0):
        super().__init__()
        self.self_rgb = SelfAttn2d(channels, num_heads=num_heads, max_proj=max_proj, dropout=dropout)
        self.self_dep = SelfAttn2d(channels, num_heads=num_heads, max_proj=max_proj, dropout=dropout)

        self.cross_rgb_from_dep = CrossAttn2d(channels, num_heads=num_heads, max_proj=max_proj, dropout=dropout)
        self.cross_dep_from_rgb = CrossAttn2d(channels, num_heads=num_heads, max_proj=max_proj, dropout=dropout)

        # Gated residual merge
        self.gate_rgb = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.gate_dep = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_dep):
        # 1) self-attention refinement
        r = self.self_rgb(x_rgb)
        d = self.self_dep(x_dep)
        # 2) cross attention (bi-directional)
        r_cross = self.cross_rgb_from_dep(r, d)
        d_cross = self.cross_dep_from_rgb(d, r)
        # 3) gated residual blend
        g_r = self.gate_rgb(torch.cat([r, r_cross], dim=1))
        g_d = self.gate_dep(torch.cat([d, d_cross], dim=1))
        out_r = r + g_r * r_cross
        out_d = d + g_d * d_cross
        return out_r, out_d


# =========================
# Positional Encoding (2D)
# =========================
class PositionalEncoding2D(nn.Module):
    """
    Sine-cosine 2D positional encoding that maps (H, W) into d_model.
    Adapted to work with channels-last token embeddings in attention.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # keep a cache by size to avoid recomputing
        self.register_buffer("_cached", torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) where L = H*W and D = d_model
        We assume tokens were created by flattening H*W with row-major order.
        """
        B, L, D = x.shape
        d_model = D
        # Infer H and W heuristically: prefer square-ish (assumes caller uses consistent H, W)
        H = W = int(math.sqrt(L))
        if H * W != L:
            # fallback: assume the caller provided (B, C, H, W) before flatten and tracked H, W
            # If you want explicit H,W passing, add it to the module signature.
            H = L
            W = 1

        device = x.device
        # Cache key
        key = (H, W, d_model)
        if (not isinstance(self._cached, torch.Tensor)) or self._cached.numel() == 1:
            self._cache = {}
            self._cached = torch.zeros(2, device=device)  # just a non-empty tensor

        if key in getattr(self, "_cache", {}):
            pe = self._cache[key]
            if pe.device != device:
                pe = pe.to(device)
                self._cache[key] = pe
        else:
            pe = self._build_pe(H, W, d_model, device)  # (H*W, D)
            if not hasattr(self, "_cache"):
                self._cache = {}
            self._cache[key] = pe

        pe = pe.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        return x + pe

    @staticmethod
    def _build_pe(H: int, W: int, d_model: int, device) -> torch.Tensor:
        """
        Classic 2D sine-cosine split: half dims for H, half for W (even d_model).
        """
        if d_model % 2 != 0:
            # make even by dropping last dim if needed
            d_model = d_model - 1
        half = d_model // 2
        # positions
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H, W)

        # frequencies
        div_term_y = torch.exp(torch.arange(0, half, 2, device=device).float() * (-math.log(10000.0) / half))
        div_term_x = torch.exp(torch.arange(0, half, 2, device=device).float() * (-math.log(10000.0) / half))

        pe_y = torch.zeros(H, W, half, device=device)
        pe_x = torch.zeros(H, W, half, device=device)

        pe_y[..., 0::2] = torch.sin(yy[..., None] * div_term_y)
        pe_y[..., 1::2] = torch.cos(yy[..., None] * div_term_y)

        pe_x[..., 0::2] = torch.sin(xx[..., None] * div_term_x)
        pe_x[..., 1::2] = torch.cos(xx[..., None] * div_term_x)

        pe = torch.cat([pe_y, pe_x], dim=-1)  # (H, W, D)
        if pe.shape[-1] + 1 == d_model + 1:  # if we dropped a dim
            pass
        pe = pe.view(H * W, -1)  # (L, D)
        # If original D was odd, pad one zero
        if pe.shape[-1] % 2 == 1:
            pe = F.pad(pe, (0,1))
        return pe


# =========================
# Helpers
# =========================
class ConvMLP(nn.Module):
    """Lightweight 1x1-Conv MLP with GELU and residual gate."""
    def __init__(self, channels, hidden_ratio=2.0):
        super().__init__()
        hidden = max(4, int(channels * hidden_ratio))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))  # residual scale

    def forward(self, x):
        return x + self.gamma * self.net(x)


def _to_tokens(x, proj, posenc: PositionalEncoding2D):
    """
    x: (B, C, H, W) -> tokens: (B, L, D)
    """
    B, C, H, W = x.shape
    y = proj(x)                        # (B, D, H, W)
    y = y.flatten(2).transpose(1, 2)   # (B, L, D)
    y = posenc(y)
    return y, (H, W)


def _from_tokens(tokens, H, W, out_proj):
    """
    tokens: (B, L, D) -> (B, C, H, W)
    """
    B, L, D = tokens.shape
    y = tokens.transpose(1, 2).view(B, D, H, W)  # (B, D, H, W)
    y = out_proj(y)                              # (B, C, H, W)
    return y




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

        # TFF modules per stage
        dropout = 0.1

        heads_stage=(4, 4, 8, 8, 8)
        maxproj_stage=(128, 192, 256, 256, 256)
        C = [64, 256, 512, 1024, 2048]

        # TFF modules per stage
        self.fuse0   = FusionBlock2D(C[0], num_heads=heads_stage[0], max_proj=maxproj_stage[0], dropout=dropout)
        self.fuse1   = FusionBlock2D(C[1], num_heads=heads_stage[1], max_proj=maxproj_stage[1], dropout=dropout)
        self.fuse2   = FusionBlock2D(C[2], num_heads=heads_stage[2], max_proj=maxproj_stage[2], dropout=dropout)
        self.fuse3_1 = FusionBlock2D(C[3], num_heads=heads_stage[3], max_proj=maxproj_stage[3], dropout=dropout)
        self.fuse4_1 = FusionBlock2D(C[4], num_heads=heads_stage[4], max_proj=maxproj_stage[4], dropout=dropout)

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
        x, x_depth = self.fuse0(x, x_depth)

        # layer1
        x1 = self.resnet.layer1(x)              # (B, 256, H/4, W/4) typical
        x1_depth = self.resnet_depth.layer1(x_depth)

        # ---- layer1 fusion (256ch) ----
        x1, x1_depth = self.fuse1(x1, x1_depth)

        # layer2
        x2 = self.resnet.layer2(x1)             # (B, 512, H/8, W/8)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        # ---- layer2 fusion (512ch) ----
        x2, x2_depth = self.fuse2(x2, x2_depth)

        # layer3_1
        x3_1 = self.resnet.layer3_1(x2)         # (B, 1024, H/16, W/16) depending on your impl
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        # ---- layer3_1 fusion (1024ch) ----
        x3_1, x3_1_depth = self.fuse3_1(x3_1, x3_1_depth)

        # layer4_1
        x4_1 = self.resnet.layer4_1(x3_1)       # (B, 2048, H/32, W/32)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        # ---- layer4_1 fusion (2048ch) ----
        x4_1, x4_1_depth = self.fuse4_1(x4_1, x4_1_depth)

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



