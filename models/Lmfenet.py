from operator import concat
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn import init
from math import gcd
from einops.layers.torch import Rearrange


class RLN(nn.Module):

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        # 使用全部特征通道进行1x1卷积
        self.conv_1x1 = nn.Conv2d(in_features, out_features, kernel_size=1)

        self.relu = nn.ReLU(True)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 对输入数据进行1x1卷积处理
        out = self.conv_1x1(x)
        return self.relu(out)

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out

class CSLM(nn.Module):
    def __init__(self, channels, factor=8):
        super(CSLM, self).__init__()
        # 设置分组数量，用于特征分组
        self.groups = factor
        # 确保分组后的通道数大于0
        assert channels // self.groups > 0
        # softmax激活函数，用于归一化
        self.softmax = nn.Softmax(-1)
        # 全局平均池化，生成通道描述符
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 水平方向的平均池化，用于编码水平方向的全局信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向的平均池化，用于编码垂直方向的全局信息
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # GroupNorm归一化，减少内部协变量偏移
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 1x1卷积，用于学习跨通道的特征
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # 3x3卷积，用于捕捉更丰富的空间信息
        self.conv3_1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU(inplace=False)

    def forward(self, x):
        b, c, h, w = x.size()
        # 对输入特征图进行分组处理
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # 应用水平和垂直方向的全局平均池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # 通过1x1卷积和sigmoid激活函数，获得注意力权重
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # 应用GroupNorm和注意力权重调整特征图
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3_1(group_x)
        x2 = self.silu(x2)
        x2 = self.conv3_2(x2)
        # 将特征图通过全局平均池化和softmax进行处理，得到权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # 通过矩阵乘法和sigmoid激活获得最终的注意力权重，调整特征图
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # 将调整后的特征图重塑回原始尺寸
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.att = CSLM(dim)
        # 添加 GSAM 模块
        self.GSAM = GSAM(channel=dim)

    def forward(self, x):
        identity = x
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn:
            x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        x1 = self.att(x)
        x = x1+x
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x

        # 在 TransformerBlock 之后应用 GSAM
        x = self.GSAM(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth
        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class MASM(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[1, 3, 5], reduction=16):
        super(MASM, self).__init__()
        self.atrous_blocks = nn.ModuleList()
        self.reduction = reduction
        # 根据不同的膨胀率构建空洞卷积
        for rate in atrous_rates:
            self.atrous_blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                                padding=rate, dilation=rate, bias=False))

        # 1x1卷积层将多尺度特征整合，减少通道维度
        self.conv1x1 = nn.Conv2d(in_channels * len(atrous_rates), out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 全局平均池化层+全连接层，用于生成通道注意力机制
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 多尺度空洞卷积特征提取
        atrous_features = [atrous_block(x) for atrous_block in self.atrous_blocks]

        # 拼接多尺度特征
        x = torch.cat(atrous_features, dim=1)

        # 1x1卷积整合特征
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)

        # 通道注意力机制
        attention = self.global_avg_pool(x)
        attention = self.fc1(attention)
        attention = F.relu(attention, inplace=True)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)

        # 对输入特征进行加权
        x = x * attention

        return x


class ECAFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(ECAFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        # 使用ECA模块替代MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 先通过一个1x1卷积调整通道数，再用ECA进行通道加权
        self.eca = ECAAttention(kernel_size=3)  # ECA模块
        self.conv1x1 = nn.Conv2d(dim, dim * height, kernel_size=1, bias=False)  # 增加通道数

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        # 拼接输入的特征
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        # 对拼接后的特征进行求和
        feats_sum = torch.sum(in_feats, dim=1)

        # 使用ECA进行通道加权
        attn = self.eca(self.avg_pool(feats_sum))  # ECA计算通道权重

        # 通过1x1卷积增加通道数
        attn = self.conv1x1(attn)

        # 应用Softmax进行归一化
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        # 通过加权特征图进行融合
        out = torch.sum(in_feats * attn, dim=1)
        return out

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法


class LMFE_Net(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[4, 4, 4, 2, 2],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[0, 1 / 2, 1, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(LMFE_Net, self).__init__()
        # Basic settings
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        # Divide input image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.MASM = MASM(in_channels=24, out_channels=24)
        self.HAM = HAM(in_channels=24, out_channels=24,  kernel_size=7)
        # Main network
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = ECAFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = ECAFusion(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        # New convolution layer to align channels of y to x
        self.match_channels_y = nn.Conv2d(embed_dims[2], embed_dims[4], kernel_size=1)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def upsample_to_match(self, x, ref):
        # Upsample tensor x to match the spatial size of tensor ref
        return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.HAM(x)
        x = self.MASM(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x



class GSAM(nn.Module):
    # 初始化ParNet注意力模块
    def __init__(self, channel=512, reduction_ratio=4):
        super().__init__()
        reduced_channel = max(channel // reduction_ratio, 32)  # 降低通道数进行参数压缩

        # 确保分组卷积的组数可以整除通道数
        groups = gcd(channel, 4)  # 假设我们希望最多分16组，gcd是求最大公约数的函数

        # 使用自适应平均池化和1x1卷积实现空间压缩
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将空间维度压缩到1x1
            nn.Conv2d(channel, reduced_channel, kernel_size=1),  # 减少通道数
            nn.ReLU(True),  # 使用ReLU以保持非线性
            nn.Conv2d(reduced_channel, channel, kernel_size=1),  # 恢复通道数
            nn.Sigmoid()  # Sigmoid函数，用于生成注意力图
        )

        # 通过1x1分组卷积实现特征重映射
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, groups=groups),  # 使用分组卷积减少参数
            nn.BatchNorm2d(channel)
        )

        # 通过3x3分组卷积捕获空间上下文信息
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=groups),  # 分组卷积
            nn.BatchNorm2d(channel)
        )

        self.silu = nn.SiLU()  # SiLU激活函数，也被称为Swish函数

    def forward(self, x):
        # x是输入的特征图，形状为(Batch, Channel, Height, Width)
        x1 = self.conv1x1(x)  # 通过1x1卷积处理x
        x2 = self.conv3x3(x)  # 通过3x3卷积处理x
        x3 = self.sse(x) * x  # 应用空间压缩的注意力权重到x上
        y = self.silu(x1 + x2 + x3)  # 将上述三个结果相加并通过SiLU激活函数激活，获得最终输出
        return y

class MFEA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFEA, self).__init__()
        
        # 多尺度卷积分支
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn3x3 = nn.BatchNorm2d(in_channels)
        self.relu3x3 = nn.ReLU(inplace=True)
        
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.bn5x5 = nn.BatchNorm2d(in_channels)
        self.relu5x5 = nn.ReLU(inplace=True)
        
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.bn7x7 = nn.BatchNorm2d(in_channels)
        self.relu7x7 = nn.ReLU(inplace=True)
        
        # 1x1卷积融合特征
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 全局平均池化 + 通道注意力
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.reweight = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 多尺度卷积
        x3 = self.relu3x3(self.bn3x3(self.conv3x3(x)))
        x5 = self.relu5x5(self.bn5x5(self.conv5x5(x)))
        x7 = self.relu7x7(self.bn7x7(self.conv7x7(x)))
        
        # 特征融合 - 按元素求和
        fused = x3 + x5 + x7
        
        # 1x1卷积调整通道
        fused = self.conv1x1(fused)
        
        # 全局平均池化 + 通道注意力
        x_pooled = self.gap(fused)
        x_pooled = x_pooled.view(x_pooled.size(0), -1)  # [N, C]
        x_reweight = self.reweight(x_pooled)
        x_reweight = self.softmax(x_reweight)
        
        # 将通道注意力权重应用到输入特征
        x_reweight = x_reweight.unsqueeze(2).unsqueeze(3)  # [N, C, 1, 1]
        out = fused * x_reweight.expand_as(fused)
        
        return out

# ChannelAttention Module
class ChannelAttention(nn.Module):
    def __init__(self, input_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, input_channels // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(input_channels // ratio, input_channels, 1, bias=False)
        self.input_channels = input_channels
        self.sigmoid =  nn.Sigmoid()

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = self.global_median_pooling(inputs)

        avg_out = self.fc1(avg_pool)
        avg_out = F.relu(avg_out, inplace=True)
        avg_out = self.fc2(avg_out)


        max_out = self.fc1(max_pool)
        max_out = F.relu(max_out, inplace=True)
        max_out = self.fc2(max_out)
   

        median_out = self.fc1(median_pool)
        median_out = F.relu(median_out, inplace=True)
        median_out = self.fc2(median_out)
        out = avg_out + max_out + median_out
        out = self.sigmoid(out)
        return out


    @staticmethod
    def global_median_pooling(x):
        median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
        median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
        return median_pooled


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # --- 1. 平均池化 ---
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # --- 2. 最大池化 ---
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # --- 3. 中值池化（正确且简洁的方式）---
        median_out = torch.median(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)

        # --- 融合 ---
        combined = torch.cat([avg_out, max_out, median_out], dim=1)  # (B, 3, H, W)

        attention_map = self.conv1(combined)
        attention_map = self.sigmoid(attention_map)
        out = x * attention_map
        return out


# HAM Module
class HAM(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=7):
        super(HAM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        assert in_channels == out_channels, "Input and output channels must be the same"

   
        self.ca = ChannelAttention(input_channels=in_channels, ratio=16)   
        self.MFEA = MFEA(in_channels=in_channels,out_channels=out_channels)
        self.sa = SpatialAttention(kernel_size)
        # 定义 1x1 卷积层和激活函数
  


    def forward(self, inputs):
        # 全局感知机
        x1 = self.MFEA(inputs)
        x2 = self.ca(inputs)
        x3 = x1+x2
        x4 = x3*self.sa(x3)
        return x4



def lmfe_net():
    return LMFE_Net(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1/2, 1, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])