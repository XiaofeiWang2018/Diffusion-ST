from abc import abstractmethod

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

import copy
import torch.nn.functional as F

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SE_Attention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            gene_num,
            model_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            root=''
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.gene_num = gene_num
        # self.in_channels = in_channels
        self.model_channels = model_channels
        # self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        co_expression = np.load(root + 'gene_coexpre.npy')
        self.pre_graph = nn.Sequential(
            conv_nd(dims, self.gene_num, self.gene_num, 3, padding=1),
            nn.SiLU()
        )
        self.gc1 = GraphConvolution(26*26, 26*26,co_expression,self.gene_num)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.gene_num, ch, 3, padding=1))]
        )

        self.input_blocks_WSI5120 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )

        self.input_blocks_WSI320 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )


        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_WSI5120.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                self.input_blocks_WSI5120.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = self.channel_mult[-1] * self.model_channels

        self.input_blocks_lr = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_WSI320 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks_WSI5120])



        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 2 * conv_ch, conv_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.fc_modulation_1 = nn.Sequential(
            nn.Linear(1024, 1024),
        )
        self.fc_modulation_2 = nn.Sequential(
            nn.Linear(1024, 1024),
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, self.gene_num*2, 3, padding=1)),
        )

        self.to_q = nn.Linear(model_channels, model_channels, bias=False)
        self.to_k = nn.Linear(model_channels, model_channels, bias=False)
        self.to_v = nn.Linear(model_channels, model_channels, bias=False)

        self.to_q_con = nn.Linear(model_channels, model_channels, bias=False)
        self.to_k_con = nn.Linear(int(model_channels*1.5), model_channels, bias=False)
        self.to_v_con = nn.Linear(int(model_channels*1.5), model_channels, bias=False)



    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps,low_res, WSI_5120,WSI_320):
        """
        Apply the model to an input batch.
        :param x: an [N x 50 x 256 x 256] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param ratio: 0-1
        :param low_res: [N x 50 x 26 x 26] round -13
        :param WSI_5120: [N x 3 x 256 256] 0-255
        :param WSI_320: [N x 256 x 3 x 16 16] 0-255
        :return: an [N x 50 x 256 x 256] Tensor of outputs.
        """


        # ratio = x[0, 0, 0, 0]
        # x = x[:, int(x.shape[1] / 2):x.shape[1], ...]  # [N x 50 x 256 x 256]
        ratio=1


        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        WSI_5120=WSI_5120/255
        WSI_320=th.reshape(WSI_320,(-1,WSI_320.shape[2],WSI_320.shape[3],WSI_320.shape[4]))/255 #[N.256 x 3 x 16 x 16]


        h_x = x.type(self.dtype)  ## backward noise of SR ST   [N x 50 x 256 x 256]  round -4
        h_spot = low_res.type(self.dtype)## spot ST            [N x 50 x 26 x 26] round -13
        h_5120WSI = WSI_5120.type(self.dtype)## #              [N x 3 x 256 x 256]
        h_320WSI = WSI_320.type(self.dtype)##  #              [N.256.ratio x 3 x 16 x 16]

        #GRAPH
        h_spot_ori=h_spot
        h_spot=self.pre_graph(h_spot)
        h_spot = F.relu(self.gc1(h_spot)) ##[N x 50 x 26 x 26]
        h_spot=h_spot_ori*0.98+h_spot*0.02

        for idx in range(len(self.input_blocks)):
            h_x = self.input_blocks[idx](h_x, emb) # [N x 16 x 64 x 64]
            h_spot = self.input_blocks_lr[idx](h_spot, emb) # [N x 16 x 6 x 6]
            h_5120WSI = self.input_blocks_WSI5120[idx](h_5120WSI, emb) # [N x 16 x 64 x 64]
            hs.append((1 / 3) * h_x + (1 / 3) * F.interpolate(h_spot,(h_x.shape[2],h_x.shape[3])) + (1 / 3) * h_5120WSI)
        for idx in range(len(self.input_blocks_WSI320)):
            h_320WSI = self.input_blocks_WSI320[idx](h_320WSI, emb)

        #########
        ######### entropy based cross attention
        #########
        h_320WSI = th.reshape(h_320WSI, (h_x.shape[0], -1, h_320WSI.shape[1], h_320WSI.shape[2], h_320WSI.shape[3]))
        h_320WSI = h_320WSI[:, 0:int(h_320WSI.shape[1] * ratio), ...]  # [N x 256.ratio x 16 x 16 x 16]
        h_320WSI = th.mean(h_320WSI, dim=1) # [N x 16 x 16 x 16]
        h_320WSI = F.interpolate(h_320WSI, size=(h_5120WSI.shape[2], h_5120WSI.shape[3])) # [N x 16 x 64 x 64]
        h_320WSI=th.reshape(h_320WSI,(h_320WSI.shape[0],h_320WSI.shape[1],-1))
        h_320WSI=th.transpose(h_320WSI,1,2) # [N x 4096 x 16]

        h_5120WSI_pre=th.reshape(h_5120WSI,(h_5120WSI.shape[0],h_5120WSI.shape[1],-1))
        h_5120WSI_pre = th.transpose(h_5120WSI_pre,1,2)  # [N x 4096 x 16]

        q = self.to_q(h_5120WSI_pre) # [N x 4096 x 16]
        k = self.to_k(h_320WSI) # [N x 4096 x 16]
        v = self.to_v(h_320WSI) # [N x 4096 x 16]
        mid_atten=torch.matmul(q,th.transpose(k,1,2))

        scale = q.shape[2] ** -0.5
        mid_atten=mid_atten*scale
        sfmax = nn.Softmax(dim=-1)
        mid_atten=sfmax(mid_atten)# [N x 4096 x 4096]
        WSI_atten = torch.matmul(mid_atten, v) # [N x 4096 x 16]
        WSI_atten=th.transpose(WSI_atten,1,2)# [N x 16 x 4096 ]
        WSI_atten = th.reshape(WSI_atten, (WSI_atten.shape[0],WSI_atten.shape[1], h_5120WSI.shape[2], h_5120WSI.shape[3]))# [N x 16 x 64 x64 ]

        ### weight
        WSI_atten=0.9*h_5120WSI+0.1*WSI_atten

        #########
        ######### Disentangle and modulation and cross atten
        #########
        com_WSI = self.conv_common(WSI_atten) # [N x 8 x 64 x 64]
        com_spot = self.conv_common(h_spot)
        com_spot =F.interpolate(com_spot, size=(WSI_atten.shape[2], WSI_atten.shape[3])) # [N x 8 x 64 x 64]

        dist_WSI = self.conv_distinct(WSI_atten) # [N x 8 x 64 x 64]
        dist_spot = self.conv_distinct(h_spot)
        dist_spot = F.interpolate(dist_spot, size=(WSI_atten.shape[2], WSI_atten.shape[3]))  # [N x 8 x 64 x 64]

        com_h = (1 / 2) * com_WSI + (1 / 2) * com_spot # [N x 8 x 64 x 64]

        ##modulatoion
        part=2
        part_width=int(dist_WSI.shape[2]/part)
        WSI_part_dist=dist_WSI
        spot_part_dist = dist_spot
        for i in range(part):
            for j in range(part):
                WSI_part=dist_WSI[...,i*part_width:(i+1)*part_width,j*part_width:(j+1)*part_width] # [N x 8 x 32 x 32]
                spot_part = dist_spot[..., i * part_width:(i + 1) * part_width, j * part_width:(j + 1) * part_width] # [N x 8 x 32 x 32]
                WSI_part = th.reshape(WSI_part, (WSI_part.shape[0], WSI_part.shape[1], -1)) # [N x 8 x 1024]
                spot_part = th.reshape(spot_part, (spot_part.shape[0], spot_part.shape[1], -1))  # [N x 8 x 1024]
                WSI_part_T = th.transpose(WSI_part, 1, 2)  # [N x 1024 x 8 ]
                spot_part_T = th.transpose(spot_part, 1, 2)  # [N x 1024 x 8 ]

                F_WSItoSpot=th.matmul(spot_part_T,WSI_part)# [N x 1024 x 1024]
                w_WSItoSpot=self.fc_modulation_1(F_WSItoSpot)# [N x 1024 x 1024]
                sfmax_module = nn.Softmax(dim=-1)
                w_WSItoSpot=sfmax_module(w_WSItoSpot)# [N x 1024 x 1024]
                spot_part_out = th.matmul(spot_part, w_WSItoSpot)  # [N x 8 x 1024]
                spot_part_out = th.reshape(spot_part_out, (spot_part_out.shape[0],spot_part_out.shape[1],
                                                           int(math.sqrt(spot_part_out.shape[2])), int(math.sqrt(spot_part_out.shape[2]))))  # [N x 8 x 32 x 32]
                spot_part_dist[...,i*part_width:(i+1)*part_width,j*part_width:(j+1)*part_width]=spot_part_out

                F_SpottoWSI = th.matmul(WSI_part_T, spot_part)  # [N x 1024 x 1024]
                w_SpottoWSI = self.fc_modulation_2(F_SpottoWSI)  # [N x 1024 x 1024]
                sfmax_module = nn.Softmax(dim=-1)
                w_SpottoWSI = sfmax_module(w_SpottoWSI)  # [N x 1024 x 1024]
                WSI_part_out = th.matmul(WSI_part, w_SpottoWSI)  # [N x 8 x 1024]
                WSI_part_out = th.reshape(WSI_part_out, (WSI_part_out.shape[0], WSI_part_out.shape[1], int(math.sqrt(WSI_part_out.shape[2])),
                                                           int(math.sqrt(WSI_part_out.shape[2]))))  # [N x 8 x 32 x 32]
                WSI_part_dist[..., i * part_width:(i + 1) * part_width,j * part_width:(j + 1) * part_width] = WSI_part_out
        ### weight
        WSI_part_dist = 0.9*dist_WSI+0.1*WSI_part_dist
        spot_part_dist =  0.9*dist_spot+0.1*spot_part_dist

        h_condition = th.cat([com_h, WSI_part_dist,spot_part_dist], dim=1) # [N x 24 x 64 x 64]
        #########  cross attention for embedding condition
        h_condition_pre = th.reshape(h_condition, (h_condition.shape[0], h_condition.shape[1], -1))
        h_condition_pre = th.transpose(h_condition_pre, 1, 2)  # [N x 4096 x 24]

        h_x_pre = th.reshape(h_x, (h_x.shape[0], h_x.shape[1], -1))
        h_x_pre = th.transpose(h_x_pre, 1, 2)  # [N x 4096 x 16]

        q = self.to_q_con(h_x_pre)  # [N x 4096 x 16]
        k = self.to_k_con(h_condition_pre)  # [N x 4096 x 16]
        v = self.to_v_con(h_condition_pre)  # [N x 4096 x 16]
        mid_atten = torch.matmul(q, th.transpose(k, 1, 2))

        scale = q.shape[2] ** -0.5
        mid_atten = mid_atten * scale
        sfmax = nn.Softmax(dim=-1)
        mid_atten = sfmax(mid_atten)  # [N x 4096 x 4096]
        Final_merge = torch.matmul(mid_atten, v)  # [N x 4096 x 16]
        Final_merge = th.transpose(Final_merge, 1, 2)  # [N x 16 x 4096 ]
        Final_merge = th.reshape(Final_merge, (Final_merge.shape[0], Final_merge.shape[1], h_x.shape[2], h_x.shape[3]))  # [N x 16 x 64 x64 ]

        h = self.middle_block(Final_merge, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        # h = h.type(x.dtype)
        # print(h.shape)
        a=1

        return com_WSI, com_spot,  dist_WSI, dist_spot,self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, gene_num, model_channels, *args, **kwargs):
        super().__init__(gene_num, model_channels, *args, **kwargs)
        # print(image_size)
    def forward(self, x, timesteps,timesteps0,**kwargs):
        # _, _, new_height, new_width = x.shape
        # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(x, timesteps,timesteps0, kwargs['low_res'], kwargs['WSI_5120'], kwargs['WSI_320'])


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            pool="adaptive",
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, co_expre,gene_num,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj = co_expre[0:gene_num,0:gene_num]
        self.adj_1 = self.adj + np.multiply(self.adj.T, self.adj.T > self.adj) - np.multiply(self.adj,self.adj.T > self.adj)
        self.adj_1 = Parameter(torch.from_numpy(np.array(self.adj_1)).float(),requires_grad=False)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
       input #[BS,50,26,26 ]
       adj #[3,3 ]
        """
        inpu_ori=input
        input=torch.reshape(input,(input.shape[0],input.shape[1],-1)) #[BS,50,26*26 ]

        support = torch.matmul(input, self.weight)#[BS,50,26*26 ]

        support= torch.transpose(support, 1, 2)  # [BS,26*26,50 ]
        output = torch.matmul( support,self.adj_1  )# [BS,26*26,50 ]
        output= torch.transpose(output, 1, 2) # #[BS,50,26*26 ]
        if self.bias is not None:
            output = torch.reshape(output, (inpu_ori.shape[0], inpu_ori.shape[1], inpu_ori.shape[2],inpu_ori.shape[3]))  # [BS,50,26,26 ]
            # print(output.shape)
            # output = output + self.bias
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'