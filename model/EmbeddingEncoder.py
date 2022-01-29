from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F

# TODO:
# - positional encoder
# - depthwise separable conv
# - layer norm
# - create 4 residual block with dw conv
#
# - self attention ...
# - fc layer ...


class DepthwiseSeparableConv(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.in_channels = config['resized_emb_dim']
        self.out_channels = config['enc_conv_out_channels']
        self.kernel_size = config['enc_conv_kernel_size']
        self.padding = config['enc_conv_pad_size']

        self.depthwise = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.in_channels,
            kernel_size = self.kernel_size,
            padding = self.padding,
            groups = self.in_channels
        )

        self.pointwise = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = 1,
        )

    def forward(self, x):
        
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class EncoderEmbeddingLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.in_channels = config['word_embed_dim'] + config['char_embed_out_dim']
        self.out_channels = config['resized_emb_dim']

        self.conv1d = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            # This must be one otherwise the logic is different
            kernel_size=1
        )


    def forward(self, context, query):

        # 1D conv to reduce the embedding size
        context = context.permute(0, 2, 1)
        context = self.conv1d(context)
        context = context.permute(0, 2, 1)

        query = query.permute(0, 2, 1)
        query = self.conv1d(query)
        query = query.permute(0, 2, 1)

        # roba
        return 
