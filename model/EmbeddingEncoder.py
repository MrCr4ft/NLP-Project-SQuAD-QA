import torch
import torch.nn as nn
import torch.nn.functional as F

from .PositionalEncoder import PositionalEncoder

class Reshape1Dconv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1d = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            # This must be one otherwise the logic is different
            kernel_size = 1
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)

        return x


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        self.in_channels = config['resized_emb_dim']
        self.out_channels = self.in_channels
        
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

        x = x.permute(0, 2, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.permute(0, 2, 1)

        return x


class EncoderBlock(nn.Module):

    # TODO: insert positional encoder in the fw pass -> if model we should not compute again pos encoding(?)

    def __init__(self, config: dict, model: bool = False):
        super().__init__()

        self.emb_dim = config['resized_emb_dim']

        if model != True:
            self.n_convs = config['enc_n_convs']
        else:
            self.n_convs = config['model_n_convs']

        self.num_heads = config['self_att_num_heads']

        self.conv = nn.ModuleList([DepthwiseSeparableConv(config) for _ in range(self.n_convs)])

        self.conv_layer_norm = nn.ModuleList([nn.LayerNorm(self.emb_dim) for _ in range(self.n_convs)])
        self.layer_norm1 = nn.LayerNorm(self.emb_dim)
        self.layer_norm2 = nn.LayerNorm(self.emb_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim = self.emb_dim,
            num_heads = self.num_heads,
            # dropout ?
            batch_first = True
        )

        self.linear = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, x, attn_mask):

        # Convolutional steps
        for dw_convs, layer_norm in zip(self.conv, self.conv_layer_norm):
            temp = layer_norm(x)
            temp = F.relu(dw_convs(temp))
            x = temp + x

        # Self-attention
        temp = self.layer_norm1(x)
        # The embedding we want to project as query, key and value is the same
        attn_mask = self.get_attn_mask(mask=attn_mask)
        temp, _ = self.self_attention(
            query = temp,
            key = temp,
            value = temp,
            attn_mask = attn_mask
        )
        x = temp + x

        # Linear layer
        temp = self.layer_norm2(x)
        temp = self.linear(x)
        x = temp + x

        return x

    def get_attn_mask(self, mask):

        mask = torch.bmm(mask.long().unsqueeze(2), mask.long().unsqueeze(1))
        mask = torch.logical_not(mask.repeat_interleave(self.num_heads, dim=0))
        return mask


class EncoderEmbeddingLayer(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        self.in_channels = config['word_embed_dim'] + config['char_embed_out_dim']
        self.out_channels = config['resized_emb_dim']
        self.encoder_n_blocks = config['encoder_n_blocks']

        self.conv1d = Reshape1Dconv(self.in_channels, self.out_channels)

        self.c_seq_len = config['context_seq_len']
        self.q_seq_len = config['query_seq_len']

        self.c_pos_encoder = PositionalEncoder(self.out_channels, self.c_seq_len)
        self.q_pos_encoder = PositionalEncoder(self.out_channels, self.q_seq_len)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(self.encoder_n_blocks)])

    def forward(self, context, query, C_attn_mask, Q_attn_mask):

        # 1D conv to reduce the embedding size
        context = self.conv1d(context, query)
        query = self.conv1d(query)

        # Positional encoding
        context = self.c_pos_encoder(context)
        query = self.q_pos_encoder(query)
        
        # Encoder embedding blocks pass
        for block in self.encoder_blocks:
            context = block(context, C_attn_mask)
            query = block(query, Q_attn_mask)

        return context, query
