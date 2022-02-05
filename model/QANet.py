from locale import MON_1
import torch
import torch.nn as nn
import torch.nn.functional as F

from .InputEmbedding import InputEmbeddingLayer
from .EmbeddingEncoder import EncoderEmbeddingLayer
from .ContextQueryAttention import ContextQueryAttention
from .ModelEncoder import ModelEncoder
from .Output import Output

class QANet(nn.Module):

    def __init__(self, word_embed: torch.Tensor, config: dict):
        super().__init__()

        self.input_embedding_layer = InputEmbeddingLayer(word_embed, config)

        self.embedding_encoder_layer = EncoderEmbeddingLayer(config)

        self.cq_attention = ContextQueryAttention(config)

        self.model_encoder_layer = ModelEncoder(config)

        self.output_layer = Output(config)


    def forward(self, Cw, Cc, Qw, Qc):

        c_mask = (torch.zeros_like(Cw) != Cw)   # False where padding
        q_mask = (torch.zeros_like(Qw) != Qw)

        # Returns concatenated words and chars embeddings
        # out dims (batch, #context_word, p1+p2 emb_dim)
        context_emb, query_emb = self.input_embedding_layer(
            Cw, Cc, Qw, Qc
        )

        # out dims (batch, #context_word, resized embed)
        context, query = self.embedding_encoder_layer(
            context = context_emb,
            query = query_emb,
            C_attn_mask = c_mask,
            Q_attn_mask = q_mask
        )

        # out dims (batch, #context_word, 4*emb_dim)
        CtQ_out = self.cq_attention(
            context = context,
            query = query,
            context_mask = c_mask,
            query_mask = q_mask
        )

        M0 = self.model_encoder_layer(CtQ_out)
        M1 = self.model_encoder_layer(M0)
        M2 = self.model_encoder_layer(M1)

        # We have to mask these logits before computing the CEloss/softmax
        p1, p2 = Output(M0, M1, M2)

        p1 = torch.masked_fill(p1, torch.logical_not(c_mask), 1e-8) 
        p2 = torch.masked_fill(p2, torch.logical_not(q_mask), 1e-8)

        return p1, p2



config = {


    # context sequence length
    'context_seq_len': 400,
    # query sequence length
    'query_seq_len': 60,


    # # INPUT EMBEDDING LAYER

    # # Char convolution output dim
    'char_embed_out_dim': 100,
    # # Kernel size for char convolution
    # 'char_kernel_size':
    # # Number of convs to perform on chars
    # 'char_n_convs': 1
    # 'highway_n_layers':



    # # ENCODER EMBEDDING LAYER

    'word_embed_dim': 100,
    # dim of word+char emb at the beginning of emb encoder layer after resize
    'resized_emb_dim': 128,

    # N.B. The following two params are chosen in a way that
    #     the number of words doesn't change during the conv
    # kernel size encoder conv
    'enc_conv_kernel_size': 7,
    # padding size encoder conv
    'enc_conv_pad_size': 3,
    # number of conv layers in one encoder block
    'enc_n_convs': 4,

    'self_att_num_heads': 8,
    # TODO: rename to '
    'encoder_n_blocks': 1,

    # MODEL ENCODER LAYER

    # conv number in a model block 
    'model_n_conv': 2,
    # blocks of encoder in a model encoder layer
    'model_n_blocks': 7,
}
