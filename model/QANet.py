import torch
import torch.nn as nn
import torch.nn.functional as F

from .InputEmbedding import InputEmbeddingLayer
from .EmbeddingEncoder import EncoderEmbeddingLayer
from .ContextQueryAttention import ContextQueryAttention
from .ModelEncoder import ModelEncoder
from .Output import Output

class QANet(nn.Module):

    def __init__(self, word_embed: torch.Tensor, char_embed: torch.Tensor, config: dict):
        super(QANet, self).__init__()

        self.input_embedding_layer = InputEmbeddingLayer(word_embed, char_embed, config)

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

        M0 = self.model_encoder_layer(CtQ_out, c_mask, first_pass=True)
        M1 = self.model_encoder_layer(M0, c_mask)
        M2 = self.model_encoder_layer(M1, c_mask)

        # We have to mask these logits before computing the CEloss/softmax
        p1, p2 = self.output_layer(M0, M1, M2)

        p1 = torch.masked_fill(p1, torch.logical_not(c_mask), -1e-8) 
        p2 = torch.masked_fill(p2, torch.logical_not(c_mask), -1e-8)

        return p1, p2
