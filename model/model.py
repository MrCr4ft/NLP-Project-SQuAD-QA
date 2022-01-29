from gc import freeze
from sre_constants import NOT_LITERAL_IGNORE
from turtle import forward
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F

from .InputEmbedding import InputEmbeddingLayer



class QANet(nn.Module):

    def __init__(self, word_embed: torch.Tensor, config: dict):
        super().__init__()

        self.input_embedding_layer = InputEmbeddingLayer(word_embed, config)

    def forward(self, Cw, Cc, Qw, Qc):


        return 

"""

config = {

  # INPUT EMBEDDING LAYER

  'char_vocab_size':
  # Initial char embedding dim
  'char_emb_in_dim':
  # Char convolution output dim
  'char_emb_out_dim':
  # Kernel size for char convolution
  'char_kernel_size':
  'highway_n_layers':


  # ENCODER EMBEDDING LAYER

  'word_emb_dim':
  # dim of word+char emb at the beginning of emb encoder layer after resize
  'resized_emd_dim': 128,
  # conv block output channels
  'enc_conv_out_channels': 128,

  N.B. The following two params are chosen in a way that
       the number of words doesn't change during the conv
  # kernel size encoder conv
  'enc_conv_kernel_size': 7
  # padding size encoder conv
  'enc_conv_pad_size': 3

  
}

"""


# N, word_in_doc, embed_dim

x = torch.rand((5, 10, 50))
x = x.permute(0,2,1)

conv = nn.Conv1d(
    in_channels=50,
    out_channels=20,
    kernel_size=1
)

print(conv(x).permute(0, 2, 1).shape)