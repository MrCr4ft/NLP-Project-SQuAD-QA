import torch
import numpy as np
from torch.autograd import Variable

# Positional encoding allows to inject some information about order of the tokens in a sequence
# The function it computes is:
#       PE(pos, 2i) = sin(pos/10000^(2i/embedding_dim)),
#       PE(pos, 2i+1) = cos(pos/10000^(2i/embedding_dim))
# where i iterates from 0 to max_sequence_len, and i iterates from 0 to ceil(embedding_dim/2))
#
# Here for numerical stability reasons it's computed as it follows:
# PE(pos, 2i) = sin(exp(log(pos/10000^(2i/embedding_dim)))) = sin(exp(log(pos) - 2i/embedding_dim * log(10000))) =
# sin(pos * exp(-2i/embedding_dim * log(10000))
#
# The positional encoding obtained is concatenated to the embedding vector
# Additionally the output is passed through a dropout layer

# Since the positional encodings can be computed once for all, and are not a parameter of the layer,
# can be persistently registered


class PositionalEncoder(torch.nn.Module):
    def __init__(self, embedding_dim: int, max_sequence_len: int, dropout_prob: float = 0.0):
        self.embedding_dim = embedding_dim
        self.max_sequence_len = max_sequence_len
        self.dropout = torch.nn.Dropout(p=dropout_prob)

        self.register_buffer('pos_encodings', self._get_positional_encodings())

        super(PositionalEncoder, self).__init__()

    def _get_positional_encodings(self):
        positions = torch.arange(0, self.max_sequence_len, 1)
        dimensions = torch.arange(0, self.embedding_dim, 2)
        frequencies = positions * torch.exp(-1 * (dimensions / self.embedding_dim) * np.log(10000))

        pos_encoding_table = torch.zeros(self.max_sequence_len, self.embedding_dim)
        pos_encoding_table[:, 0::2] = torch.sin(positions * frequencies)
        pos_encoding_table[:, 1::2] = torch.cos(positions * frequencies)
        pos_encoding_table = pos_encoding_table.unsqueeze(0)

        return pos_encoding_table

    def forward(self, x: torch.Tensor):
        # This operation is shape preserving
        return self.dropout(x + Variable(self.pos_encodings[:, :x.size(1)], requires_grad=False))
