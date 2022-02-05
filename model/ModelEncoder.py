import torch
import torch.nn as nn
import torch.nn.functional as F

from .EmbeddingEncoder import Reshape1Dconv, EncoderBlock


class ModelEncoder(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        self.n_blocks = config['model_n_blocks']

        self.emb_dim = config['resized_emb_dim']

        self.reshape_conv = Reshape1Dconv(4*self.emb_dim, self.emb_dim)

        self.model_blocks = nn.ModuleList(
            [EncoderBlock(config, model = True) for _ in range(self.n_blocks)]
        )

    def forward(self, x, mask, first_pass: bool = False):
        
        """
            The input x must have dimension (batch_size, #word_per_doc, 4*emb_dim)
        """

        # Reduce the dim of the embedding back to emb_dim
        if first_pass:
            x = self.reshape_conv(x)

        # Model encoder pass
        # 'mask' should be the context mask with the same size, I'm not 100% sure about this
        for block in self.model_blocks:
            x = block(x, mask)
        
        return x
        