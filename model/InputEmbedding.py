import torch
import torch.nn as nn
import torch.nn.functional as F 



class CharacterEmbedding(nn.Module):
    
    def __init__(self,
                 char_embed: torch.Tensor,
                 output_dim: int, kernel_size: int,
                 n_convs: int):

        super().__init__()

        self.char_embed = char_embed
        self.embed_dim = self.char_embed.size()[1]
        self.output_dim = output_dim
        self.kernel_size = (1, kernel_size)
        self.n_convs = n_convs

        self.char_embeddings = nn.Embedding.from_pretrained(
            embeddings = self.char_embed,
            freeze = False
        )

        self.conv1 = nn.Conv2d(
            in_channels = self.embed_dim,
            out_channels = self.output_dim,
            kernel_size = self.kernel_size
        )

        if self.n_convs > 1:
            self.convs = nn.ModuleList(
                [nn.Conv2d(
                    in_channels = self.output_dim,
                    out_channels = self.output_dim,
                    kernel_size = self.kernel_size
                ) for _ in range(self.n_convs - 1)]
            )

    def forward(self, x):

        # ===========================================================================
        # ! assuming that the input is in the form (batch, #words_in_doc, word_len) !
        # ===========================================================================

        x = self.char_embeddings(x) # out shape (batch, word_in_doc, word_len, emb_dim)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        
        if self.n_convs > 1:
            for convs in self.convs:
                x = F.relu(convs(x))

        x, _ = torch.max(x, 3)
        x = x.permute(0, 2, 1)  # out shape (batch, word_in_doc, embed_dim)
        # x = F.dropout(x, p=0.05)
        
        return x


class Highway(nn.Module):

    def __init__(self, input_size: int, n_layers: int):
        super().__init__()

        # Input and output sizes must be the same
        self.in_size = input_size
        self.out_size = input_size

        self.n_layers = n_layers

        self.t_gate = nn.ModuleList([nn.Linear(self.in_size, self.out_size) for _ in range(n_layers)])
        self.h_gate = nn.ModuleList([nn.Linear(self.in_size, self.out_size) for _ in range(n_layers)])
        
        for i in range(self.n_layers):
            self.t_gate[i].bias.data.fill_(-1.0)

    def forward(self, x):
        
        for h_gate, t_gate in zip(self.h_gates, self.t_gates):

            h = F.relu(h_gate(x))
            t = torch.sigmoid(t_gate(x))
            x = (h * t) + (x * (1 - t))

        return x


class InputEmbeddingLayer(nn.Module):

    def __init__(self, word_embed: torch.Tensor,
                 char_embed: torch.Tensor, config: dict):

        super().__init__()

        self.word_embed = word_embed
        self.char_embed = char_embed

        self.output_dim = config['char_embed_out_dim']
        self.kernel_size = config['char_kernel_size']
        self.n_convs = config['char_n_convs']

        self.input_size = self.char_embed.size()[1] + word_embed.size()[1]
        self.n_layers = config['highway_n_layers']

        self.word_embed = nn.Embedding.from_pretrained(
            embeddings = word_embed,
            freeze = True
        )

        self.char_embed = CharacterEmbedding(
            char_embed = char_embed,
            output_dim = self.output_dim,
            kernel_size = self.kernel_size,
            n_convs = self.n_convs
        )

        self.highway = Highway(input_size=self.input_size, n_layers=self.n_layers)

    def forward(self, Cw, Cc, Qw, Qc):
        
        #   Cw: context words, will be something like (batch, word_per_doc)     containing indeces of words
        #   Cc: context chars,                        (batch, word_per_doc, chars_per_word)           chars
        #   Qw: 
        #   Qc: 

        Cw_emb = self.word_embed(Cw)
        Qw_emb = self.word_embed(Qw)
        
        Cc_emb = self.char_embed(Cc)
        Qc_emb = self.char_embed(Qc) # now they have shape (batch, word_per_doc, char_embed_dim)
        
        context = torch.concat((Cw_emb, Cc_emb), dim=-1)
        query = torch.concat((Qw_emb, Qc_emb), dim=-1)

        context = self.highway(context)
        query = self.highway(query)

        return context, query
