import torch
import torch.nn as nn
import torch.functional as F 



class CharacterEmbedding(nn.Module):
    
    def __init__(self, num_embed: int, embed_dim: int,
                 output_dim: int, kernel_size: int):

        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.kernel_size = (1, kernel_size)

        self.char_embeddings = nn.Embedding(
            num_embeddings = self.num_embed,
            embedding_dim = self.embed_dim,
        )

        self.conv1 = nn.Conv2d(
            in_channels = self.embed_dim,
            out_channels = self.output_dim,
            kernel_size = self.kernel_size
        )
        self.conv2 = nn.Conv2d(
            in_channels = self.output_dim,
            out_channels = self.output_dim,
            kernel_size = self.kernel_size
        )

    def forward(self, x):

        # ===========================================================================
        # ! assuming that the input is in the form (batch, #words_in_doc, word_len) !
        # ===========================================================================

        x = self.char_embeddings(x) # out shape (batch, word_in_doc, word_len, emb_dim)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x, _ = torch.max(x, 3)
        x = x.permute(0, 2, 1)  # out shape (batch, word_in_doc, embed_dim)

        # x = F.dropout(x, p=0.05)
        return x


class Highway(nn.Module):

    def __init__(self, input_size, n_layers):
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

        for i in range(self.n_layers):

            h = F.relu(self.h_gate[i](x))
            t = torch.sigmoid(self.t_gate[i](x))
            x = (h * t) + (x * (1 - t))

        return x


class InputEmbeddingLayer(nn.Module):

    def __init__(self, word_embed: torch.Tensor, config: dict):

        super().__init__()

        self.num_embed = config['char_vocab_size']
        self.embed_dim = config['char_emb_in_dim']
        self.output_dim = config['char_embed_out_dim']
        self.kernel_size = config['char_kernel_size']

        self.input_size = self.embed_dim + word_embed.size()[1]
        self.n_layers = config['highway_n_layers']

        self.word_embed = nn.Embedding.from_pretrained(word_embed)

        self.char_embed = CharacterEmbedding(
            num_embed = self.num_embed,
            embed_dim = self.embed_dim,
            output_dim = self.output_dim,
            kernel_size = self.kernel_size
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