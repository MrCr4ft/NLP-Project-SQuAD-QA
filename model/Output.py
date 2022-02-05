import torch
import torch.nn as nn
import torch.nn.functional as F

class Output(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.emb_dim = config['resized_emb_dim']

        self.w1 = nn.Linear(2*self.emb_dim, 1)
        self.w2 = nn.Linear(2*self.emb_dim, 1)

    def forward(self, m0, m1, m2):

        cat1 = torch.cat((m0, m1), dim=-1)
        cat2 = torch.cat((m0, m2), dim=-1)

        # We do not apply the softmax because it's done in the CrossEntropyLoss function

        p1 = self.w1(cat1)
        p2 = self.w2(cat2)

        return p1, p2
