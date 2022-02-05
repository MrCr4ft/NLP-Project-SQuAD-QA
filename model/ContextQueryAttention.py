import typing

import torch
import torch.nn.functional as F


class ContextQueryAttention(torch.nn.Module):
    def __init__(self, config: typing.Dict, dropout_prob: float = 0.05):
        """

        :param config: The configuration dictionary for the whole model
        :param dropout_prob: The dropout rate
        """
        self.context_weight = torch.nn.Parameter(torch.zeros(config['resized_emb_dim'], 1))
        torch.nn.init.xavier_uniform_(self.context_weight)

        self.query_weight = torch.nn.Parameter(torch.zeros(config['resized_emb_dim'], 1))
        torch.nn.init.xavier_uniform_(self.query_weight)

        self.context_query_weight = torch.nn.Parameter(torch.zeros(1, 1, config['resized_emb_dim']))
        torch.nn.init.xavier_uniform_(self.context_query_weight)

        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.dropout_prob = dropout_prob

        super(ContextQueryAttention, self).__init__()

    def forward(self, context: torch.Tensor, query: torch.Tensor, context_mask: torch.Tensor, query_mask: torch.Tensor):
        """
        Implements the Bidirectional attention as defined in https://arxiv.org/abs/1611.01603
        Returns

        :param context: The context words embeddings (BATCH_SIZE, context_max_length, hidden_size)
        :param query: The query words embeddings (BATCH_SIZE, query_max_length, hidden_size)
        :param context_mask: A mask of the context used to zero out the logits' entries relative to padding elements
            (BATCH_SIZE, context_max_length)
        :param query_mask: A mask of the context used to zero out the logits' entries relative to padding elements
            (BATCH_SIZE, question_max_length)
        """

        batch_size, context_len, _ = context.size()
        query_len = query.size(1)

        s = self.get_similarity_matrix(context, query)

        context_mask = torch.logical_not(context_mask.view(batch_size, context_len, 1))
        query_mask = torch.logical_not(query_mask.view(batch_size, 1, query_len))

        s1 = F.softmax(torch.masked_fill(s, query_mask, -1e8), dim=2)  # (BATCH_SIZE, context_max_length,
        # query_max_length)
        s2 = F.softmax(torch.masked_fill(s, context_mask, -1e8), dim=1)  # (BATCH_SIZE, context_max_length,
        # query_max_length)

        a = torch.bmm(s1, query)  # context to query attention (BATCH_SIZE, context_max_length, hidden_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), context)  # query to context attention
        # (BATCH_SIZE, context_max_length, hidden_size)

        x = torch.cat([context, a, context * a, context * b], dim=2)  # (BATCH_SIZE, context_max_length, 4 *
        # hidden_size)

        return x
    
    def get_similarity_matrix(self, context: torch.Tensor, query: torch.Tensor):
        """
        Computes the similarity matrix between context and query as described in https://arxiv.org/abs/1611.01603

        :param context: The context words embeddings (BATCH_SIZE, context_max_length, hidden_size)
        :param query: The query words embeddings (BATCH_SIZE, query_max_length, hidden_size)
        :return:
        """

        context_len, query_len = context.size(1), query.size(1)
        context = F.dropout(context, self.dropout_prob, self.training)
        query = F.dropout(query, self.dropout_prob, self.training)

        s0 = torch.matmul(context, self.context_weight).expand([-1, -1, query_len])
        s1 = torch.matmul(query, self.query_weight).transpose(1, 2).expand([-1, context_len, -1])
        s2 = torch.matmul(context * self.context_query_weight, query.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
