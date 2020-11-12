import torch
from torch_geometric.nn import GCNConv


class GCE(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, features, train_mat_edges):

        super().__init__()

        self.A = train_mat_edges
        self.features = features  # so far, Identity matrix
        # GCNConv applies the convolution over the graph
        self.GCN_module = GCNConv(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.GCN_module(self.features, self.A)[x]
