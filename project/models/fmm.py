import torch
from layers import FeaturesLinear, FM_operation

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()

        # field_dims == total of nodes (sum users + context)
        # TODO duda: por qué no usamos esto?
        # self.linear = torch.nn.Linear(field_dims, 1, bias=True)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = torch.nn.Embedding(field_dims, embed_dim, sparse=False)
        self.fm = FM_operation(reduce_sum=True)

        # TODO duda no sé qué pinta el embedding otra vez, y es la inicialización de los parámetros?
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """

        # We compute the formula as it is, and then get rid of the dimensions to return a number (so I believe)
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))

        return out.squeeze(1)

    def predict(self, interactions, device):
        # return the score, inputs are numpy arrays, outputs are tensors

        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores
