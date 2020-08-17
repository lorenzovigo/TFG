import torch
from layers import FeaturesLinear, FM_operation
from models.gm import GraphModel


class FactorizationMachineModel_withGCN(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, X, A):
        super().__init__()

        self.linear = FeaturesLinear(field_dims)
        # This is the only thing that changes from fmm
        # self.embedding = torch.nn.Embedding(field_dims, embed_dim, sparse=False)
        self.embedding = GraphModel(field_dims, embed_dim, X, A)
        self.fm = FM_operation(reduce_sum=True)

        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))
        return out.squeeze(1)

    def predict(self, interactions, device):
        """
        Predicts the score for given interactions.
        :param interactions: Interactions which score we will calculated (numpy array).
        :param device: Device used to calculate the predictions.
        :return: Predicted scores (tensor).
        """
        # return the score, inputs are numpy arrays, outputs are tensors

        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores
