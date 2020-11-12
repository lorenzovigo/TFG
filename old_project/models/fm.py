import torch
from layers import FeaturesLinear, FM_operation
from IPython import embed

from models.gm import GraphModel


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, X=None, A=None, gcn=True, reduce_sum=True, fm_operation=True):
        super().__init__()

        # field_dims == total of nodes (sum users + context)
        # self.linear = torch.nn.Linear(field_dims, 1, bias=True)
        self.linear = FeaturesLinear(field_dims)
        if gcn:
            self.embedding = GraphModel(field_dims, embed_dim, X, A)
        else:
            self.embedding = torch.nn.Embedding(field_dims, embed_dim, sparse=False)
            # Parameter initialization TODO duda también para gcn?
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        self.fm = FM_operation(reduce_sum, fm_operation)


    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """

        # We compute the formula as it is, and then get rid of the dimensions to return a number
        # TODO we shouldn't send interaction_pairs once context is added.
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