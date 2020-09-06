import torch


class FeaturesLinear(torch.nn.Module):

    # We use this layer as a representation of the linear part of the formula

    def __init__(self, field_dims, output_dim=1):
        super().__init__()

        self.fc = torch.nn.Embedding(field_dims, output_dim)

        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        return torch.sum(self.fc(x), dim=1) + self.bias


class FM_operation(torch.nn.Module):

    # We use this layer as a representation of FM part of the equation

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """

        square_of_sum = torch.sum(x, dim=1) ** 2 #S1
        sum_of_square = torch.sum(x ** 2, dim=1) #S2
        ix = square_of_sum - sum_of_square

        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
