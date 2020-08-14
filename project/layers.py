import torch


class FeaturesLinear(torch.nn.Module):

    # We use this layer as a representation of the linear part of the formula

    def __init__(self, field_dims, output_dim=1):
        super().__init__()

        # TODO duda: el embedding recibe un input de tamaño n, y devuelve un solo número (entiendo lo que pretende hacer pero no cómo lo hace, los parámetros están dentro del embedding, el embedding hace la suma?)
        self.fc = torch.nn.Embedding(field_dims, output_dim)

        # TODO duda: bias es w0 en la fórmula, entiendo que al ser Parámetro el Modulo lo va corrigiendo
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        # TODO duda: entiendo que aquí la suma en la dimensión 1 se debe a que la pasamos varios inputs (batch_size), lo comprimos todo en uno?
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

        # TODO duda: entiendo que esto implementa la expresión simplificada de la parte final de los sumatorios, pero no veo donde están los parámetros ni veo muy bien lo de las dimensiones (pero eso ya viendolo funcionar debería entenderlo)

        square_of_sum = torch.sum(x, dim=1) ** 2 #S1
        sum_of_square = torch.sum(x ** 2, dim=1) #S2
        ix = square_of_sum - sum_of_square

        # TODO duda: lo de las dimensiones sobre todo no me cuadra aquí, no debería sumarse a lo largo de k, es decir, la dimensión 2?

        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
