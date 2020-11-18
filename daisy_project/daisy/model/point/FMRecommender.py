import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from IPython import embed
from daisy.model.GCE.gce import GCE


class PointFM(nn.Module):
    def __init__(self, 
                 max_dim,
                 factors=84, 
                 epochs=20,
                 optimizer='adam',
                 lr=0.001,
                 reg_1=0.001,
                 reg_2=0.001,
                 loss_type='SL',
                 gpuid='0',
                 X=None,
                 A=None,
                 GCE_flag=False,
                 early_stop=True):
        """
        Point-wise FM Recommender Class
        Parameters
        ----------
        max_dim : int, number of fields (including, users, items and contexts)
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PointFM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.reg_1 = reg_1
        self.reg_2 = reg_2

        self.GCE_flag = GCE_flag

        if GCE_flag:
            print('GCE EMBEDDINGS DEFINED')
            self.embeddings = GCE(max_dim, factors, X, A)
        else:
            self.embeddings = nn.Embedding(max_dim, factors)
            self.bias = nn.Embedding(max_dim, 1)
            self.bias_ = nn.Parameter(torch.tensor([0.0]))

            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.bias.weight, 0.0)

        self.loss_type = loss_type

    def forward(self, user, item, context): # TODO duda: por qué lo pasamos por separado y no todo junto como en el anterior proyecto
        # TODO duda: la parte lineal ahora es bias y bias_ y embeddings.prod "hace de fm"?
        if context is None:
            embeddings = self.embeddings(torch.stack((user, item), dim=1))
        else:
            embeddings = self.embeddings(torch.stack((user, item, context), dim=1))
        pred = embeddings.prod(dim=1).sum(dim=1, keepdim=True)

        if not self.GCE_flag:
            pred += self.bias(torch.stack((user, item), dim=1)).sum() + self.bias_
        # return torch.squeeze(ix)
        return pred.view(-1)


    def predict(self, u, i, c):
        pred = self.forward(u, i, c).cpu()
        
        return pred
