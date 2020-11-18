import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from IPython import embed
from daisy.model.GCE.gce import GCE


class PointMF(nn.Module):
    def __init__(self, 
                 max_dim,
                 factors=100,
                 optimizer='adam',
                 epochs=20, 
                 lr=0.01, 
                 reg_1=0.001,
                 reg_2=0.001,
                 loss_type='CL', 
                 gpuid='0',
                 X = None,
                 A = None,
                 GCE_flag=False,
                 early_stop=True):
        """
        Point-wise MF Recommender Class
        Parameters
        ----------
        max_dim: int, number of fields (including, users, items and contexts)
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PointMF, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.epochs = epochs
        self.optimizer = optimizer
        self.GCE_flag = GCE_flag

        if GCE_flag:
            print('GCE EMBEDDINGS DEFINED')
            self.embeddings = GCE(max_dim, factors, X, A)
        else:
            self.embeddings = nn.Embedding(max_dim, factors)
            nn.init.normal_(self.embeddings.weight, std=0.01)

        self.loss_type = loss_type

    def forward(self, user, item, context):

        # embed()
        if context is None:
            embeddings = self.embeddings(torch.stack((user, item), dim=1))
        else:
            embeddings = self.embeddings(torch.stack((user, item, context), dim=1))
        # ix = torch.bmm(embeddings[:, :1, :], embeddings[:, 1:, :].permute(0, 2, 1))
        pred = embeddings.prod(dim=1).sum(dim=1)
        return pred

    def predict(self, u, i, c):
        pred = self.forward(u, i, c).cpu()
        
        return pred