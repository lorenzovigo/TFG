import os
import torch
import torch.nn as nn
from model.GCE.gce import GCE, FactorizationMachine
import torch.backends.cudnn as cudnn


class PairFM(nn.Module):
    def __init__(self,
                 user_num, 
                 max_dim,
                 factors=84, 
                 epochs=20, 
                 lr=0.001, 
                 reg_1=0.,
                 reg_2=0.,
                 gpuid='0',
                 X=None,
                 A=None,
                 GCE_flag=False,
                 early_stop=True,
                 dropout=0):
        """
        Pair-wise FM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PairFM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.GCE_flag = GCE_flag
        self.fm = FactorizationMachine(reduce_sum=True)

        if GCE_flag:
            print('GCE EMBEDDINGS DEFINED')
            self.embeddings = GCE(max_dim, factors, X, A)
        else:
            self.embeddings = nn.Embedding(max_dim, factors)
            self.bias = nn.Embedding(max_dim, 1)
            self.bias_ = nn.Parameter(torch.tensor([0.0]))
            # init weight
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.bias.weight, 0.0)

        self.early_stop = early_stop

    def forward(self, u, i, j, context):

        if context is None:
            embeddings_ui = self.embeddings(torch.stack((u, i), dim=1))
            embeddings_uj = self.embeddings(torch.stack((u, j), dim=1))
        else:
            embeddings_ui = self.embeddings(torch.stack((u, i, context), dim=1))
            embeddings_uj = self.embeddings(torch.stack((u, j, context), dim=1))

        # inner prod part
        pred_i = self.fm(embeddings_ui)
        pred_j = self.fm(embeddings_uj)

        return pred_i.view(-1), pred_j.view(-1)

    def predict(self, u, i, c):
        pred_i, _ = self.forward(u, i, i, c)

        return pred_i.cpu()
