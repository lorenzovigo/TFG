import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model.GCE.gce import GCE


class PairMF(nn.Module):
    def __init__(self, 
                 user_num, 
                 max_dim,
                 factors=32,
                 epochs=20, 
                 lr=0.01, 
                 reg_1=0.001,
                 reg_2=0.001,
                 gpuid='0',
                 X=None,
                 A=None,
                 GCE_flag=False,
                 early_stop=True,
                 dropout=0):
        """
        Point-wise MF Recommender Class
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
        super(PairMF, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.dropout = dropout

        self.GCE_flag = GCE_flag

        if GCE_flag:
            print('GCE EMBEDDINGS DEFINED')
            self.embeddings = GCE(max_dim, factors, X, A)
        else:
            self.embeddings = nn.Embedding(max_dim, factors)
            nn.init.normal_(self.embeddings.weight, std=0.01)

    def forward(self, u, i, j, context):

        # embed()
        if context is None:
            embeddings_ui = self.embeddings(torch.stack((u, i), dim=1))
            embeddings_uj = self.embeddings(torch.stack((u, j), dim=1))
        else:
            embeddings_ui = self.embeddings(torch.stack((u, i, context), dim=1))
            embeddings_uj = self.embeddings(torch.stack((u, j, context), dim=1))

        # ix = torch.bmm(embeddings[:, :1, :], embeddings[:, 1:, :].permute(0, 2, 1))
        pred_i = embeddings_ui.prod(dim=1).sum(dim=1)
        pred_j = embeddings_uj.prod(dim=1).sum(dim=1)

        return pred_i, pred_j

    def predict(self, u, i, c):
        pred_i, _ = self.forward(u, i, i, c)

        return pred_i.cpu()
