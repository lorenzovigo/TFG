import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model.GCE.gce import GCE
from IPython import embed


class PairMF(nn.Module):
    def __init__(self, 
                 user_num, 
                 max_dim,
                 factors=32,
                 epochs=20, 
                 lr=0.01, 
                 reg_1=0.001,
                 reg_2=0.001,
                 loss_type='BPR', 
                 gpuid='0',
                 reindex=False,
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
        loss_type : str, loss function type
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

        self.reindex = reindex
        self.GCE_flag = GCE_flag
        self.loss_type = loss_type

        if GCE_flag:
            print('GCE EMBEDDINGS DEFINED')
            self.embeddings = GCE(max_dim, factors, X, A) if reindex else ValueError(f'Can not use GCE with'
                                                                                                 f'reindex=False')
        else:
            if reindex:
                self.embeddings = nn.Embedding(max_dim, factors)
                nn.init.normal_(self.embeddings.weight, std=0.01)
            else:
                self.embed_user = nn.Embedding(user_num, factors)
                self.embed_item = nn.Embedding(max_dim, factors)
                nn.init.normal_(self.embed_user.weight, std=0.01)
                nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, u, i, j, context):

        if self.reindex:
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
        else:
            user = self.embed_user(u)
            item_i = self.embed_item(i)
            item_j = self.embed_item(j)

            pred_i = (user * item_i).sum(dim=-1)
            pred_j = (user * item_j).sum(dim=-1)

        return pred_i, pred_j

    def predict(self, u, i, c):
        pred_i, _ = self.forward(u, i, i, c)

        return pred_i.cpu()
