import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from daisy.model.GCE.gce import GCE
from IPython import embed


class PointNFM(nn.Module):
    def __init__(self,
                 user_num, 
                 item_num, 
                 factors, 
                 act_function, 
                 num_layers, 
                 batch_norm,
                 q, 
                 epochs, 
                 lr,
                 optimizer='adam',
                 reg_1=0., 
                 reg_2=0., 
                 loss_type='CL', 
                 gpuid='0',
                 reindex=False,
                 GCE_flag=False,
                 early_stop=True,
                 X=None,
                 A=None):
        """
        Point-wise NFM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        act_function : str, activation function for hidden layer
        num_layers : int, number of hidden layers
        batch_norm : bool, whether to normalize a batch of data
        q : float, dropout rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PointNFM, self).__init__()

        self.factors = factors
        self.act_function = act_function
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = q
        self.optimizer = optimizer

        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.epochs = epochs
        self.loss_type = loss_type
        self.early_stop = early_stop
        self.reindex = reindex
        self.GCE_flag = GCE_flag

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        if reindex:
            if self.GCE_flag:
                print('GCE EMBEDDINGS DEFINED')
                self.embeddings = GCE(user_num + item_num, factors, X, A)
            else:
                self.embeddings = nn.Embedding(user_num + item_num, factors)
                self.bias = nn.Embedding(user_num + item_num, 1)
                self.bias_ = nn.Parameter(torch.tensor([0.0]))

        else:
            self.embed_user = nn.Embedding(user_num, factors)
            self.embed_item = nn.Embedding(item_num, factors)

            self.u_bias = nn.Embedding(user_num, 1)
            self.i_bias = nn.Embedding(item_num, 1)

            self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(factors))
        FM_modules.append(nn.Dropout(self.dropout))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_modules = []
        in_dim = factors
        for _ in range(self.num_layers):  # dim
            out_dim = in_dim # dim
            MLP_modules.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            if self.batch_norm:
                MLP_modules.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_modules.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_modules.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_modules.append(nn.Tanh())
            MLP_modules.append(nn.Dropout(self.dropout))
        self.deep_layers = nn.Sequential(*MLP_modules)
        predict_size = factors  # layers[-1] if layers else factors

        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight()

    def _init_weight(self):
        if self.reindex and not self.GCE_flag:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.bias.weight, 0.0)
        elif not self.reindex:
            nn.init.normal_(self.embed_item.weight, std=0.01)
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.constant_(self.u_bias.weight, 0.0)
            nn.init.constant_(self.i_bias.weight, 0.0)

        # for deep layers
        if self.num_layers > 0:  # len(self.layers)
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, user, item):

        if self.reindex:
            embeddings = self.embeddings(torch.stack((user, item), dim=1))
            fm = embeddings.prod(dim=1)  # shape [256, 32]
        else:
            embed_user = self.embed_user(user)
            embed_item = self.embed_item(item)
            fm = embed_user * embed_item

        fm = self.FM_layers(fm)

        if self.num_layers:
            fm = self.deep_layers(fm)

        if self.reindex and not self.GCE_flag:
            fm += self.bias_

        elif not self.GCE_flag:
            fm += self.u_bias(user) + self.i_bias(item) + self.bias_

        pred = self.prediction(fm)

        return pred.view(-1)

    def predict(self, u, i):
        pred = self.forward(u, i).cpu()
        
        return pred
