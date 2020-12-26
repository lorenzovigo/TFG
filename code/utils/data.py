import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import torch

class PairData(data.Dataset):
    def __init__(self, data, sampler=None, adj_mx=None, context=False, is_training=True):
        """
        Dataset formatter adapted pair-wise algorithms
        Parameters
        ----------
        neg_set : List,
        is_training : bool,
        """
        super(PairData, self).__init__()
        self.features_fill = []
        self.context = context
        self.adj_mx = adj_mx
        self.is_training = is_training
        self.set = data
        self.sampler = sampler

        self._neg_sampling()

    def __len__(self):
        return len(self.features_fill)

    def _neg_sampling(self):
        if self.is_training:
            neg_set, _ = self.sampler.transform(self.set, is_training=self.is_training, context=self.context,
                                                pair_pos=self.adj_mx)
            self.neg_set = neg_set
        else:
            assert self.sampler is None and self.adj_mx is None
            self.neg_set = self.set
            # sampler and adj_mx should be none
            # self.set == negative_set for evaluation

        if self.context:
            for u, i, c, r, js in self.neg_set:
                u, i, c, r = int(u), int(i), int(c), np.float32(1)
                if self.is_training:
                    for j in js:
                        self.features_fill.append([u, i, c, j, r])
                else:
                    self.features_fill.append([u, i, c, i, r])
        else:
            for u, i, r, js in self.neg_set:
                u, i, r = int(u), int(i), np.float32(1)
                if self.is_training:
                    for j in js:
                        self.features_fill.append([u, i, j, r])
                else:
                    self.features_fill.append([u, i, i, r])

    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        context = [] if not self.context else features[idx][2]
        item_j = features[idx][2] if not self.context else features[idx][3]
        label = features[idx][3] if not self.context else features[idx][4]

        return user, item_i, context, item_j, label


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    :param sparse_mx: Scipy sparse matrix.
    :return: Torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)