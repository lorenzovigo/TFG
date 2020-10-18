import numpy as np
import math
import torch
import scipy.sparse as sp
from tqdm import tqdm


def getHitRatio(recommend_list, gt_item):
    """
    :param recommend_list:
    :param gt_item:
    :return: Returns whether gt_item is on recommended_list or not.
    """

    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item):
    """
    Computes the normalized discounted cumulative gain (NDCG)
    :param recommend_list:
    :param gt_item:
    :return: Computed NDGC.
    """

    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2)/math.log(idx+2)
    else:
        return 0

def getRMSE(predictions):
    # TODO wrong
    return math.pow(predictions[0]-torch.max(predictions), 2)


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


def build_adjacency_matrix(field_dims, field_mins, interactions):
    """
    Builds the adjacency matrix determined by a set of interactions with contexts.

    Parameters
    ----------
    dims : int
        The dimension the adjacency matrix should have.

    interactions : list
        Set of known interactions.

    Returns
    -------
    scipy.dok_matrix
        Fully built adjacency matrix.J
    """
    dims = field_dims[-1]
    train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
    aux = np.delete(np.insert(field_dims, 0, 0), -1)
    # For every interaction, we set the value as 1 in both grids that represent the pair of items that interact
    for x in tqdm(interactions, desc="Building Adjacency Matrix..."):
        # We fix the indices for the values of the context
        indices = x - field_mins + aux
        # We fill the interactions
        train_mat[indices[0], indices[1]] = 1.0
        train_mat[indices[1], indices[0]] = 1.0
        # We fill the positions for the context
        for i in range(2, np.shape(indices)[0]):
            train_mat[indices[0], indices[i]] = 1.0
            train_mat[indices[1], indices[i]] = 1.0

    return train_mat


def build_test_set(gt_test_interactions, max_users, max_items, train_mat, neg_ratio=99):
    """
    Every known interaction is considered a positive sample.
    This method generates random negative samples from items that have not interacted with each other.

    Parameters
    ----------
    gt_test_interactions : ndarray
        Ground truth items which should be retrieved by the model from the set created in this method.

    neg_ratio : int
        How many negative samples we will produce for every positive sample in the test set.

    Returns
    -------
    list
        Complete test set with both negative and positive samples.
    """
    # We initialize an array where the test set will be saved
    test_set = []

    for pair in tqdm(gt_test_interactions, desc="Building test set..."):
        negatives = []
        for t in range(neg_ratio):
            j = np.random.randint(max_users, max_items)
            while (pair[0], j) in train_mat or j == pair[1] or j in negatives:
                j = np.random.randint(max_users, max_items)
            negatives.append(j)
        # APPEND TEST SETS FOR SINGLE USER
        single_user_test_set = np.vstack([pair, ] * (len(negatives) + 1))
        single_user_test_set[:, 1][1:] = negatives
        test_set.append(single_user_test_set.copy())
    return test_set


