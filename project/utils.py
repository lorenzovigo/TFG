import numpy as np
import math
import torch
import os, urllib, zipfile


def load_dataset(self):
    """Downloads and extracts the MovieLens - 100k dataset zip file."""
    # Check whether dataset is already downloaded or not
    if not os.path.exists(self.dataset_path):
        # Download dataset
        print(f'Downloading {self.dataset_name} dataset...')
        urllib.request.urlretrieve(self.url, self.downloaded_file)

        # Create data folder if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        # Extract dataset
        print(f'Extracting {self.dataset_name} dataset...')
        with zipfile.ZipFile(self.downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)

        # Delete zipfile
        os.remove(self.downloaded_file)


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

