import os
import gc
import re
import json
import random
import numpy as np
import pandas as pd
import scipy.io as sio

from collections import defaultdict
from IPython import embed


def load_rate(src='ml-100k', prepro='origin', binary=True, pos_threshold=None, level='ui'):
    """
    Method of loading certain raw data
    Parameters
    ----------
    src : str, the name of dataset
    prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
    binary : boolean, whether to transform rating to binary label as CTR or not as Regression
    pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
    level : str, which level to do with f'{N}core' or f'{N}filter' operation (it only works when prepro contains 'core' or 'filter')

    Returns
    -------
    df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
    user_num : int, the number of users
    item_num : int, the number of items
    """
    df = pd.DataFrame()
    # which dataset will use
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None,
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')

    elif src == 'ml-1m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
        # only consider rating >=4 for data density
        # df = df.query('rating >= 4').reset_index(drop=True).copy()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    # encoding user_id and item_id
    df['user'] = pd.Categorical(df['user']).codes
    df['item'] = pd.Categorical(df['item']).codes

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    print(f'Finish loading [{src}]-[{prepro}] dataset')

    return df, user_num, item_num


def build_feat_idx_dict(df:pd.DataFrame, 
                        cat_cols:list=['user', 'item'], 
                        num_cols:list=[]):
    """
    Method of encoding features mapping for FM
    Parameters
    ----------
    df : pd.DataFrame feature dataframe
    cat_cols : List, list of categorical column names
    num_cols : List, list of numeric column names

    Returns
    -------
    feat_idx_dict : Dictionary, dict with index-feature column mapping information
    cnt : int, the number of features
    """
    feat_idx_dict = {}
    idx = 0
    for col in cat_cols:
        feat_idx_dict[col] = idx
        idx = idx + df[col].max() + 1
    for col in num_cols:
        feat_idx_dict[col] = idx
        idx += 1
    print('Finish build feature index dictionary......')

    cnt = 0
    for col in cat_cols:
        for _ in df[col].unique():
            cnt += 1
    for _ in num_cols:
        cnt += 1
    print(f'Number of features: {cnt}')

    return feat_idx_dict, cnt


def convert_npy_mat(user_num, item_num, df):
    """
    method of convert dataframe to numoy matrix
    Parameters
    ----------
    user_num : int, the number of users
    item_num : int, the number of items
    df :  pd.DataFrame, rating dataframe

    Returns
    -------
    mat : np.matrix, rating matrix
    """
    mat = np.zeros((user_num, item_num))
    for _, row in df.iterrows():
        u, i, r = row['user'], row['item'], row['rating']
        mat[int(u), int(i)] = float(r)
    return mat


def build_candidates_set(test_ur, train_ur, item_pool, candidates_num=1000):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_pool : the set of all items
    candidates_num : int, the number of candidates
    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    test_ucands = defaultdict(list)
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        sub_item_pool = item_pool - v - train_ur[k]  # remove GT & interacted
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
            test_ucands[k] = list(set(samples))
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))
    
    return test_ucands