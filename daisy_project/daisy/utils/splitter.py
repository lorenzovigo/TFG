import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit

def split_test(df, test_method='tloo', test_size=.2):
    """
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    test_method : str, way to split test set
                    'tloo': leave one out with timestamp
    test_size : float, size of test set

    Returns
    -------
    train_set : pd.DataFrame training dataset
    test_set : pd.DataFrame test dataset

    """

    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    if test_method == 'tloo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
        del train_set['rank_latest'], test_set['rank_latest']

    else:
        raise ValueError('Invalid data_split value, expect: loo, fo, tloo, tfo')

    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)

    return train_set, test_set


def split_validation(train_set, val_method='tloo', fold_num=1, val_size=.1):
    """
    method of split data into training data and validation data.
    (Currently, this method returns list of train & validation set, but I'll change 
    it to index list or generator in future so as to save memory space) TODO

    Parameters
    ----------
    train_set : pd.DataFrame train set waiting for split validation
    val_method : str, way to split validation
                    'tloo': Leave one out with timestamp => 1-Leave one out
    fold_num : int, the number of folder need to be validated, only work when val_method is 'cv', 'loo', or 'fo'
    val_size: float, the size of validation dataset

    Returns
    -------
    train_set_list : List, list of generated training datasets
    val_set_list : List, list of generated validation datasets
    cnt : cnt: int, the number of train-validation pair

    """
    if val_method in ['tloo']:
        cnt = 1
    else:
        raise ValueError('Invalid val_method value, expect: tloo')
    
    train_set_list, val_set_list = [], []

    if val_method == 'tloo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)

        train_set['rank_latest'] = train_set.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        new_train_set = train_set[train_set['rank_latest'] > 1].copy()
        val_set = train_set[train_set['rank_latest'] == 1].copy()
        del new_train_set['rank_latest'], val_set['rank_latest']

        train_set_list.append(new_train_set)
        val_set_list.append(val_set)

    return train_set_list, val_set_list, cnt

