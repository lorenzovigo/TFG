import numpy as np
import pandas as pd
import time, torch
from tqdm import tqdm
from utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit
from IPython import embed


def perform_evaluation(loaders, candidates, model, args, device, test_ur, s_time=None, writer=None, epoch=None,
                       minutes_train=None, seconds_train=None, tune=False):

    preds = {}
    # for u_idx, tmp_loader in enumerate(loaders):
    for u in tqdm(candidates.keys(), disable=tune):
        # get top-N list with torch method
        for items in loaders[u]:
            user_u, item_i, context = items[0], items[1], items[2]
            user_u = user_u.to(device)
            item_i = item_i.to(device)
            context = context.to(device) if args.context else None
            prediction = model.predict(user_u, item_i, context)
            _, indices = torch.topk(prediction, args.topk)
            top_n = torch.take(torch.tensor(candidates[u]), indices).cpu().numpy()

        preds[u] = top_n

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    # res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    res = pd.DataFrame({'metric@K': ['hr', 'ndcg']})
    tmp_pred_10 = []
    for k in [10, 20, 30, 40, 50]:
        if k > args.topk:
            continue
        tmp_preds = preds.copy()
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        # pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        # rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        # map_k = map_at_k(tmp_preds.values())
        # mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if (writer and epoch) and not tune:
            writer.add_scalar(f'metrics/HR_@{k}', hr_k, epoch)
            writer.add_scalar(f'metrics/NDCG_@{k}', ndcg_k, epoch)
            # print(f'HR@{k}: {hr_k:.4f}  |  NDCG@{k}: {ndcg_k:.4f}')

        # res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])
        res[k] = np.array([hr_k, ndcg_k])
        if k == 10:
            tmp_pred_10 = np.array([hr_k, ndcg_k])
        if not (writer and not epoch is None) and not tune:
            if k == 10:
                print('--------------TEST METRICS ------------')
                print('+'*80)
                print('+'*80)
            # print(f'Precision@{k}: {pre_k:.4f}')
            # print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            # print(f'MAP@{k}: {map_k:.4f}')
            # print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')

    if not (writer and not epoch is None) and not tune:
        print(f'TRAINING ELAPSED TIME: {minutes_train:.2f} min, {seconds_train:.4f}seconds')

        elapsed_time_total = time.time() - s_time
        hours, rem = divmod(elapsed_time_total, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'TOTAL ELAPSED TIME: {minutes:.2f} min, {seconds:.4f}seconds')

    return res, writer, tmp_pred_10
        

def split_test(df, test_method='fo', test_size=.2):
    """
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    test_method : str, way to split test set
                    'tloo': leave one out with timestamp
                    'loo': leave one out
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

    elif test_method == 'loo':
        # # slow method
        # test_set = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        # test_key = test_set[['user', 'item']].copy()
        # train_set = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()

        # # quick method
        test_index = df.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
        test_set = df.loc[test_index, :].copy()
        train_set = df[~df.index.isin(test_index)].copy()

    else:
        raise ValueError('Invalid data_split value, expect: loo, tloo')

    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)

    return train_set, test_set


def split_validation(train_set, val_method='fo', fold_num=1, val_size=.1, list_output=True):
    """
    method of split data into training data and validation data.
    (Currently, this method returns list of train & validation set, but I'll change 
    it to index list or generator in future so as to save memory space) TODO

    Parameters
    ----------
    train_set : pd.DataFrame train set waiting for split validation
    val_method : str, way to split validation
                    'tloo': Leave one out with timestamp => 1-Leave one out
                    'loo': combine with fold_num => fold_num-Leave one out
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
    elif val_method in ['loo']:
        cnt = fold_num
    else:
        raise ValueError('Invalid val_method value, expect: loo, tloo')
    if list_output:
        train_set_list, val_set_list = [], []
    else:
        train_set_list, val_set_list = pd.DataFrame(), pd.DataFrame()

    if val_method == 'loo':
        for _ in range(fold_num):
            val_index = train_set.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
            val_set = train_set.loc[val_index, :].reset_index(drop=True).copy()
            sub_train_set = train_set[~train_set.index.isin(val_index)].reset_index(drop=True).copy()

            train_set_list.append(sub_train_set)
            val_set_list.append(val_set)
    elif val_method == 'tloo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)

        train_set['rank_latest'] = train_set.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        new_train_set = train_set[train_set['rank_latest'] > 1].copy()
        val_set = train_set[train_set['rank_latest'] == 1].copy()
        del new_train_set['rank_latest'], val_set['rank_latest']

        if list_output:
            train_set_list.append(new_train_set)
            val_set_list.append(val_set)
        else:
            train_set_list = new_train_set
            val_set_list = val_set

    return train_set_list, val_set_list, cnt


