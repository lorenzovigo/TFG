import os
import time
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe
from tqdm import tqdm

import torch
import torch.utils.data as data

from daisy.utils.sampler import Sampler
from daisy.utils.parser import parse_args
from daisy.utils.splitter import split_test
from daisy.utils.data import PointData, PairData, UAEData, sparse_mx_to_torch_sparse_tensor
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity
from IPython import embed

from daisy.utils.parser import parse_space

def main(space):
    # FIX SEED AND SELECT DEVICE
    seed = 1234
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    else:
        device = "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    # store running time in time_log file
    time_log = open('time_log.txt', 'a') 
    
    ''' Test Process for Metrics Exporting '''
    df, user_num, item_num = load_rate(space['dataset'], space['prepro'], binary=False)
    df['item'] = df['item'] + user_num

    train_set, test_set = split_test(df, space['test_size'])
    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}.dat')

    df = pd.concat([train_set, test_set], ignore_index=True)
    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    train_set['rating'] = 1.0
    test_set['rating'] = 1.0

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    # initial candidate item pool
    item_pool = set(range(user_num, item_num+user_num))
    candidates_num = space['cand_num']

    print('='*50, '\n')
    # retrain model by the whole train set
    # format training data
    sampler = Sampler(
        user_num, 
        item_num, 
        num_ng=space['num_ng'],
        sample_method=space['sample_method'],
        sample_ratio=space['sample_ratio'],
    )
    neg_set, adj_mx = sampler.transform(train_set, is_training=True)
    if space['gce']:
        X = sparse_mx_to_torch_sparse_tensor(identity(adj_mx.shape[0])).to(device)
        # We retrieve the graph's edges and send both them and graph to device in the next two lines
        edge_idx, edge_attr = from_scipy_sparse_matrix(adj_mx)
        edge_idx = edge_idx.to(device)

    train_dataset = PointData(neg_set, is_training=True)

    if space['problem_type'] == 'point':
        if space['algo_name'] == 'mf':
            from daisy.model.point.MFRecommender import PointMF
            model = PointMF(
                user_num, 
                item_num, 
                factors=space['factors'],
                epochs=space['epochs'],
                lr=space['lr'],
                reg_1=space['reg_1'],
                reg_2=space['reg_2'],
                loss_type=space['loss_type'],
                X=X if space['gce'] else None,
                GCE_flag=space['gce'],
                A=edge_idx if space['gce'] else None,
                gpuid=space['gpu']
            )
        elif space['algo_name'] == 'fm':
            from daisy.model.point.FMRecommender import PointFM
            model = PointFM(
                user_num, 
                item_num,
                factors=space['factors'],
                epochs=space['epochs'],
                lr=space['lr'],
                reg_1=space['reg_1'],
                reg_2=space['reg_2'],
                loss_type=space['loss_type'],
                GCE_flag=space['gce'],
                X=X if space['gce'] else None,
                A=edge_idx if space['gce'] else None,
                gpuid=space['gpu']
            )
        elif space['algo_name'] == 'nfm':
            from daisy.model.point.NFMRecommender import PointNFM
            model = PointNFM(
                user_num,
                item_num,
                factors=space['factors'],
                act_function=space['act_func'],
                num_layers=space['num_layers'],
                batch_norm=space['no_batch_norm'],
                q=space['dropout'],
                epochs=space['epochs'],
                lr=space['lr'],
                reg_1=space['reg_1'],
                reg_2=space['reg_2'],
                loss_type=space['loss_type'],
                GCE_flag=space['gce'],
                X=X if space['gce'] else None,
                A=edge_idx if space['gce'] else None,
                gpuid=space['gpu']
            )
        else:
            raise ValueError('Invalid algorithm name')
    else:
        raise ValueError('Invalid problem type')

    # if args.algo_name == 'mostpop':
    #     train_loader = train_dataset
    #     args.num_workers = 0
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=space['batch_size'],
        shuffle=True,
        num_workers=space['num_workers']
    )

    # build recommender model
    s_time = time.time()
    # TODO: refactor train
    if space['problem_type'] == 'point':
        from daisy.model.point.train import train
        train(space, model, train_loader, device)
    else:
        raise ValueError()
    # model.fit(train_loader)
    elapsed_time = time.time() - s_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    time_log.write(f"{space['dataset']}_{space['prepro']}_tloo_{space['problem_type']}{space['algo_name']}"
                   f"_{space['loss_type']}_{space['sample_method']}_GCE={space['gce']},  {minutes:.2f} min, {seconds:.4f}seconds" + '\n')
    time_log.close()

    print('Start Calculating Metrics......')

    test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num)

    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}
    for u in tqdm(test_ucands.keys()):
        # build a test MF dataset for certain user u to accelerate
        tmp = pd.DataFrame({
            'user': [u for _ in test_ucands[u]],
            'item': test_ucands[u],
            'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
        })
        tmp_neg_set = sampler.transform(tmp, is_training=False)
        tmp_dataset = PairData(tmp_neg_set, is_training=False)
        tmp_loader = data.DataLoader(
            tmp_dataset,
            batch_size=candidates_num,
            shuffle=False,
            num_workers=0
        )
        # get top-N list with torch method
        for items in tmp_loader:
            user_u, item_i = items[0], items[1]
            user_u = user_u.to(device)
            item_i = item_i.to(device)

            prediction = model.predict(user_u, item_i)
            _, indices = torch.topk(prediction, space['topk'])
            top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()

        preds[u] = top_n

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f"./res/{space['dataset']}/{space['prepro']}/tloo/"
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    for k in [1, 5, 10, 20, 30, 50]:
        if k > space['topk']:
            continue
        tmp_preds = preds.copy()        
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        map_k = map_at_k(tmp_preds.values())
        mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if k == 10:
            # print(f'Precision@{k}: {pre_k:.4f}')
            # print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            # print(f'MAP@{k}: {map_k:.4f}')
            # print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')

        res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])

    common_prefix = f"with_{space['sample_ratio']}{space['sample_method']}"
    algo_prefix = f"{space['loss_type']}_{space['problem_type']}_{space['algo_name']}"

    res.to_csv(
        f"{result_save_path}{algo_prefix}_{common_prefix}_GCE={space['gce']}_kpi_results.csv",
        index=False
    )

    print('+'*80)
    print('+'*80)
    print(f'TRAINING ELAPSED TIME: {minutes:.2f} min, {seconds:.4f}seconds')

    elapsed_time_total = time.time() - s_time
    hours, rem = divmod(elapsed_time_total, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'TOTAL ELAPSED TIME: {minutes:.2f} min, {seconds:.4f}seconds')

    # if space['tune']:
    #    return {'loss': 0, 'status': STATUS_OK}
    return 0

if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()
    space = parse_space(args, tune=args.tune)

    if not space['tune']:
        main(space)
    else:
        trials = Trials()

        best = fmin(fn=main,
                    space=space, algo=tpe.suggest,
                    max_evals=1000,
                    trials=trials)

        print(best)
