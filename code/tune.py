import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from hyperopt import hp, tpe, fmin, Trials, space_eval

import torch
import torch.utils.data as data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity
from IPython import embed


from utils.sampler import Sampler
from utils.parser import parse_args
from model.pair.train import train
from utils.data import PairData, sparse_mx_to_torch_sparse_tensor
from utils.splitter import split_test, split_validation
from utils.loader import load_rate, get_ur, build_candidates_set, add_last_clicked_item_context
from utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k
from utils.tunner import param_extract, confirm_space

from main import build_evaluation_set


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def opt_func(space):

    ''' FORMAT DATA AND CHOOSE MODEL '''
    f = space['f']
    args = Struct(**space)

    user_num = dims[0]
    max_dim = dims[2] if args.context else dims[1]

    if args.algo_name == 'mf':
        from daisy.model.pair.MFRecommender import PairMF

        model = PairMF(
            user_num,
            max_dim,
            factors=args.factors,
            epochs=args.epochs,
            lr=args.lr,
            reg_1=args.reg_1,
            reg_2=args.reg_2,
            loss_type=args.loss_type,
            GCE_flag=args.gce,
            reindex=args.reindex,
            X=X if args.gce else None,
            A=edge_idx if args.gce else None,
            gpuid=args.gpu,
            dropout=args.dropout
        )
    elif args.algo_name == 'fm':
        from daisy.model.pair.FMRecommender import PairFM

        model = PairFM(
            user_num,
            max_dim,
            factors=args.factors,
            epochs=args.epochs,
            lr=args.lr,
            reg_1=args.reg_1,
            reg_2=args.reg_2,
            loss_type=args.loss_type,
            GCE_flag=args.gce,
            reindex=args.reindex,
            X=X if args.gce else None,
            A=edge_idx if args.gce else None,
            gpuid=args.gpu,
            dropout=args.dropout
        )
    elif args.algo_name == 'nfm':
        from daisy.model.pair.NFMRecommender import PairNFM

        model = PairNFM(
            user_num,
            max_dim,
            factors=args.factors,
            act_function=args.act_func,
            num_layers=args.num_layers,
            batch_norm=args.no_batch_norm,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            reg_1=args.reg_1,
            reg_2=args.reg_2,
            loss_type=args.loss_type,
            GCE_flag=args.gce,
            reindex=args.reindex,
            X=X if args.gce else None,
            A=edge_idx if args.gce else None,
            gpuid=args.gpu,
            mf=args.mf
        )
    elif args.algo_name == 'ncf':
        layers = [len(dims[:-2]) * 32, 32, 16, 8] if not args.context else [len(dims[:-2]) * 32, 32, 16, 8]
        from daisy.model.pair.NCFRecommender import PairNCF

        model = PairNCF(
            user_num,
            max_dim,
            factors=args.factors,
            layers=layers,
            GCE_flag=args.gce,
            reindex=args.reindex,
            X=X if args.gce else None,
            A=edge_idx if args.gce else None,
            gpuid=args.gpu,
            mf=args.mf,
            dropout=args.dropout
        )
    elif args.algo_name == 'deepfm':
        from daisy.model.pair.DeepFMRecommender import PairDeepFM

        model = PairDeepFM(
            user_num,
            max_dim,
            factors=args.factors,
            act_activation=args.act_func,
            num_layers=args.num_layers,
            batch_norm=args.no_batch_norm,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            reg_1=args.reg_1,
            reg_2=args.reg_2,
            loss_type=args.loss_type,
            GCE_flag=args.gce,
            reindex=args.reindex,
            context_flag=args.context,
            X=X if args.gce else None,
            A=edge_idx if args.gce else None,
            gpuid=args.gpu,
        )
    else:
        raise ValueError('Invalid algorithm name')

    ''' BUILD RECOMMENDER PIPELINE '''

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    loaders, candidates = build_evaluation_set(val_ur, total_train_ur, item_pool, candidates_num, sampler,
                                               context_flag=args.context, tune=args.tune)
    score = train(args, model, train_loader, device, args.context, loaders, candidates, val_ur, tune=args.tune, f=f)
    return score


if __name__ == '__main__':

    ''' all parameter part '''
    args = parse_args()
    seed = 1234
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    else:
        device = "cpu"

    ''' LOAD DATA AND ADD CONTEXT IF NECESSARY '''
    df, users, items = load_rate(args.dataset, args.prepro, binary=True, context=args.context, gce_flag=args.gce,
                                 cut_down_data=args.cut_down_data)
    if args.reindex:
        df = df.astype(np.int64)
        df['item'] = df['item'] + users
        if args.context:
            df = add_last_clicked_item_context(df, args.dataset)
            # check last number is positive
            assert df['item'].tail().values[-1] > 0

    ''' SPLIT DATA '''
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    train_set, val_set, _ = split_validation(train_set, val_method=args.test_method, list_output=False)

    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    df = pd.concat([train_set, test_set], ignore_index=True)
    dims = np.max(df.to_numpy().astype(int), axis=0) + 1
    if args.dataset in ['yelp']:
        train_set['timestamp'] = pd.to_datetime(train_set['timestamp'], unit='ns')
        test_set['timestamp'] = pd.to_datetime(test_set['timestamp'], unit='ns')

    ''' GET GROUND-TRUTH AND CANDIDATES '''
    # get ground truth
    test_ur = get_ur(test_set, context=args.context, eval=False)
    val_ur = get_ur(val_set, context=args.context, eval=False)

    total_train_ur = get_ur(train_set, context=args.context, eval=True)
    # initial candidate item pool
    item_pool = set(range(dims[0], dims[1])) if args.reindex else set(range(dims[1]))
    candidates_num = args.cand_num

    print('=' * 50, '\n')

    sampler = Sampler(
        dims,
        num_ng=args.num_ng,
        sample_method=args.sample_method,
        sample_ratio=args.sample_ratio,
        reindex=args.reindex
    )

    neg_set, adj_mx = sampler.transform(train_set, is_training=True, context=args.context, pair_pos=None)
    if args.gce:
        if args.mh > 1:
            print(f'[ MULTI HOP {args.mh} ACTIVATED ]')
            adj_mx = adj_mx.__pow__(int(args.mh))
        X = sparse_mx_to_torch_sparse_tensor(identity(adj_mx.shape[0])).to(device)
        # We retrieve the graph's edges and send both them and graph to device in the next two lines
        edge_idx, edge_attr = from_scipy_sparse_matrix(adj_mx)
        edge_idx = edge_idx.to(device)
    train_dataset = PairData(train_set, sampler=sampler, adj_mx=adj_mx, is_training=True, context=args.context)

    print('='*50, '\n')
    # begin tuning here
    tune_log_path = 'tune_logs'
    os.makedirs(tune_log_path, exist_ok=True)

    f = open(tune_log_path + "/" + f'{args.loss_type}_{args.algo_name}_GCE={args.gce}_{args.dataset}_{args.prepro}_{args.val_method}.csv',
             'w', encoding='utf-8')
    f.write('HR, NDCG, best_epoch, num_ng, factors, dropout, lr, batch_size, reg_1, reg_2' + '\n')
    f.flush()

    # param_limit = param_extract(args)
    # param_dict = confirm_space(param_limit)

    args_dict = vars(args)

    lr_range = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    batch_size_range = [256, 512, 1024, 2048]
    do_range = [0, 0.15, 0.5]

    args_dict['lr'] = hp.choice('lr', lr_range)
    args_dict['batch_size'] = hp.choice('batch_size', batch_size_range)
    args_dict['dropout'] = hp.choice('dropout', do_range)
    args_dict['epochs'] = args_dict['tune_epochs']
    args_dict['f'] = f
    args_dict['tune'] = True

    trials = Trials()
    # trials = pickle.load(open("myfile.p", "rb"))  # then max_evals needs to be set to 200

    space = defaultdict(None, args_dict)
    best = fmin(fn=opt_func,
                space=space, algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    # pickle.dump(trials, open("myfile.p", "wb"))
    print(""*20 +'BEST HYPER_PARAMS:' + ""*20)
    print("lr = " + str(lr_range[best['lr']]))
    print("batch_size = " + str(batch_size_range[best['batch_size']]))
    print("dropout = " + str(do_range[best['dropout']]))
    # lr_range[best['lr']]
    best_options = space_eval(space, trials.argmin)

    f.write('BEST ITERATION PARAMS' + '\n')
    f.write(f"-, -, -, {best_options['num_ng']}, {best_options['factors']}, {best_options['dropout']},"
            f"+ {best_options['lr']}, {best_options['batch_size']}, {best_options['reg_1']}, {best_options['reg_2']}" + '\n')
    f.flush()
    f.close()

    # def hyperopt_bug():
    #     space = defaultdict(dict)
    #     space["a"]["aa"] = hp.choice("aa", [1])
    #     space["a"]["aaa"] = hp.uniform("aaa", 1, 100)
    #
    #     space["b"]["bb"] = hp.choice("bb", [1])
    #     space["b"]["bbb"] = hp.uniform("bbb", 1, 100)
    #
    #     func = lambda r: r["a"]["aa"] + r["a"]["aaa"] + r["b"]["bb"] + r["b"]["bbb"]
    #     trials = Trials()
    #     best = fmin(func, space, trials=trials, algo=tpe.suggest, max_evals=120)
