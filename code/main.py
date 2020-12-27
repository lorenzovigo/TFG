import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from utils.sampler import Sampler
from utils.parser import parse_args
from utils.splitter import split_test, split_validation, perform_evaluation
from utils.data import PairData, sparse_mx_to_torch_sparse_tensor
from utils.loader import load_rate, get_ur, build_candidates_set, add_last_clicked_item_context

from model.pair.train import train

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity


def build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num, sampler, context_flag=False, tune=False):
    test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num, context_flag=context_flag)

    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    loaders = {}
    for u in tqdm(test_ucands.keys(), disable=tune):
        # build a test MF dataset for certain user u to accelerate
        if context_flag:
            tmp = pd.DataFrame({
                'user': [u[0] for _ in test_ucands[u]],
                'item': test_ucands[u],
                'context': [u[1] for _ in test_ucands[u]],
                'rating': [0. for _ in test_ucands[u]],  # fake label, make nonsense
            })
        else:
            tmp = pd.DataFrame({
                'user': [u for _ in test_ucands[u]],
                'item': test_ucands[u],
                'rating': [0. for _ in test_ucands[u]],  # fake label, make nonsense
            })
        tmp_neg_set = sampler.transform(tmp, is_training=False, context=context_flag)
        tmp_dataset = PairData(tmp_neg_set, is_training=False, context=context_flag)
        tmp_loader = data.DataLoader(
            tmp_dataset,
            batch_size=candidates_num,
            shuffle=False,
            num_workers=0
        )
        loaders[u] = tmp_loader

    return loaders, test_ucands


if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()

    # for visualization
    date = datetime.now().strftime('%y%m%d%H%M%S')
    if args.logs:
        if len(args.logsname) == 0:
            string = "reindexed" if  not args.gce else "graph"
            context_folder = "context" if args.context else "no_context"
            loss = 'BPR'
            sampling = 'neg_sampling_each_epoch' if args.neg_sampling_each_epoch else ""
            writer = SummaryWriter(log_dir=f'logs/{args.dataset}/{context_folder}/'
            f'logs_{loss}_lr={args.lr}_DO={args.dropout}_{args.algo_name}_{string}_{args.epochs}epochs_{sampling}_{date}/')
        else:
            writer = SummaryWriter(log_dir=f'logs/{args.dataset}/logs_{args.logsname}_{date}/')
    else:
        writer = SummaryWriter(log_dir=f'logs/nologs/logs/')


    # p = multiprocessing.Pool(args.num_workers)
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
    
    ''' LOAD DATA AND ADD CONTEXT IF NECESSARY '''
    df, users, items = load_rate(args.dataset, args.prepro, context=args.context, gce_flag=args.gce,
                                 cut_down_data=args.cut_down_data)
    df = df.astype(np.int64)
    df['item'] = df['item'] + users
    if args.context:
        df = add_last_clicked_item_context(df, args.dataset)
        # check last number is positive
        assert df['item'].tail().values[-1] > 0

    ''' SPLIT DATA '''
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    train_set, val_set, _ = split_validation(train_set, val_method=args.test_method, list_output=False)

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
    item_pool = set(range(dims[0], dims[1]))
    candidates_num = args.cand_num

    print('='*50, '\n')

    ''' FORMAT DATA AND CHOOSE MODEL '''
    sampler = Sampler(
        dims,
        num_ng=args.num_ng, 
        sample_method=args.sample_method, 
        sample_ratio=args.sample_ratio,
    )
    neg_set, adj_mx = sampler.transform(train_set, is_training=True, context=args.context, pair_pos=None)
    if args.gce:
        # embed()
        if args.mh > 1:
            print(f'[ MULTI HOP {args.mh} ACTIVATED ]')
            adj_mx = adj_mx.__pow__(int(args.mh))
        X = sparse_mx_to_torch_sparse_tensor(identity(adj_mx.shape[0])).to(device)
        # We retrieve the graph's edges and send both them and graph to device in the next two lines
        edge_idx, edge_attr = from_scipy_sparse_matrix(adj_mx)
        # TODO: should I pow the matrix here?
        edge_idx = edge_idx.to(device)

    train_dataset = PairData(train_set, sampler=sampler, adj_mx=adj_mx, is_training=True, context=args.context)

    user_num = dims[0]
    max_dim = dims[2] if args.context else dims[1]
    if args.algo_name == 'mf':
        from model.pair.MFRecommender import PairMF
        model = PairMF(
            user_num,
            max_dim,
            factors=args.factors,
            epochs=args.epochs,
            lr=args.lr,
            reg_1=args.reg_1,
            reg_2=args.reg_2,
            GCE_flag=args.gce,
            X=X if args.gce else None,
            A=edge_idx if args.gce else None,
            gpuid=args.gpu,
            dropout=args.dropout
        )
    elif args.algo_name == 'fm':
        from model.pair.FMRecommender import PairFM
        model = PairFM(
            user_num,
            max_dim,
            factors=args.factors,
            epochs=args.epochs,
            lr=args.lr,
            reg_1=args.reg_1,
            reg_2=args.reg_2,
            GCE_flag=args.gce,
            X=X if args.gce else None,
            A=edge_idx if args.gce else None,
            gpuid=args.gpu,
            dropout=args.dropout
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
                                               context_flag=args.context)
    
    s_time = time.time()
    train(args, model, train_loader, device, args.context, loaders, candidates, val_ur, writer=writer)

    # model.fit(train_loader)
    elapsed_time = time.time() - s_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_{args.problem_type}{args.algo_name}'
                   f'_{args.loss_type}_{args.sample_method}_GCE={args.gce},  {minutes:.2f} min, {seconds:.4f}seconds' + '\n')
    time_log.close()

    print('+'*80)
    ''' TEST METRICS '''
    print('TEST_SET: Start Calculating Metrics......')
    loaders_test, candidates_test = build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num,
                                                         sampler, context_flag=args.context)
    perform_evaluation(loaders_test, candidates_test, model, args, device, test_ur, s_time, minutes_train=minutes,
                       writer=None, seconds_train=seconds)
