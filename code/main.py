import time
import numpy as np
import pandas as pd
import ast
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
from scipy.sparse import identity, lil_matrix, hstack


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
    # reindex items and context if needed to guarantee different ids
    df['item'] = df['item'] + users
    if args.context:
        df = add_last_clicked_item_context(df, args.dataset)
        # check last number is positive
        assert df['item'].tail().values[-1] > 0
        df['context'] = df['context'] + items

    ''' SPLIT DATA '''
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    train_set, val_set, _ = split_validation(train_set, val_method=args.test_method, list_output=False)

    df = pd.concat([train_set, test_set], ignore_index=True)
    dims = np.max(df.to_numpy().astype(int), axis=0) + 1

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
    # negative sampling and adjacency matrix construction
    neg_set, adj_mx = sampler.transform(train_set, is_training=True, context=args.context, pair_pos=None)

    # create graph needed structure if it is activated
    if args.gce:
        # embed()
        if args.mh > 1:
            print(f'[ MULTI HOP {args.mh} ACTIVATED ]')
            adj_mx = adj_mx.__pow__(int(args.mh))
        X = identity(adj_mx.shape[0])

        # we may add side-information to X if activated
        if args.genres or args.actors:
            extended_dataset = pd.read_csv('../data/online_data/extended-' + args.dataset + '.csv', sep=',')
            movie_id_mapping = train_set[['item', 'original item id']].copy().drop_duplicates().sort_values(by=['item']).reset_index(drop=True)

        if args.genres:
            # Check how many genres there are. Since we know we saved genres in order in every movie we can do this:
            num_genres = max([sublist[-1] if sublist else -1 for sublist in extended_dataset['genres'].apply(ast.literal_eval).tolist()]) + 1
            # If we can't guarantee genres are in order, use this: (less efficient)
            # num_genres = max([x for sublist in extended_dataset['genres'].apply(ast.literal_eval).tolist() for x in sublist])

            # Add genre information to a new matrix and then concatenate with X
            genre_extension_matrix = lil_matrix((adj_mx.shape[0], num_genres), dtype=np.int8)
            for index, row in tqdm(movie_id_mapping.iterrows(), desc='Adding genres'):
                genres = extended_dataset.loc[extended_dataset['id'] == row['original item id'] - 1, 'genres'].reset_index(drop=True)
                if not genres.empty:
                    for genre in ast.literal_eval(genres[0]):
                        genre_extension_matrix[row['item'] + 1, genre] = 1
            X = hstack([X, genre_extension_matrix])

        if args.actors:
            # Check how many genres there are. Since we know we saved genres in order in every movie we can do this:
            num_actors = max([sublist[-1] if sublist else -1 for sublist in extended_dataset['actors'].apply(ast.literal_eval).tolist()]) + 1
            print(num_actors)
            # Add genre information to a new matrix and then concatenate with X
            actor_extension_matrix = lil_matrix((adj_mx.shape[0], num_actors + 3), dtype=np.int8)
            for index, row in tqdm(movie_id_mapping.iterrows(), desc='Adding actors'):
                actors = extended_dataset.loc[extended_dataset['id'] == row['original item id'] - 1, 'actors'].reset_index(drop=True)
                if not actors.empty:
                    for actor in ast.literal_eval(actors[0]):
                        actor_extension_matrix[row['item'] + 1, actor] = 1
                for i in range(1, 4):
                    flag = extended_dataset.loc[extended_dataset['id'] == row['original item id'] - 1, 'flag' + str(i)].reset_index(drop=True)
                    if not flag.empty:
                        actor_extension_matrix[row['item'], i*-1] = flag[0]
            X = hstack([X, actor_extension_matrix])

        X = X.transpose()

        # We retrieve the graph's edges and send both them and graph to device in the next two lines

        X = sparse_mx_to_torch_sparse_tensor(X).to(device)
        edge_idx, edge_attr = from_scipy_sparse_matrix(adj_mx)
        edge_idx = edge_idx.to(device)

    # these columns are not neeeded anymore
    train_set.drop(['original item id'], axis=1)
    test_set.drop(['original item id'], axis=1)
    val_set.drop(['original item id'], axis=1)

    # create training set
    train_dataset = PairData(train_set, sampler=sampler, adj_mx=adj_mx, is_training=True, context=args.context)

    user_num = dims[0]
    max_dim = dims[2] if args.context else dims[1]
    # choose model
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

    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_pair{args.algo_name}'
                   f'_BPR_{args.sample_method}_GCE={args.gce},  {minutes:.2f} min, {seconds:.4f}seconds' + '\n')
    time_log.close()

    print('+'*80)
    ''' TEST METRICS '''
    print('TEST_SET: Start Calculating Metrics......')
    loaders_test, candidates_test = build_evaluation_set(test_ur, total_train_ur, item_pool, candidates_num,
                                                         sampler, context_flag=args.context)
    perform_evaluation(loaders_test, candidates_test, model, args, device, test_ur, s_time, minutes_train=minutes,
                       writer=None, seconds_train=seconds)
