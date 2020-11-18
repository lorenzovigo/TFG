import argparse

from hyperopt import hp


def parse_args(tune=False):
    parser = argparse.ArgumentParser(description='test recommender')
    # common settings
    parser.add_argument('--context',
                        action='store_true',
                        default=False,
                        help='activate context aware model')
    parser.add_argument('--gce',
                        action='store_true',
                        default=False,
                        help='activate to use GCE layer instead of current embbedding layer')
    parser.add_argument('--problem_type',
                        type=str, 
                        default='point',
                        help='pair-wise or point-wise')
    parser.add_argument('--algo_name', 
                        type=str, 
                        default='mf',
                        help='algorithm to choose')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k',
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='10filter',
                        help='dataset preprocess op.: origin/Ncore/filter')
    parser.add_argument('--topk', 
                        type=int, 
                        default=10,
                        help='top number of recommend list')
    parser.add_argument('--test_size', 
                        type=float, 
                        default=0.2,
                        help='split ratio for test set')
    parser.add_argument('--val_size', 
                        type=float, 
                        default=0.1, help='split ratio for validation set')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=99,
                        help='No. of candidates item for predict')
    parser.add_argument('--sample_method', 
                        type=str, 
                        default='uniform', 
                        help='negative sampling method mixed with uniform, options: item-ascd, item-desc')
    parser.add_argument('--sample_ratio', 
                        type=float, 
                        default=1,
                        help='mix sample method ratio, 0 for all uniform')
    parser.add_argument('--init_method', 
                        type=str, 
                        default='', 
                        help='weight initialization method')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=4,
                        help='negative sampling number')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default='CL',
                        help='loss function type: BPR/CL')
    # algo settings
    parser.add_argument('--reg_1', 
                        type=float, 
                        # default=0.001,
                        default=0,
                        help='L1 regularization')
    parser.add_argument('--reg_2', 
                        type=float, 
                        # default=0.001,
                        default=0.01,
                        help='L2 regularization')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0,
                        # default=0.5,
                        help='dropout rate')
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help='num_workers')

    parser.add_argument('--num_layers', 
                        type=int, 
                        default=1,
                        help='number of layers in MLP model')
    parser.add_argument('--act_func', 
                        type=str, 
                        default='relu', 
                        help='activation method in interio layers')
    parser.add_argument('--out_func', 
                        type=str, 
                        default='sigmoid', 
                        help='activation method in output layers')
    parser.add_argument('--no_batch_norm', 
                        action='store_false', 
                        default=True, 
                        help='whether do batch normalization in interior layers')

    # not used when tuning
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='batch size for training')
    parser.add_argument('--factors',
                        type=int,
                        default=64,
                        help='latent factors numbers in the model')

    parser.add_argument('--tune',
                        action='store_true',
                        default=False,
                        help='activate to tune using Bayersian HyperOpt')


    args = parser.parse_args()

    return args


def parse_space(args, tune=False):
    if tune:
        space = {
            'context': args.context,
            'gce': args.gce,
            'problem_type': args.problem_type,
            'algo_name': args.algo_name,
            'dataset': args.dataset,
            'prepro': args.prepro,
            'topk': args.topk,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'cand_num': args.cand_num,
            'sample_method': args.sample_method,
            'sample_ratio': args.sample_ratio,
            'init_method': args.init_method,
            'gpu': args.gpu,
            'num_ng': args.num_ng,
            'loss_type': args.loss_type,
            'reg_1': args.reg_1,
            'reg_2': args.reg_2,
            'dropout': args.dropout,
            'num_workers': args.num_workers,
            'num_layers': args.num_layers,
            'act_func': args.act_func,
            'out_func': args.out_func,
            'no_batch_norm': args.no_batch_norm,
            'lr': hp.choice('lr', [0.05, 0.01, 0.001]),
            'epochs': hp.choice('epochs', [10, 20, 40, 50, 70, 100, 120, 150, 180, 200]),
            'factors': hp.choice('factors', [16, 32, 64, 128]),
            'batch_size': hp.choice('batch_size', [256, 512]),
            'tune': tune
        }
    else:
        space = {
            'context': args.context,
            'gce': args.gce,
            'problem_type': args.problem_type,
            'algo_name': args.algo_name,
            'dataset': args.dataset,
            'prepro': args.prepro,
            'topk': args.topk,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'cand_num': args.cand_num,
            'sample_method': args.sample_method,
            'sample_ratio': args.sample_ratio,
            'init_method': args.init_method,
            'gpu': args.gpu,
            'num_ng': args.num_ng,
            'loss_type': args.loss_type,
            'reg_1': args.reg_1,
            'reg_2': args.reg_2,
            'dropout': args.dropout,
            'num_workers': args.num_workers,
            'num_layers': args.num_layers,
            'act_func': args.act_func,
            'out_func': args.out_func,
            'no_batch_norm': args.no_batch_norm,
            'lr': args.lr,
            'epochs': args.epochs,
            'factors': args.factors,
            'batch_size': args.batch_size,
            'tune': tune
        }
    return space
