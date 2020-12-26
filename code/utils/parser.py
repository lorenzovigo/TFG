import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='test recommender')
    # common settings
    # python main.py --algo_name mf --dataset ml-100k  --epochs 10 --gce
    parser.add_argument('--tune_epochs',
                        type=int,
                        default=30,
                        help='tuning epochs')
    parser.add_argument("--logs", action="store_true", default=True, help="Enables logs")
    parser.add_argument("--not_early_stopping", action="store_true", default=False, help="Enables not doing early stopping")
    parser.add_argument("--logsname", default="", help="Enables logs")
    parser.add_argument('--reindex',
                        action='store_false',
                        default=True,
                        help='activate if do not want to reindex items')
    parser.add_argument('--neg_sampling_each_epoch',
                        action='store_true',
                        default=False,
                        help='activate if we want to perform neg_sampling in each epoch')
    parser.add_argument('--context',
                        action='store_false',
                        default=True,
                        help='activate if do not want to add context')
    parser.add_argument('--gce',
                        action='store_true',
                        default=False,
                        help='activate to use GCE layer instead of current embbedding layer')
    # parser.add_argument('--reg2',
    #                     action='store_true',
    #                     default=False,
    #                     help='activate to use regularizations')
    parser.add_argument('--cut_down_data',
                        action='store_true',
                        default=False,
                        help='activate to use half interactions per user --> reduce dataset size')
    parser.add_argument('--mf',
                        action='store_true',
                        default=False,
                        help='activate to use MF in NFM ')
    parser.add_argument('--mh',
                        type=int,
                        default=1,
                        help='HOPS TO ENABLE -- MULTI HOP FUNCTION')
    parser.add_argument('--problem_type', 
                        type=str, 
                        default='pair',
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
                        default=21,
                        help='top number of recommend list')
    parser.add_argument('--test_method', 
                        type=str, 
                        default='tloo',
                        help='method for split test,options: ufo/loo/fo(split by ratio)/tfo/tloo')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tloo',
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--test_size', 
                        type=float, 
                        default=0.2,
                        help='split ratio for test set')
    parser.add_argument('--val_size', 
                        type=float, 
                        default=0.1, help='split ratio for validation set')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
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
                        default='BPR',
                        help='loss function type: BPR/CL')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='type of optimizer: SGD /adam')
    # algo settings
    parser.add_argument('--factors', 
                        type=int, 
                        default=64,
                        help='latent factors numbers in the model')
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
                        # type=float,
                        default=0,
                        # default=0.5,
                        help='dropout rate')
    parser.add_argument('--lr', 
                        default=0.001,
                        help='learning rate')
    # parser.add_argument('--lr',
    #                     type=float,
    #                     default=0.001,
    #                     help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50,
                        help='training epochs')
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help='num_workers')
    parser.add_argument('--batch_size',
                        default=256,
                        help='batch size for training')
    # parser.add_argument('--batch_size',
    #                     type=int,
    #                     default=256,
    #                     help='batch size for training')
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

    args = parser.parse_args()

    return args
