import os
import sys
from datetime import datetime
from distutils.util import strtobool
from statistics import mean

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity

from dataset_processors.movielens100k import MovieLens100kDataset
from epochs import test, run
from models.fm import FactorizationMachineModel
from utils import getNDCG, getHitRatio, sparse_mx_to_torch_sparse_tensor

def tensorboard_config(args):
    """Carries out needed tensorboard configuration."""

    # for visualization
    date = datetime.now().strftime('%y%m%d%H%M%S')
    global writer
    if args.logs:
        if len(args.logsname) == 0:
            writer = SummaryWriter(log_dir=f'logs/logs_{date}_{args.model}_{args.gcn}_{args.operation}_{args.reduction}/')
        else:
            writer = SummaryWriter(log_dir=f'logs/logs_{args.logsname}/')
    else:
        writer = SummaryWriter(log_dir=f'logs/undesired/')


def main(args):
    """
    Initializes the training of the model, trains it for a given number of epochs and tests it.
    :param model: Model to be trained.
    :param optimizer: Optimizer to be used in the training.
    :param criterion: Criterion to be used in the training, our loss function.
    :param data_loader: Data Loader created with the dataset.
    :param full_dataset: Full dataset to train with, with test set included.
    :param device: CPU or CUDA, whichever we are going to use.
    :param topk: Number of recommendations to return. (Top k scoredef main(model, optimizer, data_loader, criterion, device, full_dataset, topk=10, epochs=100):
    :param epochs: Number of epochs the model should be trained for.
    """
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    # Initial configurations, incluiding processing the dataset
    tensorboard_config(args)

    if args.dataset == "movielens100k":
        full_dataset = MovieLens100kDataset(add_context=args.add_context,
                                            negative_ratio_train=args.neg_sample_ratio,
                                            negative_ratio_test=args.neg_sample_ratio_test)
    # TODO else:

    # We define our dataloader to generate batches
    data_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Which reduction and operation were chosen?
    reduce_sum = args.reduction != 'mean'
    fm_operation = args.operation != 'dot'

    # We define our tools for prediction: model, criterion and optimizer
    if args.model == 'mf':
        model = 0
    elif args.model == 'fm':
        if args.gcn:
            X = sparse_mx_to_torch_sparse_tensor(identity(full_dataset.train_mat.shape[0]))
            # We retrieve the graph's edges and send both them and graph to device in the next two lines
            edge_idx, edge_attr = from_scipy_sparse_matrix(full_dataset.train_mat)
            model = FactorizationMachineModel(full_dataset.field_dims[-1], 64, gcn=args.gcn, X=X.to(device),
                                              A=edge_idx.to(device), reduce_sum=reduce_sum,
                                              fm_operation=fm_operation).to(device)
        else:
            model = FactorizationMachineModel(full_dataset.field_dims[-1], 32, gcn=args.gcn, reduce_sum=reduce_sum,
                                              fm_operation=fm_operation).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Check our model's performance before training
    hr, ndcg, rmse = test(model, full_dataset.test_set, device, topk=args.top_k)
    print("initial HR: ", hr)
    print("initial NDCG: ", ndcg)
    print("initial RMSE ", rmse)
    writer.add_scalar('initial/HR@{top_k}', hr)
    writer.add_scalar('initial/NDCG@{top_k}', ndcg)
    writer.add_scalar('initial/RMSE@{top_k}', rmse)

    # Training initialization.
    run(model, optimizer, criterion, data_loader, full_dataset, writer, device, tb=args.logs, top_k=args.top_k, epochs=args.epochs)

    writer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="movielens100k", choices=["movielens100k"], help="Dataset to process.")
    parser.add_argument("--add_context", default=False, type=lambda x: bool(strtobool(str(x))), help="Include context in dataset.")
    parser.add_argument("--model", choices=['fm', 'mf'], default='fm', help="Model used to compute predictions.")
    parser.add_argument("--gcn", default=True, type=lambda x: bool(strtobool(str(x))), help="Use GCN in fm model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--top_k", type=int, default=10, help="Top k recommendations to compute.")
    parser.add_argument("--neg_sample_ratio", type=int, default=4, help="Negative sample ratio for training.")
    parser.add_argument("--neg_sample_ratio_test", type=int, default=99, help="Negative sample ratio for testinng.")
    parser.add_argument("--operation", choices=['fm', 'dot'], default='fm')
    parser.add_argument("--reduction", choices=['mean', 'sum'], default='sum')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--logs", default=True, type=lambda x: bool(strtobool(str(x))), help="Enable logging.")
    parser.add_argument("--logsname", default="", type=str, help="Custon log file name.")

    args = parser.parse_args()

    # Printing settings on screen
    print("Chosen configuration:", args) # TODO: hacer más bonito?

    main(args)
