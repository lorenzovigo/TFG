import os
import sys
from distutils.util import strtobool
from statistics import mean

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity

from dataset_processors.movielens100k import MovieLens100kDataset
from models.fm import FactorizationMachineModel
from models.fm_gcn import FactorizationMachineModel_withGCN
from utils import getNDCG, getHitRatio, sparse_mx_to_torch_sparse_tensor

def tensorboard_config():
    """Carries out needed tensorboard configuration."""
    logs_base_dir = "runs"
    os.makedirs(logs_base_dir, exist_ok=True)

    global tb_fm, tb_gcn
    tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
    tb_gcn = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN/')


def train(model, optimizer, criterion, data_loader, device, log_interval=100):
    """
    One epoch training for our given model.
    :param model: Model to be trained.
    :param optimizer: Optimizer to be used in the training.
    :param data_loader: Data loader to be used in the training.
    :param criterion: Criterion to be used in the training, our loss function.
    :param device: Device to be used.
    :return: Average loss computed among all interactions.
    """
    # We tell our model we are training it.
    model.train()
    data_loader.dataset.negative_sampling()
    total_loss = []

    for i, (interactions) in enumerate(data_loader):
        model.zero_grad()

        # We calculate our predictions and the loss value between them and targets
        interactions = interactions.to(device)
        targets = interactions[:, 2]
        predictions = model(interactions[:, :2])

        # We seek to optimize the loss function and do so with the given optimizer
        loss = criterion(predictions, targets.float())
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def test(model, test_set, device, topk=10):
    """
    Carries out the testing of a model.
    :param model: Model to be tested.
    :param test_set: Set with which we will test our model.
    :param device: Device to be used.
    :param topk: Number of recommendations to return. (Top k scores)
    :return: Two metric values (Hit Ratio and NDGC) of the model.
    """
    # Test the HR and NDCG for the model @topK
    # We tell our model we are testing it.
    model.eval()

    HR, NDCG = [], []

    for user_test in test_set:
        # For each user in the test set we get the ground truth item
        gt_item = user_test[0][1]

        # We compute the predictions using our model and retrieve our recommendations (those with best score)
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        # We compute the hit ratio and NDCG
        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
    return mean(HR), mean(NDCG)

def main(model, optimizer, criterion, data_loader, full_dataset, device, topk=10, epochs=100):
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
    tb = False
    for epoch_i in range(epochs):
        # We train our model in every epoch and compute our metrics afterwards.
        # TODO data_loader.dataset.negative_sampling() dentro de train o aquí, donde es correcto?
        train_loss = train(model, optimizer, criterion, data_loader, device)
        hr, ndcg = test(model, full_dataset.test_set, device, topk=topk)

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')
        print('\n')
        if tb:
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="movielens100k", choices=["movielens100k"], help="Dataset to process.")
    parser.add_argument("--add_context", default=False, type=lambda x: bool(strtobool(str(x))), help="Include context in dataset.")
    parser.add_argument("--model", choices=['fm', 'fm_gcn', 'mf'], default='fm_gcn', help="Model used to compute predictions.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--top_k", type=int, default=10, help="Top k recommendations to compute.")
    parser.add_argument("--neg_sample_ratio", type=int, default=4, help="Negative sample ratio for training.")
    parser.add_argument("--neg_sample_ratio_test", type=int, default=99, help="Negative sample ratio for testinng.")
    parser.add_argument("--operation", choices=['fm', 'dot'], default='fm')
    parser.add_argument("--reduction", choices=['mean', 'sum'], default='sum')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    # Printing settings on screen
    print("Chosen configuration:", args) # TODO: hacer más bonito?

    # Initial configurations, incluiding processing the dataset
    # TODO: tensorboard_config()
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
        model = FactorizationMachineModel(full_dataset.field_dims[-1], 32, reduce_sum=reduce_sum, fm_operation=fm_operation).to(device)
    else:
        X = sparse_mx_to_torch_sparse_tensor(identity(full_dataset.train_mat.shape[0]))
        # We retrieve the graph's edges and send both them and graph to device in the next two lines
        edge_idx, edge_attr = from_scipy_sparse_matrix(full_dataset.train_mat)
        model = FactorizationMachineModel_withGCN(full_dataset.field_dims[-1], 64, X.to(device), edge_idx.to(device), reduce_sum=reduce_sum, fm_operation=fm_operation).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean') # TODO sum or mean aquí?
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Check our model's performance before training
    hr, ndcg = test(model, full_dataset.test_set, device, topk=args.top_k)
    print("initial HR: ", hr)
    print("initial NDCG: ", ndcg)

    # Training initialization.
    main(model, optimizer, criterion, data_loader, full_dataset, device, topk=args.top_k, epochs=args.epochs)

# TODO tengo que implementar los logs de tensorboard aquí "½tensorboard --logdir runs"

    # tb_fm.close()
    # tb_gcn.close()
