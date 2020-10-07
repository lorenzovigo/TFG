from statistics import mean

import torch
from scipy.sparse import identity
from torch.utils.data import DataLoader

from models.fmm import FactorizationMachineModel
from models.fmm_gcn import FactorizationMachineModel_withGCN
from utils import getNDCG, getHitRatio, sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.utils.tensorboard import SummaryWriter
from dataset_processors.movielens100k import MovieLens100kDataset
from dataset_processors.movielens100k_withcontext import MovieLens100kDataset_WithContext
import os

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

def tensorboard_config():
    """Carries out needed tensorboard configuration."""
    logs_base_dir = "runs"
    os.makedirs(logs_base_dir, exist_ok=True)

    global tb_fm, tb_gcn
    tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
    tb_gcn = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN/')


def train_one_epoch(model, optimizer, data_loader, criterion, device, log_interval=100):
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
    total_loss = []

    for i, (interactions) in enumerate(data_loader):
        # We calculate our predictions and the loss value between them and targets
        interactions = interactions.to(device)
        targets = interactions[:, 2]
        predictions = model(interactions[:, :2])

        # We seek to optimize the loss function and do so with the given optimizer
        loss = criterion(predictions, targets.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def test(model, full_dataset, device, topk=10):
    """
    Carries out the testing of a model.
    :param model: Model to be tested.
    :param full_dataset: Dataset with which we will test our model.
    :param device: Device to be used.
    :param topk: Number of recommendations to return. (Top k scores)
    :return: Two metric values (Hit Ratio and NDGC) of the model.
    """
    # Test the HR and NDCG for the model @topK
    # We tell our model we are testing it.
    model.eval()

    HR, NDCG = [], []

    for user_test in full_dataset.test_set:
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


def train_and_test(model, optimizer, criterion, topk=10, epochs=30):
    """
    Trains the model for a given number of epochs and tests it afterwards.
    :param model: Model to be trained.
    :param optimizer: Optimizer to be used in the training.
    :param criterion: Criterion to be used in the training, our loss function.
    :param topk: Number of recommendations to return. (Top k scores)
    :param epochs: Number of epochs the model should be trained for.
    """
    tb = True
    for epoch_i in range(epochs):
        # We train our model in every epoch and compute our metrics afterwards.
        # data_loader.dataset.negative_sampling()
        train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)
        hr, ndcg = test(model, full_dataset, device, topk=topk)

        print('\n')

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')
        print('\n')
        if tb:
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)


# Initial configurations, incluiding processing the dataset
tensorboard_config()
# full_dataset = MovieLens100kDataset()
full_dataset = MovieLens100kDataset_WithContext()

# We define our dataloader to generate batches
data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

# Needed only for GCN
X = sparse_mx_to_torch_sparse_tensor(identity(full_dataset.train_mat.shape[0]))
edge_idx, edge_attr = from_scipy_sparse_matrix(full_dataset.train_mat)

# We define our tools for prediction: model, criterion and optimizer
# model = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)
model = FactorizationMachineModel_withGCN(full_dataset.field_dims[-1], 64, X.to(device), edge_idx.to(device)).to(device)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Number of recommendations we want to retrieve for each user
topk = 10

# Check our model's performance before training
hr, ndcg = test(model, full_dataset, device, topk=topk)
print("initial HR: ", hr)
print("initial NDCG: ", ndcg)

train_and_test(model, optimizer, criterion)

# TODO tengo que implementar los logs de tensorboard aquí "½tensorboard --logdir runs"

tb_fm.close()
tb_gcn.close()
