from statistics import mean

import torch
from scipy.sparse import identity
from torch.utils.data import DataLoader

from models.fmm import FactorizationMachineModel
from models.fmm_gcn import FactorizationMachineModel_withGCN
from utils import getNDCG, getHitRatio, sparse_mx_to_torch_sparse_tensor
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.utils.tensorboard import SummaryWriter
from dataset_processors.movielens100k import MovieLens100kDataset
import os

device = "cpu"


# Needed tensorboard configuration
def tensorboard_config():
    logs_base_dir = "runs"
    os.makedirs(logs_base_dir, exist_ok=True)

    global tb_fm, tb_gcn
    tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
    tb_gcn = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN/')


def train_one_epoch(model, optimizer, data_loader, criterion, device, log_interval=100):
    # TODO duda: qué hace el model.train() si lo entrenamos luego en el bucle
    model.train()
    total_loss = []

    # TODO duda: repasar qué hace cada cosa en cada momento para ver si lo he entendido bien
    for i, (interactions) in enumerate(data_loader):
        interactions = interactions.to(device)
        targets = interactions[:, 2]
        predictions = model(interactions[:, :2])

        loss = criterion(predictions, targets.float())
        model.zero_grad() # TODO duda: esto es porque las gradientes se iban acumulando, no?
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)

def test(model, full_dataset, device, topk=10):
    # Test the HR and NDCG for the model @topK
    # TODO qué hace eval si lo hacemos luego
    model.eval()

    HR, NDCG = [], []

    for user_test in full_dataset.test_set:
        # For each user in the test set we get the gt_item # TODO duda: no sé qué es
        gt_item = user_test[0][1]

        # We compute the predictions using our model and retrieve our recommendations (those with best score)
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        # We compute the hit ratio and NDCG
        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
    return mean(HR), mean(NDCG)


def train(model, optimizer, criterion, topk = 10, epochs = 30):
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
full_dataset = MovieLens100kDataset()

# TODO duda: entiendo que divide el dataset en batches, había varios imports posibles
data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

# Needed only for GCN, TODO duda: construcción del grafo
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

train(model, optimizer, criterion)

# TODO tengo que implementar los logs de tensorboard aquí "½tensorboard --logdir runs"

tb_fm.close()
tb_gcn.close()

