# Checking problematic imports
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.utils.tensorboard import SummaryWriter
from dataset_processors.movielens100k import MovieLens100kDataset
import os


# Needed tensorboard configuration
def tensorboard_config():
    logs_base_dir = "runs"
    os.makedirs(logs_base_dir, exist_ok=True)

    tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
    tb_gcn = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN/')


tensorboard_config()
full_dataset = MovieLens100kDataset()

