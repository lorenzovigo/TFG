# from torch.utils.data import DataLoader, Dataset
# from IPython import embed
# from sklearn.metrics import roc_auc_score
import pandas as pd
# import numpy as np
# import csv
import os
# import scipy.sparse as sp
# from tqdm import tqdm, trange
import urllib.request
import zipfile

# from torch_geometric.nn import GCNConv

if not os.path.exists('data/ml-100k'):
    # Download dataset
    url = 'https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ'
    print('Downloading Movielens - 100k dataset...')
    urllib.request.urlretrieve(url, 'ml-dataset-splitted.zip')

    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # Extract dataset
    print('Extracting Movielens - 100k dataset...')
    with zipfile.ZipFile('ml-dataset-splitted.zip', 'r') as zip_ref:
        zip_ref.extractall('data/ml-100k')

    # Delete zipfile
    os.remove('ml-dataset-splitted.zip')

# Read data
data = pd.read_csv('data/ml-100k/movielens.train.rating', sep="\t", header=None, names=colnames)
data = data.to_numpy()

test_data = pd.read_csv('data/ml-100k/movielens.test.rating', sep="\t", header=None, names=colnames)
test_Data = test_data.to_numpy()


