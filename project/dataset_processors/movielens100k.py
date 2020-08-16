import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
import scipy.sparse as sp
from tqdm import tqdm

import torch


class MovieLens100kDataset(torch.utils.data.Dataset):

    data_dir = 'data'
    dataset_path = f'{data_dir}/ml-100k/ml-dataset-splitted/movielens'
    url = 'https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ'
    downloaded_file = 'ml-dataset-splitted.zip'
    dataset_name = 'Movielens - 100k'

    def __init__(self, negative_ratio_train=4, negative_ratio_test=99, sep='\t'):
        # Download dataset
        self.load_dataset()

        colnames = ["user_id", 'item_id', 'label', 'timestamp'] # dataset-specific

        # Read several data from dataset files (filenames are dataset-specific)
        self.data = pd.read_csv(f'{self.dataset_path}.train.rating', sep=sep, header=None, names=colnames).to_numpy()
        self.test_data = pd.read_csv(f'{self.dataset_path}.test.rating', sep=sep, header=None, names=colnames).to_numpy()

        # Define targets (known interactions) and items (users and movies) taken from the dataset
        self.targets = self.data[:, 2] # dataset-specific
        self.items = self.preprocess_items(self.data)

        # We get our adjacency matrix dimension (max id) and build the matrix
        self.field_dims = np.max(self.items, axis=0) + 1
        self.max_users, self.max_items = self.field_dims
        max_id = np.max(self.field_dims)
        self.train_mat = self.build_adjacency_matrix(max_id, self.items.copy())

        # Generate train interactions with 4 negative samples for each positive
        self.negative_sampling(self.items, neg_ratio=negative_ratio_train)

        # Build test set by passing as input the test item interactions
        test_set_items = self.preprocess_items(self.test_data)
        self.test_set = self.build_test_set(test_set_items, neg_ratio=negative_ratio_test)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.interactions[index]

    def load_dataset(self):
        # Check whether dataset is already downloaded or not
        if not os.path.exists(self.dataset_path):
            # Download dataset'
            print(f'Downloading {self.dataset_name} dataset...')
            urllib.request.urlretrieve(self.url, self.downloaded_file)

            # Create data folder if it doesn't exist
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)

            # Extract dataset
            print(f'Extracting {self.dataset_name} dataset...')
            with zipfile.ZipFile(self.downloaded_file, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)

            # Delete zipfile
            os.remove(self.downloaded_file)

    def preprocess_items(self, data):
        # Whole method is dataset-specific
        reindexed_items = data[:, :2].astype(np.int) - 1  # IDs start by 1 and we want them to start by 0
        # We get the highest user ID and highest movie ID
        users, items = np.max(reindexed_items, axis=0)[:2] + 1
        # We want IDs to be unique among users and movies, so we reindex movies
        reindexed_items[:, 1] = reindexed_items[:, 1] + users

        return reindexed_items

    def build_adjacency_matrix(self, dims, interactions):
        train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
        # For every interaction, we set the value as 1 in both grids that represent the pair of items that interact
        for x in tqdm(interactions, desc="Building Adjacency Matrix..."):
            train_mat[x[0], x[1]] = 1.0
            train_mat[x[1], x[0]] = 1.0

        return train_mat

    def negative_sampling(self, items, neg_ratio):
        # We initialize an array where interactions will be saved
        self.interactions = []
        # We put together the item pairs with their respective targets
        data = np.c_[(items, self.targets)].astype(int)

        for x in tqdm(data, desc="Performing negative sampling on test data..."):  # x are triplets (u, i , 1)
            # Append positive interaction
            self.interactions.append(x)
            # Copy user and maintain last position to 0. Now we will need to update neg_triplet[1]
            neg_triplet = np.vstack([x, ] * neg_ratio)
            neg_triplet[:, 2] = np.zeros(neg_ratio)

            # We take the ids of items the user has not interacted with
            possible_negatives = np.where(self.train_mat[x[0]].toarray()[0] == 0)[0]

            # Then, we filter those items in order to keep only the movies the user has not interacted with
            possible_negatives = possible_negatives[self.max_users <= possible_negatives]
            possible_negatives = possible_negatives[possible_negatives < self.max_items]

            # TODO tengo una duda aquí (1 de agosto)
            # We replace neg_triplet[1] with one of these movies in each negative sample
            neg_triplet[:, 1] = np.random.choice(possible_negatives, neg_ratio, replace=False)
            self.interactions.append(neg_triplet.copy())

        self.interactions = np.vstack(self.interactions)

    def build_test_set(self, gt_test_interactions, neg_ratio):
        # We initialize an array where the test set will be saved
        test_set = []

        for pair in tqdm(gt_test_interactions, desc="Building test set..."):
            # We take the ids of items the user has not interacted with
            possible_negatives = np.where(self.train_mat[pair[0]].toarray()[0] == 0)[0]

            # Then, we filter those items in order to keep only the movies the user has not interacted with
            # TODO duda: en el código original comprueba que j no sea igual que el movie del pair (por qué? en negative sampling no se evitan repeticiones y esto no lo evita del todo)
            possible_negatives = possible_negatives[self.max_users <= possible_negatives]
            possible_negatives = possible_negatives[possible_negatives < self.max_items]
            possible_negatives = possible_negatives[possible_negatives != pair[1]]

            # We replace neg_triplet[1] with one of these movies in each negative sample
            negatives = np.random.choice(possible_negatives, neg_ratio, replace=False)

            # TODO esto se podría hacer con el esquema anterior
            # We append pair and the negatives we generated to the test set
            single_user_test_set = np.vstack([pair, ] * (len(negatives)+1))
            single_user_test_set[:, 1][1:] = negatives

            test_set.append(single_user_test_set.copy())
        return test_set

