import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
import scipy.sparse as sp
from tqdm import tqdm

import torch


class MovieLens100kDataset(torch.utils.data.Dataset):
    """
    Dataset processor for MovieLens - 100k

    Downloading and processing of the MovieLens dataset found online, with 100k rows.
    Items will be processed and we will perform negative sampling and build the test set.


    Attributes
    ----------
    data_dir : str
        Folder where all downloaded datasets should be found.
    dataset_path : str
        Folder where MovieLens - 100k will be saved.
    url : str
        Dataset download url.
    downloaded_file : str
        Name the downloaded file will have.
    dataset_name : str
        Dataset's name.
    negative_ratio_train : int
        How many negative samples we will produce for every positive sample in the dataset.
    negative_ratio_test : int
        How many negative samples we will produce for every positive sample in the test set.
    sep : str
        Separator used in dataset.

    Methods
    -------
    __init__(self, negative_ratio_train=4, negative_ratio_test=99, sep='\t')
        Executes the whole pipeline, from downloading to fully processing the dataset.

    __len__(self)
        Returns the number of interactions in the dataset after negative sampling.

    __getitem__(self, index)
        Retrieves an item from the dataset given an index, after the negative sampling.

    load_dataset(self)
        Handles the dataset downloading.

    preprocess_items(self, data)
        Makes sure every Id is unique, even if the items are different entities.

    build_adjacency_matrix(self, dims, interactions):
        Builds the adjacency matrix given the interactions between the items.

    negative_sampling(self, items, neg_ratio)
        Generates negative samples in the dataset.

    build_test_set(self, gt_test_interactions, neg_ratio)
        Generates negative samples for the test set.
    """

    data_dir = 'data'
    extract_path = f'{data_dir}/ml-100k'
    dataset_path = f'{data_dir}/ml-100k/ml-dataset-splitted/'
    url = 'https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ'
    downloaded_file = 'ml-dataset-splitted.zip'
    dataset_name = 'Movielens - 100k'

    def __init__(self, negative_ratio_train=4, negative_ratio_test=99, sep='\t'):
        """
        Executes the whole pipeline, from downloading to fully processing the dataset.
        :param negative_ratio_train: Number of negative samples generated for every interaction in the dataset.
        :param negative_ratio_test: Number of negative samples generated for every interaction in the test set.
        :param sep: Separator used in the dataset.
        """
        # Download dataset
        self.load_dataset()

        colnames = ["user_id", 'item_id', 'label', 'timestamp']

        # Read several data from dataset files
        self.data = pd.read_csv(f'{self.dataset_path}movielens.train.rating', sep=sep, header=None, names=colnames).to_numpy()
        self.test_data = pd.read_csv(f'{self.dataset_path}movielens.test.rating', sep=sep, header=None, names=colnames).to_numpy()

        # Define targets (known interactions) and items (users and movies) taken from the dataset
        self.targets = self.data[:, 2]
        self.items = self.preprocess_items(self.data)

        # We get our adjacency matrix dimension (max id) and build the matrix
        self.field_dims = np.max(self.items, axis=0) + 1
        self.max_users, self.max_items = self.field_dims
        max_id = np.max(self.field_dims)
        self.train_mat = self.build_adjacency_matrix(max_id, self.items.copy())

        # Generate train interactions with 4 negative samples for each positive
        self.negative_sampling(self.items, neg_ratio=negative_ratio_train)

        # We define the test set items as we did with the dataset and generate test negative samples
        test_set_items = self.preprocess_items(self.test_data)
        self.test_set = self.build_test_set(test_set_items, neg_ratio=negative_ratio_test)

    def __len__(self):
        """
        Returns the number of interactions in the dataset after negative sampling.
        :return: Number of interactions in the dataset after negative sampling.
        """
        return self.targets.shape[0]

    def __getitem__(self, index):
        """
        Retrieves an interaction from the dataset given an index.
        :param index: Index of the item to be retrieved.
        :return: Interaction with the given index.
        """
        return self.interactions[index]

    def load_dataset(self):
        """Downloads and extracts the MovieLens - 100k dataset zip file."""
        # Check whether dataset is already downloaded or not
        if not os.path.exists(self.dataset_path):
            # Download dataset
            print(f'Downloading {self.dataset_name} dataset...')
            urllib.request.urlretrieve(self.url, self.downloaded_file)

            # Create data folder if it doesn't exist
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)

            # Extract dataset
            print(f'Extracting {self.dataset_name} dataset...')
            with zipfile.ZipFile(self.downloaded_file, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)

            # Delete zipfile
            os.remove(self.downloaded_file)

    def preprocess_items(self, data):
        """
        Gives the items new indexes so that they have unique ids.

        :param data: Items to be preprocessed.
        :return: Preprocessed items.
        """

        # Whole method is dataset-specific
        reindexed_items = data[:, :2].astype(np.int) - 1  # IDs start by 1 and we want them to start by 0
        # We get the highest user ID and highest movie ID
        users, items = np.max(reindexed_items, axis=0)[:2] + 1
        # We want IDs to be unique among users and movies, so we reindex movies
        reindexed_items[:, 1] = reindexed_items[:, 1] + users

        return reindexed_items

    def build_adjacency_matrix(self, dims, interactions):
        """
        Builds the adjacency matrix determined by a set of interactions.
        :param dims: The dimension the adjacency matrix should have.
        :param interactions: Set of known interactions.
        :return: Fully built adjacency matrix.
        """
        train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
        # For every interaction, we set the value as 1 in both grids that represent the pair of items that interact
        for x in tqdm(interactions, desc="Building Adjacency Matrix..."):
            train_mat[x[0], x[1]] = 1.0
            train_mat[x[1], x[0]] = 1.0

        return train_mat

    # TODO: podríamos unificar los dos métodos siguientes (?). En el segundo no se usan los targets.

    def negative_sampling(self, items, neg_ratio):
        """
        Every known interaction is considered a positive sample.
        This method generates random negative samples from items that have not interacted with each other.
        :param items: Preprocessed dataset items.
        :param neg_ratio: How many negative samples we will produce for every positive sample in the dataset.
        """
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
            used_j = []

        # TODO duda evitar js repetidos es correcto?
            for idx in range(neg_ratio):
                j = np.random.randint(self.max_users, self.max_items)
                # IDEA: Loop to exclude true interactions (set to 1 in adj_train) user - item
                while (x[0], j) in self.train_mat and j not in used_j:
                    j = np.random.randint(self.max_users, self.max_items)
                neg_triplet[:, 1][idx] = j
                used_j.append(j)
            self.interactions.append(neg_triplet.copy())

        self.interactions = np.vstack(self.interactions)

    def build_test_set(self, gt_test_interactions, neg_ratio):
        """
        Every known interaction is considered a positive sample.
        This method generates random negative samples from items that have not interacted with each other.
        :param gt_test_interactions: Preprocessed test set items.
        :param neg_ratio: How many negative samples we will produce for every positive sample in the test set.
        :return: Complete test set with both negative and positive samples.
        """
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

