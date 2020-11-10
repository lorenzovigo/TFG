import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
import scipy.sparse as sp
from tqdm import tqdm

import torch


class MovieLens100kDataset_WithContext(torch.utils.data.Dataset):
    """
    Dataset processor for MovieLens - 100k

    Downloading and processing of the MovieLens dataset found online, with 100k rows.
    Items will be processed and we will perform negative sampling and build the test set.

    Could be applied to other datasets as long as the columns follow this specefic order:
    users, movies, labels, context1, context2...

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
    # TODO añadir los self.

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

    preprocess_dataset(self, data)
        Deletes the target column and makes sure the item ids are unique.

    add_last_clicked_item(self, dataset)
        Uses the timestamp information to add the Last Clicked Item context to the dataset.

    build_adjacency_matrix(self, dims, interactions):
        Builds the adjacency matrix given the interactions between the items.

    preprocess_test_items(self, data)
        Makes sure every item in the test set has unique ids, even if the items are different entities.

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

        Parameters
        ----------
        negative_ratio_train : int, optional
        Number of negative samples generated for every interaction in the dataset.

        negative_ratio_test : int, optional
        Number of negative samples generated for every interaction in the test set.

        sep: str, optional
        Separator used in the dataset.
        """

        # Download dataset
        self.load_dataset()

        colnames = ["user_id", 'item_id', 'label', 'timestamp'] # TODO generalizar

        # Read several data from dataset files
        self.data = pd.read_csv(f'{self.dataset_path}movielens.train.rating', sep=sep, header=None, names=colnames).to_numpy()
        self.test_data = pd.read_csv(f'{self.dataset_path}movielens.test.rating', sep=sep, header=None, names=colnames).to_numpy()

        # Define targets (known interactions) and dataset (items [users, movies] and context) taken from the dataset
        self.targets = self.data[:, 2]
        self.dataset = self.preprocess_dataset(self.data)

        print(self.dataset.shape)
        print("Taking context into account...")
        print(self.dataset.shape)
        self.dataset = self.add_last_clicked_item(self.dataset)

        # We get our adjacency matrix dimension (max id) and build the matrix
        self.field_dims = np.cumsum(np.max(self.dataset, axis=0) - np.min(self.dataset, axis = 0) + 1)
        self.field_mins = np.min(self.dataset, axis=0)
        self.max_users, self.max_items = self.field_dims[:2]  # TODO: generalizar
        self.train_mat = self.build_adjacency_matrix(self.field_dims[-1], self.dataset.copy())

        # Generate train interactions with 4 negative samples for each positive
        self.negative_sampling(self.dataset, neg_ratio=negative_ratio_train)

        # We define the test set items as we did with the dataset and generate test negative samples
        test_dataset = self.preprocess_test_items(self.test_data)
        self.test_set = self.build_test_set(test_dataset, neg_ratio=negative_ratio_test)

    def __len__(self):
        """
        Returns the number of interactions in the dataset after negative sampling.

        Returns
        -------
        int
            Number of interactions in the dataset after negative sampling.
        """
        return self.targets.shape[0]

    def __getitem__(self, index):
        """
        Retrieves an interaction from the dataset given an index.

        Parameters
        ----------
        index : int
            Index of the item to be retrieved.

        Returns
        -------
        tuple
            Interaction with the given index.
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

    def preprocess_dataset(self, data):
        """
        Deletes the target column and makes sure the item ids are unique.
    
        Parameters
        ----------
        data : DataFrame
            The whole dataset to be processed, including targets in third column and context and metadata.

        Returns
        -------
        ndarray
            Dataset as a numpy array with unique ids for every item and without target column.
        """
        # Whole method is dataset-specific. Data should be in this order: user, item, labels, context1, context2 ...
        reindexed_items = np.delete(data, 2, axis=1).astype(np.int)  # We delete the label column but keep the rest
        reindexed_items[:, :2] = reindexed_items[:, :2] - 1  # IDs start by 1 and we want them to start by 0
        # We get the highest user ID
        users = np.max(reindexed_items, axis=0)[:1]
        # We want IDs to be unique among users and movies, so we reindex movies
        reindexed_items[:, 1] = reindexed_items[:, 1] + users + 1

        return reindexed_items

    def add_last_clicked_item(self, dataset):
        """
        Uses the timestamp information to add the Last Clicked Item context to the dataset.

        Parameters
        ----------
        dataset : ndarray
            Dataset with timestamp as last column # TODO no sería mejor como primer contexto?

        Returns
        -------
        ndarray
            Dataset with the Last Clicked Item context added.
        """
        # We sort the dataset by the timestamp. TODO duda: no tiene en cuenta el usuario o sí?
        sorted_data = dataset[dataset[:, -1].argsort()]
        # We add as last clicked item the item of the previous interaction. Except for the first interaction that gets a 1 TODO duda: why?
        # We don't use the last interaction's movie since there is no following interaction.
        sorted_data[:, -1] = np.concatenate(([1], sorted_data[:-1][:, 1]))

        return sorted_data

    def build_adjacency_matrix(self, dims, interactions):
        """
        Builds the adjacency matrix determined by a set of interactions with contexts.

        Parameters
        ----------
        dims : int
            The dimension the adjacency matrix should have.

        interactions : list
            Set of known interactions.

        Returns
        -------
        scipy.dok_matrix
            Fully built adjacency matrix.J
        """
        train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
        aux = np.delete(np.insert(self.field_dims, 0, 0), -1)
        # For every interaction, we set the value as 1 in both grids that represent the pair of items that interact
        for x in tqdm(interactions, desc="Building Adjacency Matrix..."):
            # We fix the indices for the values of the context
            indices = x - self.field_mins + aux
            # We fill the interactions
            train_mat[indices[0], indices[1]] = 1.0
            train_mat[indices[1], indices[0]] = 1.0
            # We fill the positions for the context
            for i in range(2, np.shape(indices)[0]):
                train_mat[indices[0], indices[i]] = 1.0
                train_mat[indices[1], indices[i]] = 1.0

        return train_mat

    def negative_sampling(self, items, neg_ratio):
        """
        Every known interaction is considered a positive sample.
        This method generates random negative samples from items that have not interacted with each other.

        Parameters
        ----------
        items : ndarray
            Preprocessed dataset items.

        neg_ratio : int
            How many negative samples we will produce for every positive sample in the dataset.
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
            neg_triplet[:, -1] = np.zeros(neg_ratio)
            used_js = []

            for idx in range(neg_ratio):
                j = np.random.randint(self.max_users, self.max_items)
                # IDEA: Loop to exclude true interactions (set to 1 in adj_train) user - item
                while (x[0], j) in self.train_mat or j in used_js:
                    j = np.random.randint(self.max_users, self.max_items)
                neg_triplet[:, 1][idx] = j
                used_js.append(j)
            self.interactions.append(neg_triplet.copy())

        self.interactions = np.vstack(self.interactions)

    def preprocess_test_items(self, data): # TODO adaptar a contexto
        """
        Gives the test items new indexes so that they have unique ids.
        :param data: Test items to be preprocessed.
        :return: Preprocessed items.
        """

        # Whole method is dataset-specific
        reindexed_items = data[:, :2].astype(np.int) - 1  # IDs start by 1 and we want them to start by 0
        # We get the highest user ID and highest movie ID
        users, items = np.max(reindexed_items, axis=0)[:2] + 1
        # We want IDs to be unique among users and movies, so we reindex movies
        reindexed_items[:, 1] = reindexed_items[:, 1] + users

        return reindexed_items

    def build_test_set(self, gt_test_interactions, neg_ratio):
        """
        Every known interaction is considered a positive sample.
        This method generates random negative samples from items that have not interacted with each other.

        Parameters
        ----------
        gt_test_interactions : ndarray
            Ground truth items which should be retrieved by the model from the set created in this method.

        neg_ratio : int
            How many negative samples we will produce for every positive sample in the test set.

        Returns
        -------
        list
            Complete test set with both negative and positive samples.
        """
        # We initialize an array where the test set will be saved
        test_set = []

        for pair in tqdm(gt_test_interactions, desc="Building test set..."):
            negatives = []
            for t in range(neg_ratio):
                j = np.random.randint(self.max_users, self.max_items)
                while (pair[0], j) in self.train_mat or j == pair[1] or j in negatives:
                    j = np.random.randint(self.max_users, self.max_items)
                negatives.append(j)
            # APPEND TEST SETS FOR SINGLE USER
            single_user_test_set = np.vstack([pair, ] * (len(negatives) + 1))
            single_user_test_set[:, 1][1:] = negatives
            test_set.append(single_user_test_set.copy())
        return test_set