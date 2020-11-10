import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
import os, urllib, zipfile
from IPython import embed
import torch


class MovieLens100kDataset(torch.utils.data.Dataset):
    """
    Dataset processor for MovieLens - 100k with

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

    def __init__(self, negative_ratio_train=4, negative_ratio_test=99, sep='\t', add_context=False):
        """
        Executes the whole pipeline, from downloading to fully processing the dataset.

        Parameters
        ----------
        negative_ratio_train : int, optional
            Number of negative samples generated for every interaction in the dataset.

        negative_ratio_test : int, optional
            Number of negative samples generated for every interaction in the test set.

        sep : str, optional
            Separator used in the dataset.
        """
        # TODO: Change dataset
        self.load_dataset()

        colnames = ["user_id", 'item_id', 'label', 'timestamp']

        # Read several data from dataset files
        self.data_pd = pd.read_csv(f'{self.dataset_path}movielens.train.rating', names=["user_id", "item_id", "rating", "timestamp"], sep="\t", usecols=[0, 1, 2], engine="python")
        self.data = pd.read_csv(f'{self.dataset_path}movielens.train.rating', sep=sep, header=None, names=colnames).to_numpy()
        self.test_data = pd.read_csv(f'{self.dataset_path}movielens.test.rating', sep=sep, header=None, names=colnames).to_numpy()

        # Define targets (known interactions) and items (users and movies) taken from the dataset
        self.targets = self.data[:, 2]
        self.dataset = self.preprocess_dataset(self.data) # previously named items

        if add_context:
             self.dataset = self.add_last_clicked_item(self.dataset)

        # We get our adjacency matrix dimension (max id) and build the matrix
        self.field_dims = np.cumsum(np.max(self.dataset, axis=0) - np.min(self.dataset, axis = 0) + 1)
        self.field_mins = np.min(self.dataset, axis=0)
        self.max_users, self.max_items = self.field_dims[:2]
        self.train_mat = utils.build_adjacency_matrix(self.field_dims, self.field_mins, self.dataset.copy())

        # Generate train interactions with 4 negative samples for each positive
        self.negative_sampling(neg_ratio=negative_ratio_train)

        # We define the test set items as we did with the dataset and generate test negative samples
        test_set_items = self.preprocess_dataset(self.test_data)
        self.test_set = utils.build_test_set(test_set_items, self.max_users, self.max_items, self.train_mat, neg_ratio=negative_ratio_test)

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
        Gives the items new indexes so that they have unique ids.

        Parameters
        ----------
        data : DataFrame
            Items to be preprocessed.

        Returns
        -------
        ndarray
            Preprocessed items.
        """

        # Whole method is dataset-specific. Data should be in this order: user, item, labels, context1, context2 ...
        reindexed_items = np.delete(data, 2, axis=1).astype(np.int)  # We delete the label column but keep the rest
        reindexed_items = reindexed_items[:, :2].astype(np.int) - 1  # IDs start by 1 and we want them to start by 0
        # We get the highest user ID and highest movie ID
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

    def negative_sampling(self, neg_ratio=4):
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
        data = np.c_[(self.dataset, self.targets)].astype(int)

        for x in tqdm(data, desc="Performing negative sampling on test data..."):  # x are triplets (u, i , 1)
            # Append positive interaction
            self.interactions.append(x)
            # Copy user and maintain last position to 0. Now we will need to update neg_triplet[1]
            neg_triplet = np.vstack([x, ] * neg_ratio)
            neg_triplet[:, 2] = np.zeros(neg_ratio)
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
