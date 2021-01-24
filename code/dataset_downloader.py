import argparse
import os
import requests
import zipfile


def download_dataset(dataset='ml-100k'):
    """Method that downloads the selected dataset

    Parameters
    ----------
    dataset : str
        Dataset name to download
    """

    print("Downloading dataset: " + dataset + ".")

    data_path = '../data/'
    # Create folders if they don't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Download corresponding datasets
    if dataset == 'ml-100k':
        r = requests.get(url="http://files.grouplens.org/datasets/movielens/ml-100k.zip")
    else:
        r = requests.get(url="http://files.grouplens.org/datasets/movielens/ml-1m.zip")

    # Save zip file
    open(data_path + dataset + '.zip', 'wb').write(r.content)

    # Extract zip file in data folder
    with zipfile.ZipFile(data_path + dataset + '.zip', 'r') as zip_ref:
        zip_ref.extractall(data_path + '/')

    # Delete zip file
    os.remove(data_path + dataset + '.zip')

    print("Dataset " + dataset + " downloaded.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset postprocesser')

    #available arguments
    parser.add_argument("--dataset", default='ml-100k', choices=['ml-100k', 'ml-1m'], help="Dataset to download")

    args = parser.parse_args()
    download_dataset(args.dataset)