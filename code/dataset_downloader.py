import argparse
import os
import requests
import zipfile


def download_dataset(dataset='ml-100k'):
    print("Downloading dataset: " + dataset + ".")

    data_path = '../data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if dataset == 'ml-100k':
        r = requests.get(url="http://files.grouplens.org/datasets/movielens/ml-100k.zip")
    else:
        r = requests.get(url="http://files.grouplens.org/datasets/movielens/ml-1m.zip")

    open(data_path + dataset + '.zip', 'wb').write(r.content)

    with zipfile.ZipFile(data_path + dataset + '.zip', 'r') as zip_ref:
        zip_ref.extractall(data_path + '/')

    os.remove(data_path + dataset + '.zip')

    print("Dataset " + dataset + " downloaded.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset postprocesser')

    # other args
    parser.add_argument("--dataset", default='ml-100k', choices=['ml-100k', 'ml-1m'], help="Dataset to download")

    args = parser.parse_args()
    download_dataset(args.dataset)