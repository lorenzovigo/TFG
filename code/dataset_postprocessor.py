import argparse
import pandas as pd
import utils.request_handler as rh
from utils.actordb import ActorDB
from utils.ml1m_genredb import Ml1M_GenreDB
from tqdm import tqdm
import ast
import shutil

def postprocess_dataset(dataset='ml-100k', min_actor_appearances=10):
    """This method will post-process a dataset: delete actors that aren't relevant enough.

    Parameters
    ----------
    dataset : str, optional
        Dataset to post-process. Needs to have been extended before.
    min_actor_appearances : int, optional
        Number of minimum appearances needed for an actor not to be filtered out the dataset.
    """

    # Read the extended dataset
    if dataset == 'ml-100k':
        extended_dataset = pd.read_csv('../data/online_data/extended-' + dataset + '.csv')

    elif dataset == 'ml-1m':
        extended_dataset = pd.read_csv(f'../data/online_data/extended-' + dataset + '.csv')

    # ActorDB that includes information about the number of appearances by actor
    actor_db = ActorDB(path='../data/online_data/' + dataset + '_artistDB.csv')

    # For every movie
    for index, row in tqdm(extended_dataset.iterrows(), total=extended_dataset.shape[0], position=0,
                           desc='Post-processing'):
        # Filter actors that don't appear at least min_actor_appearances in dataset, and reindex those who do
        new_actors = [actor_db.get_artist_rank_by_id(id) for id in ast.literal_eval(row['actors']) if
                      actor_db.get_artist_appearances_by_id(id) >= min_actor_appearances]
        extended_dataset.at[index, 'actors'] = new_actors

        # Add flag columns to the dataset:
        # Flag 1 is active if there is a top 10 actor in the movie
        # Flag 2, if flag 1 is not active, is active if there is a top 25 actor in the movie
        # Flag 3 is active if neither Flag 1 nor Flag 2 are
        if len(new_actors) == 0:
            extended_dataset.at[index, 'flag1'] = 0
            extended_dataset.at[index, 'flag2'] = 0
            extended_dataset.at[index, 'flag3'] = 0
        else:
            extended_dataset.at[index, 'flag1'] = int(min(new_actors) < 11)
            extended_dataset.at[index, 'flag2'] = int(min(new_actors) > 10 and min(new_actors) < 24)
            extended_dataset.at[index, 'flag3'] = int(min(new_actors) > 25)

    # Save dataset and a copy of the original extended dataset
    shutil.copy2('../data/online_data/extended-' + dataset + '.csv', '../data/online_data/notprocessed-extended-' + dataset + '.csv',)
    extended_dataset.to_csv('../data/online_data/extended-' + dataset + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset postprocesser')

    # other args
    parser.add_argument("--dataset", default='ml-100k', choices=['ml-100k', 'ml-1m'], help="Dataset to extend")
    parser.add_argument("--min_actor_appearances", default=10, type=int, help="Minimum appearances needed by an actor not to be deleted from extended dataset during post-processing")

    args = parser.parse_args()
    postprocess_dataset(args.dataset, args.min_actor_appearances)
