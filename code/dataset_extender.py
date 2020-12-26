import argparse
import pandas as pd
import utils.request_handler as rh
from utils.actordb import ActorDB
from tqdm import tqdm

# TODO implementar post_processing
def extend_dataset(api_key, dataset='ml-100k', add_genres=True, add_actors=True, post_processing=True, min_actor_appearances=10):
    df = pd.read_csv('../data/' + dataset + '/u.item', sep='|', header=None,
                         names=['movie id', 'movie title', 'release date', 'video release date',
                                'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                'Thriller', 'War', 'Western'], encoding = "ISO-8859-1")

    genre_columns = ['Action', 'Adventure', 'Animation',
                     'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']

    # names given to columns in extended dataset
    column_names = ['id', 'title', 'year', 'release date', 'genres', 'actors']

    # extended dataset
    downloaded_data = pd.DataFrame(columns=column_names)

    if post_processing and add_actors: # dataset including information about artists, only needed if post_processing is activated
        actor_db = ActorDB(dataset=dataset)

    for index, row in tqdm(df.iterrows(), total=df.shape[0], position=0, desc='Getting movies info'):

        # consider a title finished when first dot or parenthesis is found
        fp = row['movie title'].find(' (')
        fc = row['movie title'].find(',')
        if fp == -1 or fc == -1:
            eotitle = max(fp, fc)
        else:
            eotitle = min(fp, fc)

        # split title and year (release year and title year might not be the same, and both of them might be wrong)
        title = row['movie title'][:eotitle]
        year = row['movie title'][-5:-1]

        # error control
        title, year = error_control(index, title, year)

        # process movie genres included in movielens
        if add_genres:
            genre_id = 0
            genre_ids = []
            for genre in genre_columns:
                if row[genre]:
                    genre_ids.append(genre_id)
                genre_id += 1

        # process actors that take part in movie according to movieDB
        if add_actors:
            if index != 562 and index != 1358 and index != 1409:
                actors = rh.getMovieActorIds(rh.getMovieDBId(title, year, api_key), api_key)
            elif index == 562 or index == 1409: # error control: some movies are listed as TV shows in movie DB
                actors = rh.getTVActorIds(rh.getTVDBId(title, year, api_key), api_key)
            else: # for unknown movies
                actors = []

            if post_processing: # save relevant information for post_processing in actor database
                for actor in tqdm(actors, desc='Getting movie actors info', position=1):
                    actor_db.push_new_appearance(actor, api_key)

        # add row to extended dataset
        rowDF = pd.DataFrame([[index, title, year, row['release date'], genre_ids, actors]], columns=column_names)
        downloaded_data = downloaded_data.append(rowDF, ignore_index=True)

    # filter actors that don't appear at least min_actor_appearances in dataset
    if post_processing:
        for index, row in tqdm(downloaded_data.iterrows(), total=downloaded_data.shape[0], position=0, desc='Post-processing'):
            downloaded_data.at[index, 'actors'] = [id for id in row['actors'] if actor_db.get_artist_appearances_by_id(id) >= min_actor_appearances]

    # save dataset
    downloaded_data.to_csv('../data/online_data/extended-' + dataset + '.csv')

def error_control(index, title, year):
    # ml-100k
    if index == 1234:
        title = 'Bang'
        year = 1995
    if index == 1404:
        title = 'Boys Life 2'
    if index == 1632:
        title = 'Cold Fever'
        year = 1995
    if index == 473:
        title = 'Dr. Strangelove'
        year = 1964
    if index == 1590:
        title = 'Fallen Angels'
        year = 1998
    if index == 1419:
        year = 2001
        title = 'Gilligans Island'
    if index == 1585:
        title = 'Hard Boiled'
    if index == 242:
        title = 'Jungle 2 Jungle'
    if index == 1420:
        title = 'Mi Vida Loca'
        year = 1994
    if index == 598:
        title = 'Police Story 4'
        year = 1996
    if index == 599:
        title = 'Robinson Crusoe'
    if index == 1018:
        title = 'The Killer'
    if index == 562:
        title = 'The Langoliers'
    if index == 911:
        title = 'U.S. Marshals'
    if index == 1535:
        title = "Vive L'Amour"
        year = 1995
    if index == 1409:
        title = 'Where I Live'
        year = 1993
    if index == 1582:
        title = 'Zaproszenie'
    if index == 1330:
        year = ''
    if index == 1368:
        year = 1951
    if index == 1455:
        year = 1953
    if index == 850:
        year = 1967
    if index == 1265:
        year = 1974
    if index == 162 or index == 167:
        year = 1975
    if index == 678:
        year = 1982
    if index == 860:
        year = 1988
    if index == 188:
        year = 1990
    if index == 644:
        year = 1991
    if index == 1191 or index == 1401:
        year = 1992
    if index in [465, 1055, 1303, 1356, 1357, 1402, 1556, 1567]:
        year = 1993
    if index in [389, 783, 788, 798, 962, 1152, 1343, 1538]:
        year = 1994
    if index in [43, 74, 375, 376, 718, 772, 787, 867, 1079, 1129, 1177, 1235, 1305, 1306, 1328, 1483, 1558, 1599, 1618]:
        year = 1995
    if index in [278, 829, 838, 1014, 1085, 1086, 1228, 1258, 1492, 1504, 1514, 1659]:
        year = 1996
    if index in [252, 896, 972, 1394, 1497, 1606, 1655, 1666, 1670]:
        year = 1997
    if index in [1025, 1293, 1363, 1393]:
        year = 1998
    if index == 1432 or index == 1591:
        year = 1999
    return title, year

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset extender')

    # required args
    parser.add_argument('--api_key', help='MovieDB API Key needed', required=True)

    # other args
    parser.add_argument("--dataset", default='ml-100k', choices=['ml-100k', 'ml-1m'], help="Dataset to extend")
    parser.add_argument("--genres", default=True, action='store_false', help="Add genres to extended dataset")
    parser.add_argument("--actors", default=True, action='store_false', help="Add actors to extended dataset")
    parser.add_argument("--post_processing", default=True, action='store_false', help="Delete marginal actors from dataset")
    parser.add_argument("--min_actor_appearances", default=10, type=int, help="Minimum appearances needed by an actor not to be deleted from extended dataset during post-processing")

    args = parser.parse_args()
    extend_dataset(args.api_key, args.dataset, args.genres, args.actors, args.post_processing, args.min_actor_appearances)

