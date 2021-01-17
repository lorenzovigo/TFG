import argparse
import pandas as pd
import utils.request_handler as rh
from utils.actordb import ActorDB
from utils.ml1m_genredb import Ml1M_GenreDB
from tqdm import tqdm

def extend_dataset(api_key, dataset='ml-100k', add_genres=True, add_actors=True):
    if dataset == 'ml-100k':
        df = pd.read_csv('../data/ml-100k/u.item', sep='|', header=None,
                         names=['movie id', 'movie title', 'release date', 'video release date',
                                'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                'Thriller', 'War', 'Western'], encoding = "ISO-8859-1")

        genre_columns = ['Action', 'Adventure', 'Animation',
                     'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']

    elif dataset == 'ml-1m':
        df = pd.read_csv(f'../data/ml-1m/movies.dat', sep='::', header=None,
                         names=['index', 'movie title', 'genres'], engine='python')
        genre_db = Ml1M_GenreDB()

    else:
        raise ValueError('Invalid Dataset Error')

    # names given to columns in extended dataset
    column_names = ['id', 'title', 'year', 'release date', 'genres', 'actors']

    # extended dataset
    downloaded_data = pd.DataFrame(columns=column_names)

    # dataset including information about artists, needed for post_processing
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
        if dataset == 'ml-100k':
            index, title, year = error_control_100k(index, title, year)
        elif dataset == 'ml-1m':
            index, title, year = error_control_1m(index, title, year)

        # process movie genres included in movielens
        if add_genres:
            if dataset == 'ml-100k':
                genre_id = 0
                genre_ids = []
                for genre in genre_columns:
                    if row[genre]:
                        genre_ids.append(genre_id)
                    genre_id += 1
            elif dataset == 'ml-1m':
                genre_ids = [genre_db.push(genre) for genre in row['genres'].split('|')]

        # process actors that take part in movie according to movieDB
        if add_actors:
            if index == -1: # for unknown movies
                    actors = []
            elif index == -2: # error control: some movies are listed as TV shows in movie DB
                actors = rh.getTVActorIds(rh.getTVDBId(title, year, api_key), api_key)
            else: # general case
                actors = rh.getMovieActorIds(rh.getMovieDBId(title, year, api_key), api_key)

            # save relevant information for post_processing in actor database
            for actor in tqdm(actors, desc='Getting movie actors info', position=1):
                actor_db.push_new_appearance(actor, api_key)

        # add row to extended dataset
        rowDF = pd.DataFrame([[index, title, year, row['release date'] if dataset == 'ml-100k' else year, genre_ids, actors]], columns=column_names)
        downloaded_data = downloaded_data.append(rowDF, ignore_index=True)

    # save dataset
    downloaded_data.to_csv('../data/online_data/extended-' + dataset + '.csv')

def error_control_100k(index, title, year):
    if index == 562 or index == 1409:
        index = -2
    if index == 1358:
        index = -1
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
    return index, title, year

def error_control_1m(index, title, year):
    if index in [760, 1091, 1139, 1397, 1400, 1418, 1718, 3491, 3780]:
        index = -1
    if index == 3001:
        title = 'Adventures of Buckaroo Banzai Across The 8th Dimension'
    if index == 3784:
        title = 'Aimee & Jaguar'
    if index == 2810:
        title = 'Armour Of God'
    if index == 3820:
        title = 'Backstage'
    if index == 1650:
        title = 'Bang'
        year = 1995
    if index == 2854:
        title = 'Citizens Band'
    if index == 3693:
        title = 'Daughter of Dr. Jekyll'
    if index == 2106:
        title = 'Deja Vu'
        year = 1997
    if index == 399:
        title = 'Dos Crímenes'
    if index == 2865:
        title = 'El amor brujo'
    if index == 1704:
        title = 'Fallen Angels'
        year = 1998
    if index == 712:
        title = 'Halbmond'
    if index == 2261:
        title = 'Hands on a Hardbody'
        year = 1997
    if index == 3268:
        title = "I'll never forget what's'isname"
    if index == 1133:
        title = 'JLG'
    if index == 1446:
        title = 'Jungle 2 Jungle'
    if index == 988:
        title = "L'Associé"
        year = 1979
    if index == 1383:
        title = "La Ceremonie"
    if index in [72, 1804]:
        title = 'Les Miserables'
    if index == 2707:
        title = 'Marcello Mastroianni: I Remember'
    if index == 2673:
        title = 'Menage'
    if index == 1735:
        title = 'Midaq Alley'
    if index == 1618:
        title = 'Nenette and Boni'
        year = 1997
    if index == 2811:
        title = 'Operation Condor'
    if index in [2985, 3730]:
        title = 'Pokemon'
    if index == 865:
        title = 'Police Story 4'
        year = 1996
    if index == 2317:
        title = 'Ringmaster'
    if index == 1658:
        title = 'Tar'
    if index == 3387:
        title = 'The Colour of Paradise'
    if index == 1864:
        title = "The Life of Emile Zola"
    if index == 1758:
        title = "The Players Club"
    if index == 2375:
        title = "TwentyFourSeven"
    if index == 3508:
        title = 'Two Moon Junction'
    if index == 684:
        title = 'Under the Domim Tree'
    if index == 1732:
        title = 'U.S. Marshals'
    if index == 1328:
        title = "Vampire in Venice"
        year = 1988
    if index == 861:
        title = "Vive L'Amour"
        year = 1995
    if index == 1733:
        title = "Welcome to Woop-Woop"
    if index == 2776:
        title = 'Whiteboyz'
    if index == 781:
        year = ''
    if index == 3240:
        year = 1918
    if index in [1941, 2158]:
        year = 1927
    if index == 2033:
        year = 1928
    if index == 2144:
        year = 1934
    if index == 2867:
        year = 1941
    if index == 3307:
        year = 1942
    if index == 1140:
        year = 1944
    if index in [1347, 2779, 3160]:
        year = 1951
    if index == 958:
        year = 1953
    if index == 3271:
        year = 1955
    if index == 3419:
        year = 1958
    if index in [1855, 2450]:
        year = 1959
    if index in [830, 2230]:
        year = 1960
    if index in [2663, 3514]:
        year = 1962
    if index in [740, 2485]:
        year = 1964
    if index in [3095, 3267]:
        year = 1966
    if index in [1148, 2231]:
        year = 1967
    if index == 2853:
        year = 1968
    if index == 1971:
        year = 1969
    if index == 3703:
        year = 1970
    if index in [3147, 3279]:
        year = 1971
    if index == 2156:
        year = 1972
    if index == 2852:
        year = 1973
    if index in [611, 2170, 3865]:
        year = 1974
    if index in [1109, 1120, 3165, 3401, 3561, 3777]:
        year = 1975
    if index == 1945:
        year = 1976
    if index == 2094:
        year = 1978
    if index in [1114, 3772]:
        year = 1981
    if index in [2443, 3492]:
        year = 1983
    if index == 3624:
        year = 1984
    if index == 3100:
        year = 1985
    if index in [2685, 2810]:
        year = 1986
    if index in [2671, 3581]:
        year = 1987
    if index in [2903, 3002]:
        year = 1988
    if index == 3360:
        year = 1989
    if index in [1205, 1920, 3395]:
        year = 1990
    if index in [1175, 2811, 3695]:
        year = 1991
    if index in [119, 569, 3659]:
        year = 1992
    if index in [281, 369, 384, 535, 561, 579, 651, 742, 1057]:
        year = 1993
    if index in [266, 282, 319, 342, 414, 445, 541, 671, 833, 2023]:
        year = 1994
    if index in [126, 155, 181, 227, 245, 247, 252, 264, 308, 400, 402, 600, 643, 660, 669, 689, 721, 746, 832, 852, 973, 1094, 1126, 1133, 1399, 1419, 1667, 2034, 2447, 2540, 2996]:
        year = 1995
    if index in [32, 82, 127, 288, 618, 661, 666, 678, 804, 805, 815, 854, 863, 1092, 1102, 1288, 1299, 1508, 1515, 1645, 1762, 1961, 2205, 2393, 2495]:
        year = 1996
    if index in [55, 1150, 1435, 1520, 1613, 1662, 1671, 1723, 1724, 1783, 1799, 1830, 2088, 2415, 3562, 3797]:
        year = 1997
    if index in [1587, 1610, 1692, 1722, 1730, 1733, 1753, 1782, 1784, 1832, 1836, 1960, 2208, 2223, 2250, 2911, 2988, 3015, 3727]:
        year = 1998
    if index in [1689, 2417, 2519, 2535, 2542, 2646, 2840, 2844, 2889, 2929, 3173, 3258, 3526, 3540, 3578]:
        year = 1999
    if index in [2700, 3108, 3120, 3157, 3159, 3210, 3234, 3249, 3388, 3471, 3498, 3652, 3653, 3783, 3790, 3837]:
        year = 2000
    if index == 3252:
        year = 2001
    return index, title, year

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset extender')

    # required args
    parser.add_argument('--api_key', help='MovieDB API Key needed', required=True)

    # other args
    parser.add_argument("--dataset", default='ml-100k', choices=['ml-100k', 'ml-1m'], help="Dataset to extend")
    parser.add_argument("--no_genres", default=False, action='store_true', help="Skip adding genres to extended dataset")
    parser.add_argument("--no_actors", default=False, action='store_true', help="Skin adding actors to extended dataset")

    args = parser.parse_args()
    extend_dataset(args.api_key, args.dataset, not args.no_genres, not args.no_actors)

