import os
import pickle

import utils.movieDB_requests as mdbr

base_dir = '../data/online_data/'

# general behavior:
# check if request was already called previously in some kind of cache saved in base_dir
# if it was already called, we retrieve the information offline
# if it wasn't, we create the cache, call the API online and save the information we get


def getMovieDBId(title, year, api_key):
    file = title.replace("/", " ") + " " + str(year)

    if os.path.exists(base_dir + 'movies/ids'):
        if '{}.bin'.format(file) in os.listdir(base_dir + 'movies/ids'):
            with open(base_dir + 'movies/ids/{}.bin'.format(file), 'rb') as pickle_file:
                return pickle.load(pickle_file)
    else:
        os.makedirs(base_dir + 'movies/ids')

    id = mdbr.getMovieDBId(title, year, api_key)
    with open(base_dir + 'movies/ids/{}.bin'.format(file), 'wb') as pickle_file:
        pickle.dump(id, pickle_file)
    return id


def getTVDBId(title, year, api_key):
    file = title.replace("/", " ") + " " + str(year)

    if os.path.exists(base_dir + 'tv/ids'):
        if '{}.bin'.format(file) in os.listdir(base_dir + 'tv/ids'):
            with open(base_dir + 'tv/ids/{}.bin'.format(file), 'rb') as pickle_file:
                return pickle.load(pickle_file)
    else:
        os.makedirs(base_dir + 'tv/ids')

    id = mdbr.getTVDBId(title, year, api_key)
    with open(base_dir + 'tv/ids/{}.bin'.format(file), 'wb') as pickle_file:
        pickle.dump(id, pickle_file)
    return id


def getMovieActorIds(movieDBId, api_key):
    if os.path.exists(base_dir + 'movies/actors'):
        if '{}.bin'.format(movieDBId) in os.listdir(base_dir + 'movies/actors'):
            with open(base_dir + 'movies/actors/{}.bin'.format(movieDBId), 'rb') as pickle_file:
                return pickle.load(pickle_file)
    else:
        os.makedirs(base_dir + 'movies/actors')

    ids = mdbr.getMovieActorIds(movieDBId, api_key)
    with open(base_dir + 'movies/actors/{}.bin'.format(movieDBId), 'wb') as pickle_file:
        pickle.dump(ids, pickle_file)
    return ids


def getTVActorIds(TVDBId, api_key):
    if os.path.exists(base_dir + 'tv/actors'):
        if '{}.bin'.format(TVDBId) in os.listdir(base_dir + 'tv/actors'):
            with open(base_dir + 'tv/actors/{}.bin'.format(TVDBId), 'rb') as pickle_file:
                return pickle.load(pickle_file)
    else:
        os.makedirs(base_dir + 'tv/actors')

    ids = mdbr.getTVActorIds(TVDBId, api_key)
    with open(base_dir + 'tv/actors/{}.bin'.format(TVDBId), 'wb') as pickle_file:
        pickle.dump(ids, pickle_file)
    return ids


def getActorName(movieDBId, api_key):
    if os.path.exists(base_dir + 'people/actors'):
        if '{}.bin'.format(movieDBId) in os.listdir(base_dir + 'people/actors'):
            with open(base_dir + 'people/actors/{}.bin'.format(movieDBId), 'rb') as pickle_file:
                return pickle.load(pickle_file)
    else:
        os.makedirs(base_dir + 'people/actors')

    name = mdbr.getActorName(movieDBId, api_key)
    with open(base_dir + 'people/actors/{}.bin'.format(movieDBId), 'wb') as pickle_file:
        pickle.dump(name, pickle_file)
    return name
