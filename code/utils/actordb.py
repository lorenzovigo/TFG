import os
import pandas as pd
import utils.request_handler as rh


class ActorDB():
    # TODO doc
    def __init__(self, dataset='ml-100k'):
        self.column_names = ['id', 'name', 'appearances']
        # TODO delete or consider
        # if 'artistDB.csv' in os.listdir("../../data/online_data"):
        #    self.df = pd.read_csv('../../data/online_data/artistDB.csv', sep=',', header=None,
        #                          names=self.column_names, encoding = "ISO-8859-1")
        # else:
        self.dataset = dataset
        self.df = pd.DataFrame(columns=self.column_names)

    def get_artist_by_id(self, id):
        if id in self.df['id'].values:
            return self.df.loc[self.df['id'] == id]
        else:
            return None

    def get_artist_name_by_id(self, id):
        artist = self.get_artist_by_id(id)
        if artist is None:
            return None
        else:
            return artist['name'][0]

    def get_artist_appearances_by_id(self, id):
        artist = self.get_artist_by_id(id)
        if artist is None:
            return None
        else:
            return artist['appearances'][0]

    def push_new_appearance(self, id, api_key):
        artist = self.get_artist_by_id(id)
        if artist is None:
            name = rh.getActorName(id, api_key)
            self.df = self.df.append(pd.DataFrame([[id, name, 1]], columns=self.column_names))
        else:
            self.df.loc[self.df['id'] == id, ['appearances']] = self.get_artist_appearances_by_id(id) + 1
        self.df.to_csv('../data/online_data/' + self.dataset + '_artistDB.csv')