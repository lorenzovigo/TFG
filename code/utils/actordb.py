import os
import pandas as pd
import utils.request_handler as rh
from tqdm import tqdm

class ActorDB():
    def __init__(self, dataset='ml-100k', path=None):
        self.column_names = ['id', 'name', 'appearances']
        if not path is None:
            self.df = pd.read_csv(path)
        else:
            self.dataset = dataset
            self.df = pd.DataFrame(columns=self.column_names)
        self.sorted = False

    def get_artist_by_id(self, id):
        if id in self.df['id'].values:
            return self.df.loc[self.df['id'] == id]
        else:
            return None

    def get_artist_appearances_by_id(self, id):
        artist = self.get_artist_by_id(id)
        if artist is None:
            return None
        else:
            return artist.iloc[0]['appearances']

    def get_artist_rank_by_id(self, id):
        if not self.sorted:
            self.df = self.df.sort_values(by=['appearances'], ascending=False)
            self.df = self.df.reset_index(drop=True)
            self.sorted = True
        return self.df[self.df['id'] == id].index.tolist()[0] + 1

    def push_new_appearance(self, id, api_key):
        artist = self.get_artist_by_id(id)
        if artist is None:
            name = rh.getActorName(id, api_key)
            self.df = self.df.append(pd.DataFrame([[id, name, 1]], columns=self.column_names), ignore_index=True)
        else:
            self.df.loc[self.df['id'] == id, ['appearances']] = self.get_artist_appearances_by_id(id) + 1
        self.df.to_csv('../data/online_data/' + self.dataset + '_artistDB.csv')
        self.sorted = False