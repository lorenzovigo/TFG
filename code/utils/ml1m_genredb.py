import os
import pandas as pd
import utils.request_handler as rh


class Ml1M_GenreDB():
    # TODO doc
    def __init__(self):
        self.next_id = 0
        self.column_names = ['id', 'genre']
        self.df = pd.DataFrame(columns=self.column_names)
        # TODO if they don't exist: os.makedirs('../data/online_data')

    def get_row_by_id(self, id):
        if id in self.df['id'].values:
            return self.df.loc[self.df['id'] == id]
        else:
            return None

    def get_row_by_genre(self, genre):
        if genre in self.df['genre'].values:
            return self.df.loc[self.df['genre'] == genre]
        else:
            return None

    def get_genre_name_by_id(self, id):
        genre = self.get_row_by_id(id)
        if genre is None:
            return None
        else:
            return genre['genre'][0]

    def get_id_by_genre(self, genre):
        genre = self.get_row_by_genre(genre)
        if genre is None:
            return None
        else:
            return genre['id'][0]

    def push(self, genre_name):
        id = self.get_id_by_genre(genre_name)
        if id is None:
            self.df = self.df.append(pd.DataFrame([[self.next_id, genre_name]], columns=self.column_names))
            self.next_id += 1
            self.df.to_csv('../data/online_data/ml-1m_genreDB.csv')
            return self.next_id - 1
        else:
            return id
