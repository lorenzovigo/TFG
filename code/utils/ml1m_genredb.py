import pandas as pd


class Ml1M_GenreDB():
    """This class handles the inclusion of genres in the dataset for MovieLens-1M"""

    def __init__(self):
        # Saves the id the next genre will get assigned
        self.next_id = 0

        self.column_names = ['id', 'genre']
        # Dataframe that relates ids and genres
        self.df = pd.DataFrame(columns=self.column_names)

    def get_row_by_id(self, id):
        """ Gets a row of the dataset with a given id.

        Parameters
        ----------
        id : int
            id to be found in the genre mapping dataset.
        Returns
        -------
        pandas.Series
            The dataset row with the given id. None if not found.
        """
        if id in self.df['id'].values:
            return self.df.loc[self.df['id'] == id]
        else:
            return None

    def get_row_by_genre(self, genre):
        """Gets a row with a given genre

        Parameters
        ----------
        genre : str
            genre to be found in the genre mapping dataset.
        Returns
        -------
        pandas.Series
            The dataset row with the given genre. None if not found.
        """
        if genre in self.df['genre'].values:
            return self.df.loc[self.df['genre'] == genre]
        else:
            return None

    def get_genre_name_by_id(self, id):
        """Gets the genre name given the genre id

        Parameters
        ----------
        id : int
            Genre id.
        Returns
        -------
        str
            Genre name. None if not found.
        """
        genre = self.get_row_by_id(id)
        if genre is None:
            return None
        else:
            return genre['genre'][0]

    def get_id_by_genre(self, genre):
        """Gets the genre id given the genre name

        Parameters
        ----------
        genre : str
            Genre name.
        Returns
        -------
        int
            Genre id. None if not found.
        """
        genre = self.get_row_by_genre(genre)
        if genre is None:
            return None
        else:
            return genre['id'][0]

    def push(self, genre_name):
        """Adds a new genre to the dataset if it is not already included

        Parameters
        ----------
        genre_name : str
            Genre name to look for and add to in the dataset.
        Returns
        -------
        int
            Genre id, the id it just got assigned or the one it already had if it was already present in the dataset.
        """
        id = self.get_id_by_genre(genre_name)
        if id is None:
            self.df = self.df.append(pd.DataFrame([[self.next_id, genre_name]], columns=self.column_names))
            self.next_id += 1
            self.df.to_csv('../data/online_data/ml-1m_genreDB.csv')
            return self.next_id - 1
        else:
            return id
