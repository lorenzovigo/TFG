import pandas as pd
import utils.request_handler as rh

class ActorDB():
    """Class that keeps the information of all the actors included in the dataset"""
    def __init__(self, dataset='ml-100k', path=None):
        self.column_names = ['id', 'name', 'appearances']

        # Creates a new dataset if it does not exist already, or loads it if it is present.
        if not path is None:
            self.df = pd.read_csv(path)
        else:
            self.dataset = dataset
            self.df = pd.DataFrame(columns=self.column_names)
        # Shows if the dataset is sorted by number of appearances in the dataset.
        self.sorted = False

    def get_artist_by_id(self, id):
        """Gets the artist with given id.

        Parameters
        ----------
        id : int
            Artist id
        Returns
        -------
        pandas.Series
            Row with said artist id. None, if not found.
        """
        if id in self.df['id'].values:
            return self.df.loc[self.df['id'] == id]
        else:
            return None

    def get_artist_appearances_by_id(self, id):
        """Get the number of appearances of an artist in the dataste.

        Parameters
        ----------
        id : int
            Artist id.
        Returns
        -------
        int
            Number of appearances of the artist with given id. None if said artist does not exist.
        """
        artist = self.get_artist_by_id(id)
        if artist is None:
            return None
        else:
            return artist.iloc[0]['appearances']

    def get_artist_rank_by_id(self, id):
        """Gets artist ranking sorting by number of appearances.

        Parameters
        ----------
        id : int
            Artist id.
        Returns
        -------
        int
            Artist position in appearances ranking.
        """
        if not self.sorted:
            self.df = self.df.sort_values(by=['appearances'], ascending=False)
            self.df = self.df.reset_index(drop=True)
            self.sorted = True
        return self.df[self.df['id'] == id].index.tolist()[0] + 1

    def push_new_appearance(self, id, api_key):
        """Handles adding a new artist appearance in the dataset. If it is the first appearance, it will add the artist to the dataset.

        Parameters
        ----------
        id : int
            Artist id.
        api_key : str
            MovieDB API Key to download the actor online information.
        """
        artist = self.get_artist_by_id(id)

        if artist is None:
            # If the artist is not found already in teh dataset, actor information is downloaded and it is added to the dataset.
            name = rh.getActorName(id, api_key)
            self.df = self.df.append(pd.DataFrame([[id, name, 1]], columns=self.column_names), ignore_index=True)
        else:
            # If the artist is already included in the dataset, an appearance is added to their role
            self.df.loc[self.df['id'] == id, ['appearances']] = self.get_artist_appearances_by_id(id) + 1
        # Save dataset locally and indicate dataset may not be sorted anymore.
        self.df.to_csv('../data/online_data/' + self.dataset + '_artistDB.csv')
        self.sorted = False