import numpy as np
import scipy.sparse as sp
from IPython import embed


class Sampler(object):
    def __init__(self, dims, num_ng=4, sample_method='item-desc', sample_ratio=0):
        """
        negative sampling class for some algorithms
        Parameters
        ----------
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, # of nagative sampling per sample
        sample_method : str, sampling method
                        'uniform' discrete uniform
                        'item-desc' descending item popularity, high popularity means high probability to choose
                        'item-ascd' ascending item popularity, low popularity means high probability to choose
        sample_ratio : float, scope [0, 1], it determines what extent the sample method except 'uniform' occupied
        """
        self.dims = dims
        self.num_ng = num_ng
        self.sample_method = sample_method
        self.sample_ratio = sample_ratio

        assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
        assert 0 <= sample_ratio <= 1, 'Invalid sample ratio value'

    def transform(self, sampled_df, is_training=True, context=False):
        """

        Parameters
        ----------
        sampled_df : pd.DataFrame, dataframe waiting for sampling
        is_training : boolean, whether the procedure using this method is training part

        Returns
        -------
        neg_set : List, list of (user, item, rating, negative sampled items)
        """
        if not is_training:
            neg_set = []
            for _, row in sampled_df.iterrows():
                u = int(row['user'])
                i = int(row['item'])
                c = int(row['context']) if context else None
                r = row['rating']
                js = []
                neg_set.append([u, i, c, r, js]) if context else neg_set.append([u, i, r, js])
            
            return neg_set

        user_num, item_num = np.max(sampled_df[['user', 'item']].to_numpy(), axis=0) + 1
        # IDEA: build_adj_mx
        dims = self.dims[:3] if context else self.dims[:2]
        pair_pos = sp.dok_matrix((dims[-1], dims[-1]), dtype=np.float32)
        neg_sample_pool = list(range(user_num, item_num))

        for _, row in sampled_df.iterrows():
            pair_pos[int(row['user']), int(row['item'])] = 1.0
            pair_pos[int(row['item']), int(row['user'])] = 1.0

            # TODO: EXTEND TO MULTIPLE DIMENSIONS
            if context:
                # for idx in range(len(row[2:]) -2):  #subtract rating and timestamp
                pair_pos[int(row['user']), int(row['context'])] = 1.0
                pair_pos[int(row['item']), int(row['context'])] = 1.0

                pair_pos[int(row['context']), int(row['user'])] = 1.0
                pair_pos[int(row['context']), int(row['item'])] = 1.0

        popularity_item_list = sampled_df['item'].value_counts().index.tolist()
        if self.sample_method == 'item-desc':
            neg_sample_pool = popularity_item_list
        elif self.sample_method == 'item-ascd':
            neg_sample_pool = popularity_item_list[::-1]
        
        neg_set = []
        uni_num = int(self.num_ng * (1 - self.sample_ratio))
        ex_num = self.num_ng - uni_num
        for _, row in sampled_df.iterrows():
            u = int(row['user'])
            i = int(row['item'])
            c = None if not context else int(row['context'])
            r = row['rating']

            js = []
            for _ in range(uni_num):
                j = np.random.randint(user_num, item_num)
                while (u, j) in pair_pos:
                    j = np.random.randint(user_num, item_num)
                js.append(j)
            for _ in range(ex_num):
                if self.sample_method in ['item-desc', 'item-ascd']:
                    idx = 0
                    j = int(neg_sample_pool[idx])
                    while (u, j) in pair_pos:
                        idx += 1
                        j = int(neg_sample_pool[idx])
                    js.append(j)
                else:
                    # maybe add other sample methods in future, uniform as default
                    # j = np.random.randint(item_num)
                    j = np.random.randint(user_num, item_num)

                    while (u, j) in pair_pos:
                        j = np.random.randint(user_num, item_num)
                    js.append(j)
            neg_set.append([u, i, c, r, js]) if context else neg_set.append([u, i, r, js])

        print(f'Finish negative samplings, sample number is {len(neg_set) * self.num_ng}......')
        return neg_set, pair_pos
