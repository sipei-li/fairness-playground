from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

from utils import df_to_mat

class BasicRecommender:
    
    def __init__(self, rating_df, movie_df):
        
        self.user_n = rating_df['user_id'].nunique()
        self.item_n = movie_df['movie_id'].nunique()

        self.id_to_iid = {movie_df['movie_id'][i]:i for i in movie_df.index}
        self.iid_to_id = {i:movie_df['movie_id'][i] for i in movie_df.index}

    def fit(self, train_df):
        

class ContentBasedRecommender(BasicRecommender):

    def __init__(self, rating_df, movie_df, n_gram):
        super().__init__(rating_df, movie_df)
        
        self.tf = TfidfVectorizer(analyzer=lambda x: (c for i in range(1, n_gram) 
                                           for c in combinations(x.split('|'), r=i)))
        self.X_tfidf = self.tf.fit_transform(movie_df['genres'])
    
    def fit(self, train_df):
        self.item_user_mat = df_to_mat(train_df)