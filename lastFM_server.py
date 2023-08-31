import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from itertools import combinations
from scipy import sparse
from scipy.sparse.linalg import svds
import implicit

import random
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

import matplotlib.pyplot as plt
import json

import math
# ground_truth: list of items ordered by time
def nDCG_Time(ground_truth, _recList):
    rec_num = len(_recList) # topK
    # ground_truth is already sorted by time
    idealOrder = ground_truth
    idealDCG = 0.0
    for j in range(min(rec_num, len(idealOrder))):
        idealDCG += ((math.pow(2.0, len(idealOrder) - j) - 1) / math.log(2.0 + j))

    recDCG = 0.0
    for j in range(rec_num):
        item = _recList[j]
        if item in ground_truth:
            rank = len(ground_truth) - ground_truth.index(item) # why ground truth?
            recDCG += ((math.pow(2.0, rank) - 1) / math.log(1.0 + j + 1))

    return (recDCG / idealDCG)


def Recall(_test_set, _recList):
    hit = len(set(_recList).intersection(set(_test_set)))
    # return hit / float(len(_test_set))
    return hit / min(float(len(_test_set)), float(len(_recList)))

def Precision(_test_set, _recList):
    hit = len(set(_recList).intersection(set(_test_set)))
    return hit / float(len(_recList))

def df_to_mat(df, user_n, item_n, user_id_to_iid, item_id_to_iid):
    """
    Convert DataFrame to sparse matrix.

    Arg:
        df: DataFrame, ratings dataframe with user_id, movie_id and rating

    Return:
        mat: scipy.sparse.csr_matrix, sparse ratings matrix with rows being users and cols being items
    """
    
    mat = sparse.lil_matrix((user_n, item_n))
    for _, row in df.iterrows():
        user_id = int(row[0])
        item_id = int(row[1])
        user_iid = user_id_to_iid[user_id]
        item_iid = item_id_to_iid[item_id]
        mat[user_iid, item_iid] = 1
    
    return mat

def cf_recommend(user_id, topk, user_id_to_iid, item_iid_to_id, train_mat, est_mat):
    
    user_iid = user_id_to_iid[user_id]
    user_interactions = train_mat[user_iid, :]
    interacted_before = np.nonzero(user_interactions)[1]
    estimations = est_mat[user_iid, :].copy()
    estimations[interacted_before] = 0

    top_item_iids = np.argsort(-estimations)[:topk]
    top_item_ids = [item_iid_to_id[i] for i in top_item_iids]

    return (user_id, np.array(top_item_ids))

def cf_experiment(n_epochs, listening_df, user_df):
    
    all_f_cf_r = []
    all_m_cf_r = []

    for _ in range(n_epochs):
        # small_listening_df = listening_df.sample(frac=0.005, ignore_index=True) #1/200 of dataset
        small_listening_df = listening_df.sample(frac=0.005)  #ignore_index removed for pandas version < 1.3.0
        # small_listening_df = listening_df #1/1 of dataset

        user_n = small_listening_df['user_id'].nunique()
        item_n = small_listening_df['track_id'].nunique()

        user_ids = small_listening_df['user_id'].unique()
        item_ids = small_listening_df['track_id'].unique()

        user_id_to_iid = {user_ids[i]:i for i in range(len(user_ids))}
        user_iid_to_id = {i:user_ids[i] for i in range(len(user_ids))}

        item_id_to_iid = {item_ids[i]:i for i in range(len(item_ids))}
        item_iid_to_id = {i:item_ids[i] for i in range(len(item_ids))}

        gender_df = pd.merge(user_df, small_listening_df, on='user_id')[['user_id', 'gender']]
        f_user_ids = gender_df[gender_df['gender'] == 'f']['user_id'].unique()
        m_user_ids = gender_df[gender_df['gender'] == 'm']['user_id'].unique()

        train_df, test_df = train_test_split(small_listening_df, test_size=0.2)

        train_mat = df_to_mat(train_df, user_n, item_n, user_id_to_iid, item_id_to_iid)
        train_mat = train_mat.tocsr()

        # mf = NMF(n_components=10, init='random', random_state=0, max_iter=500, verbose=False)
        # mf = TruncatedSVD(n_components=10, algorithm='arpack', tol=0.0)
        # user_f = mf.fit_transform(train_mat)
        # item_f = mf.components_.T
        # est_mat = np.dot(user_f, item_f.T)
        """
        user_svd = TruncatedSVD(n_components=10, algorithm='arpack', tol=0.0)
        user_f = user_svd.fit_transform(train_mat)
        item_svd = TruncatedSVD(n_components=10, algorithm='arpack', tol=0.0)
        item_f = item_svd.fit_transform(train_mat.transpose())
        est_mat = np.dot(user_f, item_f.T)
        """
        mf = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.05, alpha=2.0)
        mf.fit(train_mat)
        user_f = mf.user_factors
        item_f = mf.item_factors
        est_mat = np.dot(user_f, item_f.T)

        test_mat = df_to_mat(test_df, user_n, item_n, user_id_to_iid, item_id_to_iid)
        test_mat = test_mat.tocsr()
        
        f_cf_r = []
        for user_id in f_user_ids:
            user_iid = user_id_to_iid[user_id]
            test_item_iids = list(np.argwhere(test_mat[user_iid] > 0)[:, 1])
            test_item_ids = list(map(lambda x: item_iid_to_id[x], test_item_iids))

            if len(test_item_ids) > 0:
                top_item_ids = list(cf_recommend(user_id, 10, user_id_to_iid, item_iid_to_id, train_mat, est_mat)[1])

                recall = Recall(test_item_ids, top_item_ids)
                f_cf_r.append(recall)
        
        all_f_cf_r.append(np.average(f_cf_r))

        m_cf_r = []
        for user_id in m_user_ids:
            user_iid = user_id_to_iid[user_id]
            test_item_iids = list(np.argwhere(test_mat[user_iid] > 0)[:, 1])
            test_item_ids = list(map(lambda x: item_iid_to_id[x], test_item_iids))

            if len(test_item_ids) > 0:
                top_item_ids = list(cf_recommend(user_id, 10, user_id_to_iid, item_iid_to_id, train_mat, est_mat)[1])

                recall = Recall(test_item_ids, top_item_ids)
                m_cf_r.append(recall)
        
        all_m_cf_r.append(np.average(m_cf_r))
        # print(all_f_cf_r, all_m_cf_r)
    
    return (all_f_cf_r, all_m_cf_r)

def cb_recommend(user_id, topk, knn, item_user_mat, X, user_id_to_iid, item_iid_to_id, item_n):
    user_iid = user_id_to_iid[user_id]
    user_ratings = item_user_mat[:, user_iid]
    rated_before = np.nonzero(user_ratings)[0]
    sorted_rated_before = rated_before[
        np.argsort(user_ratings[rated_before].toarray().squeeze())][::-1]
    
    if sorted_rated_before.size > 0:

        raw_recommends = {}
        for item_iid in sorted_rated_before:
            distances, indices = knn.kneighbors(X[item_iid], 
                                                n_neighbors=topk+1)
            sorted_pairs = sorted(list(zip(indices.squeeze().tolist(),
                                           distances.squeeze().tolist())),
                                  key=lambda x: x[1])
            raw_recommends[item_iid] = sorted_pairs 
        
        top_item_ids = []
        pos = 0
        while True:
            for item_iid in sorted_rated_before:
                next_neighbor_iid = raw_recommends[item_iid][pos][0]
                if next_neighbor_iid not in rated_before:
                    top_item_ids.append(item_iid_to_id[next_neighbor_iid])
                if len(top_item_ids) > topk - 1:
                    return (user_id, np.array(top_item_ids))
            
            pos += 1
    else:

        top_item_ids = list(map(lambda x: item_iid_to_id[x], 
                             random.sample(list(range(0, item_n)), topk)))
        return (user_id, np.array(top_item_ids))
    
def sample_evaluate(test_user_ids, knn, X, user_id_to_iid, item_iid_to_id, train_mat, test_mat, item_n):

    r = []

    for user_id in test_user_ids:
        user_iid = user_id_to_iid[user_id]
        test_item_iids = list(np.argwhere(test_mat[:, user_iid] > 0)[:, 0])
        test_item_ids = list(map(lambda x: item_iid_to_id[x], test_item_iids))

        if len(test_item_ids) > 0:
            top_item_ids = list(cb_recommend(user_id, 10, knn, train_mat, X, user_id_to_iid, item_iid_to_id, item_n)[1])

            recall = Recall(test_item_ids, top_item_ids)

            r.append(recall)
    
    return np.average(r)

def cb_experiment(n_epochs, n_iters, listening_df, user_df, tag_df):

    all_f_cb_r = []
    all_m_cb_r = []
    
    tagged_listening_df = pd.merge(listening_df, tag_df, on='track_id')

    for _ in range(n_epochs):
        small_listening_df = tagged_listening_df.sample(frac=0.02, ignore_index=True)

        user_n = small_listening_df['user_id'].nunique()
        item_n = small_listening_df['track_id'].nunique()

        user_ids = small_listening_df['user_id'].unique()
        item_ids = small_listening_df['track_id'].unique()

        user_id_to_iid = {user_ids[i]:i for i in range(len(user_ids))}
        user_iid_to_id = {i:user_ids[i] for i in range(len(user_ids))}

        item_id_to_iid = {item_ids[i]:i for i in range(len(item_ids))}
        item_iid_to_id = {i:item_ids[i] for i in range(len(item_ids))}

        gender_df = pd.merge(user_df, small_listening_df, on='user_id')[['user_id', 'gender']]
        f_user_ids = gender_df[gender_df['gender'] == 'f']['user_id'].unique()
        m_user_ids = gender_df[gender_df['gender'] == 'm']['user_id'].unique()

        small_tag_df = small_listening_df.drop_duplicates(subset=['track_id'])[['track_id', 'tags']]
        tf = TfidfVectorizer(analyzer = lambda x: (g for g in x))
        X_tfidf = tf.fit_transform(small_tag_df['tags'])

        knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=10, n_jobs=-1)
        knn.fit(X_tfidf)

        train_df, test_df = train_test_split(small_listening_df, test_size=0.2)

        train_mat = df_to_mat(train_df, user_n, item_n, user_id_to_iid, item_id_to_iid)
        train_mat = train_mat.transpose().tocsr()

        test_mat = df_to_mat(test_df, user_n, item_n, user_id_to_iid, item_id_to_iid)
        test_mat= test_mat.transpose().tocsr()

        
        for _ in range(n_iters):

            test_f_user_ids = np.random.choice(f_user_ids, size=500, replace=False)
            test_m_user_ids = np.random.choice(m_user_ids, size=500, replace=False)

            f_cb_r = sample_evaluate(test_f_user_ids, knn, X_tfidf, user_id_to_iid, item_iid_to_id, train_mat, test_mat, item_n)
            m_cb_r = sample_evaluate(test_m_user_ids, knn, X_tfidf, user_id_to_iid, item_iid_to_id, train_mat, test_mat, item_n)

            all_f_cb_r.append(f_cb_r)
            all_m_cb_r.append(m_cb_r)

    return (np.array(all_f_cb_r), np.array(all_m_cb_r))

if __name__ == '__main__':
    """
    # for local
    listening_df = pd.read_csv('data/lastfm/listening_events.tsv', header=1, sep='\t',
                      names=['user_id', 'track_id', 'album_id', 'timestamp'])
    user_df = pd.read_csv('data/lastfm/users.tsv', header=1, sep='\t',
                      names=['user_id', 'country', 'age', 'gender', 'creation_time'])
    
    """
    # for server
    listening_df = pd.read_csv('/data/sli21/lastfm/listening_events.tsv', header=1, sep='\t',
                      names=['user_id', 'track_id', 'album_id', 'timestamp'])
    user_df = pd.read_csv('/data/sli21/lastfm/users.tsv', header=1, sep='\t',
                      names=['user_id', 'country', 'age', 'gender', 'creation_time'])
    
    listening_df = listening_df[listening_df['user_id'] != 2]

    f_users = user_df[user_df['gender'] == 'f']
    m_users = user_df[user_df['gender'] == 'm']

    """
    cf_results = cf_experiment(10, listening_df, user_df)

    all_f_cf_r = np.array(cf_results[0])
    all_m_cf_r = np.array(cf_results[1])
    cf_fairness_scores = np.abs(all_f_cf_r - all_m_cf_r)
    print(f'Average fairness score for cf: {np.average(cf_fairness_scores)}; std: {np.std(cf_fairness_scores)}.')
    """
    track_json_lst = []
    with open('/data/sli21/lastfm/tags.json', 'r', encoding='utf-8') as f:
        for obj in f:
            track_dict = json.loads(obj)
            track_json_lst.append(track_dict)

    track_tags_lst = []
    for obj in track_json_lst:
        track_id = obj['i']
        tags = list(obj['tags'].keys())[:10]
        track_tags_lst.append([track_id, tags])

    tag_df = pd.DataFrame(track_tags_lst, columns=['track_id', 'tags'])

    (all_f_cb_r, all_m_cb_r) = cb_experiment(10, 5, listening_df, user_df, tag_df)

    cb_fairness_scores = np.abs(all_f_cb_r - all_m_cb_r)
    print(f'Average fairness score for cb: {np.average(cb_fairness_scores)}; std: {np.std(cb_fairness_scores)}.')