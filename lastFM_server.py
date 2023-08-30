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
        small_listening_df = listening_df.sample(frac=0.0005)  #ignore_index removed for pandas version < 1.3.0
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

if __name__ == '__main__':
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
    """
    listening_df = listening_df[listening_df['user_id'] != 2]

    f_users = user_df[user_df['gender'] == 'f']
    m_users = user_df[user_df['gender'] == 'm']

    cf_results = cf_experiment(1, listening_df, user_df)

    all_f_cf_r = np.array(cf_results[0])
    all_m_cf_r = np.array(cf_results[1])
    cf_fairness_scores = np.abs(all_f_cf_r - all_m_cf_r)
    print(f'Average fairness score for cf: {np.average(cf_fairness_scores)}; std: {np.std(cf_fairness_scores)}.')