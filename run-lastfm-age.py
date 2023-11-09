import sys
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

def ugf(scores):
    return np.mean([abs(i[0] - i[1]) for i in combinations(scores, 2)])

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

def cb_recommend(user_id, topk, knn, train_df, X, item_n, item_id_to_iid, item_iid_to_id):
    sorted_rated_before = np.asarray(train_df[(train_df['user_id']==user_id)]['track_id'].value_counts().index)
    
    if sorted_rated_before.size > 0:

        raw_recommends = {}
        for item_id in sorted_rated_before:
            item_iid = item_id_to_iid[item_id]
            distances, indices = knn.kneighbors(X[item_iid], 
                                                n_neighbors=topk+1)
            sorted_pairs = sorted(list(zip(indices.squeeze().tolist(),
                                           distances.squeeze().tolist())),
                                  key=lambda x: x[1])
            raw_recommends[item_iid] = sorted_pairs 
        
        top_item_ids = []
        pos = 0
        while True:
            for item_id in sorted_rated_before:
                item_iid = item_id_to_iid[item_id]
                next_neighbor_iid = raw_recommends[item_iid][pos][0]
                next_neighbor_id = item_iid_to_id[next_neighbor_iid]
                if next_neighbor_id not in sorted_rated_before:
                    top_item_ids.append(next_neighbor_id)
                if len(top_item_ids) > topk - 1:
                    return (user_id, np.array(top_item_ids))
            
            pos += 1
    else:

        top_item_iids = random.sample(list(range(0, item_n)), topk)
        top_item_ids = [item_iid_to_id[iid] for iid in top_item_iids]
        return (user_id, np.asarray(top_item_ids))

def sample_evaluate(test_user_ids, knn, X, train_df, test_df, item_n):

    r = []
    for user_id in test_user_ids:
        test_item_ids = np.asarray(test_df[test_df['user_id']==user_id]['track_id'].unique())
        
        if len(test_item_ids) > 0:
            top_item_ids = list(cb_recommend(user_id, 10, knn, train_df, X, item_n, item_id_to_iid, item_iid_to_id)[1])
            recall = Recall(test_item_ids, top_item_ids)
            r.append(recall)
    
    return np.average(r)

if __name__ == '__main__':
    n = sys.argv[1]
    
    # for server
    listening_df = pd.read_csv('/data/sli21/lastfm/listening_events.tsv', header=1, sep='\t',
                               names=['user_id', 'track_id', 'album_id', 'timestamp'])
    user_df = pd.read_csv('/data/sli21/lastfm/users.tsv', header=1, sep='\t',
                          names=['user_id', 'country', 'age', 'gender', 'creation_time'])
    
    # user with id 2 is not in the `user_df`, so we delete their record from `listening_df` as well.
    listening_df = listening_df[listening_df['user_id'] != 2]

    # filter out users with interactions <= 10
    user_counts = listening_df['user_id'].value_counts()
    count_filtered_users = user_counts[user_counts > 10]
    count_filtered_users = count_filtered_users.index.to_numpy()
    filtered_user_df = user_df[(user_df['age'] != -1) & (user_df['user_id'].isin(count_filtered_users))].reset_index()

    # group the users into three age groups:
    # ( ,20], (20, 30], (30, )
    group_one_user_df = filtered_user_df[filtered_user_df['age'] <= 20]
    group_two_user_df = filtered_user_df[(filtered_user_df['age'] > 20) & (filtered_user_df['age'] <= 30)]
    group_thr_user_df = filtered_user_df[filtered_user_df['age'] > 30]

    # only keep the records from filtered users
    filtered_listening_df = listening_df.merge(filtered_user_df, on='user_id')

    # cf
    user_n = filtered_listening_df['user_id'].nunique()
    item_n = filtered_listening_df['track_id'].nunique()

    user_ids = filtered_listening_df['user_id'].unique()
    item_ids = filtered_listening_df['track_id'].unique()

    user_id_to_iid = {user_ids[i]:i for i in range(len(user_ids))}
    user_iid_to_id = {i:user_ids[i] for i in range(len(user_ids))}

    item_id_to_iid = {item_ids[i]:i for i in range(len(item_ids))}
    item_iid_to_id = {i:item_ids[i] for i in range(len(item_ids))}

    group_one_user_ids = group_one_user_df['user_id'].unique()
    group_two_user_ids = group_two_user_df['user_id'].unique()
    group_thr_user_ids = group_thr_user_df['user_id'].unique()

    all_cf_ugf = []
    for _ in range(n):
        
        train_df, test_df = train_test_split(filtered_listening_df, test_size=0.2)
        train_mat = df_to_mat(train_df, user_n, item_n, user_id_to_iid, item_id_to_iid)
        train_mat = train_mat.tocsr()

        test_mat = df_to_mat(test_df, user_n, item_n, user_id_to_iid, item_id_to_iid)
        test_mat = test_mat.tocsr()

        mf = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.01, alpha=1.0)
        mf.fit(train_mat)

        group_one_recall = []
        for user_id in group_one_user_ids:
            user_iid = user_id_to_iid[user_id]
    
            test_item_iids = list(np.argwhere(test_mat[user_iid] > 0)[:, 1])
            test_item_ids = [item_iid_to_id[iid] for iid in test_item_iids]

            if len(test_item_ids) > 0:
                top_item_iids = list(mf.recommend(user_iid, train_mat[user_iid], N=10, filter_already_liked_items=True)[0])
                top_item_ids = [item_iid_to_id[iid] for iid in top_item_iids]

                recall = Recall(test_item_ids, top_item_ids)
                group_one_recall.append(recall)
        
        group_two_recall = []
        for user_id in group_two_user_ids:
            user_iid = user_id_to_iid[user_id]
    
            test_item_iids = list(np.argwhere(test_mat[user_iid] > 0)[:, 1])
            test_item_ids = [item_iid_to_id[iid] for iid in test_item_iids]

            if len(test_item_ids) > 0:
                top_item_iids = list(mf.recommend(user_iid, train_mat[user_iid], N=10, filter_already_liked_items=True)[0])
                top_item_ids = [item_iid_to_id[iid] for iid in top_item_iids]

                recall = Recall(test_item_ids, top_item_ids)
                group_two_recall.append(recall)
        
        group_thr_recall = []
        for user_id in group_thr_user_ids:
            user_iid = user_id_to_iid[user_id]
    
            test_item_iids = list(np.argwhere(test_mat[user_iid] > 0)[:, 1])
            test_item_ids = [item_iid_to_id[iid] for iid in test_item_iids]

            if len(test_item_ids) > 0:
                top_item_iids = list(mf.recommend(user_iid, train_mat[user_iid], N=10, filter_already_liked_items=True)[0])
                top_item_ids = [item_iid_to_id[iid] for iid in top_item_iids]

                recall = Recall(test_item_ids, top_item_ids)
                group_thr_recall.append(recall)

        cf_scores = [np.average(group_one_recall), np.average(group_two_recall), np.average(group_thr_recall)]
        all_cf_ugf.append(ugf(cf_scores))

    # cb
    track_json_lst = []
    with open('/data/sli21/lastfm/tags.json', 'r', encoding='utf-8') as f:
        for obj in f:
            track_dict = json.loads(obj)
            track_json_lst.append(track_dict)
    
    track_tags_lst = []
    for obj in track_json_lst:
        track_id = obj['i']
        tags = list(obj['tags'].keys())[:10]    # use the first 10 tags
        track_tags_lst.append([track_id, tags])

    tag_df = pd.DataFrame(track_tags_lst, columns=['track_id', 'tags'])
    tagged_listening_df = pd.merge(listening_df, tag_df, on='track_id')

    all_cb_ugf = []
    for _ in range(n):
        tagged_listening_df = tagged_listening_df.sample(frac=0.1, ignore_index=True)

        # filter out users with interactions <= 10
        user_counts = tagged_listening_df['user_id'].value_counts()
        count_filtered_users = user_counts[user_counts > 10]
        ount_filtered_users = count_filtered_users.index.to_numpy()

        # filter out users with age=-1 and users with interactions <= 10
        filtered_user_df = user_df[(user_df['age'] != -1) & (user_df['user_id'].isin(count_filtered_users))].reset_index()

        # group the users into three age groups:
        # ( ,20], (20, 30], (30, )
        group_one_user_df = filtered_user_df[filtered_user_df['age'] <= 20]
        group_two_user_df = filtered_user_df[(filtered_user_df['age'] > 20) & (filtered_user_df['age'] <= 30)]
        group_thr_user_df = filtered_user_df[filtered_user_df['age'] > 30]

        # only keep the records from filtered users
        filtered_tagged_listening_df = tagged_listening_df.merge(filtered_user_df, on='user_id')

        user_n = filtered_tagged_listening_df['user_id'].nunique()
        item_n = filtered_tagged_listening_df['track_id'].nunique()

        user_ids = filtered_tagged_listening_df['user_id'].unique()
        item_ids = filtered_tagged_listening_df['track_id'].unique()

        item_id_to_iid = {item_ids[i]:i for i in range(len(item_ids))}
        item_iid_to_id = {i:item_ids[i] for i in range(len(item_ids))}

        group_one_user_ids = group_one_user_df['user_id'].unique()
        group_two_user_ids = group_two_user_df['user_id'].unique()
        group_thr_user_ids = group_thr_user_df['user_id'].unique()

        filtered_tag_df = filtered_tagged_listening_df.drop_duplicates(subset=['track_id'])[['track_id', 'tags']]

        tf = TfidfVectorizer(analyzer = lambda x: (g for g in x))
        X_tfidf = tf.fit_transform(filtered_tag_df['tags'])

        knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=10, n_jobs=-1)
        knn.fit(X_tfidf)

        train_df, test_df = train_test_split(filtered_tagged_listening_df, test_size=0.2)

        all_group_one_recall = []
        all_group_two_recall = []
        all_group_thr_recall = []

        n_iters = 1
        for _ in range(n_iters):

            test_group_one_user_ids = np.random.choice(group_one_user_ids, size=100, replace=False)
            test_group_two_user_ids = np.random.choice(group_two_user_ids, size=100, replace=False)
            test_group_thr_user_ids = np.random.choice(group_thr_user_ids, size=100, replace=False)

            group_one_recall = sample_evaluate(test_group_one_user_ids, knn, X_tfidf, train_df, test_df, item_n)
            group_two_recall = sample_evaluate(test_group_two_user_ids, knn, X_tfidf, train_df, test_df, item_n)
            group_thr_recall = sample_evaluate(test_group_thr_user_ids, knn, X_tfidf, train_df, test_df, item_n)

            all_group_one_recall.append(group_one_recall)
            all_group_two_recall.append(group_two_recall)
            all_group_thr_recall.append(group_thr_recall)
        
        cb_scores = [np.mean(all_group_one_recall),
                     np.mean(all_group_two_recall),
                     np.mean(all_group_thr_recall)]
        all_cb_ugf.append(ugf(cb_scores))

        print(f'cf avg ugf: {np.mean(all_cf_ugf)}, std: {np.std(all_cf_ugf)}; cb avg ugf: {np.mean(all_cb_ugf)}, std: {np.std(all_cb_ugf)}.')