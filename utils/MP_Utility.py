from math import log
import numpy as np
import pandas as pd
import copy
from operator import itemgetter
import time
import tqdm

from scipy import stats
from scipy.sparse import coo_matrix
from multiprocessing import Process, Queue, Pool, Manager
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import pickle
import random as rd

top1 = 1
top2 = 5
top3 = 10
top4 = 20
k_set = [top1, top2, top3, top4]

position_bias = 1. / np.log2(np.arange(5000) + 2)

# def neg_sampling_helper(num_user, num_item, train_user_list, neg_rate, item_pop):
#     pos_items = []
#     neg_items = []
#     pos_pop = []
#     neg_pop = []
#     items = np.arange(num_item)
#     for u in range(num_user):
#         if train_user_list[u] == []:
#             p_item = 0
#         else:
#             p_item = rd.choice(train_user_list[u])
#         pos_items.append(p_item)
#         pos_pop.append(item_pop[p_item])
#
#         if neg_rate == 1:
#             while True:
#                 n_item = rd.choice(items)
#                 if n_item not in train_user_list[u]:
#                     break
#         else:
#             n_item = randint_choice(items, size=neg_rate, exclusion=train_user_list[u])
#         neg_items.append(n_item)
#         neg_pop.append(item_pop[n_item])
#
#     return np.array(pos_items), np.array(neg_items), np.array(pos_pop), np.array(neg_pop)

def neg_sampling_help(num_user, num_item, pos_user_array, pos_item_array, neg_rate, train_user_list):
    user_list = []
    pos_item_list = []
    neg_item_list = []
    items = np.arange(num_item)
    # for u in pos_user_array:
    #     user_list.append(u)
    #     pos_item_list.append(pos_item_array[u])
    #
    #     if neg_rate == 1:
    #         while True:
    #             n_item = rd.choice(items)
    #             if n_item not in train_user_list[u]:
    #                 break
    #     else:
    #         n_item = randint_choice(items, size=neg_rate, exclusion=train_user_list[u])
    #     neg_item_list.append(n_item)

    for u in pos_user_array:
        for t in range(neg_rate):
            while True:
                n_item = rd.choice(items)
                if n_item not in train_user_list[u]:
                    break
            user_list.append(u)
            pos_item_list.append(pos_item_array[u])
            neg_item_list.append(n_item)

    return np.array(user_list), np.array(pos_item_list), np.array(neg_item_list)

def neg_sampling_help_bc(num_user, num_item, pos_user_array, pos_item_array, neg_rate, train_user_list):
    user_list = []
    pos_item_list = []
    neg_item_list = []
    items = np.arange(num_item)

    # for u in pos_user_array:
    #     for t in range(neg_rate):
    #         while True:
    #             n_item = rd.choice(items)
    #             if n_item not in train_user_list[u]:
    #                 break
    #         user_list.append(u)
    #         pos_item_list.append(pos_item_array[u])
    #         neg_item_list.append(n_item)

    # for u in pos_user_array:
    #     user_list.append(u)
    #     pos_item_list.append(pos_item_array[u])
    #
    #     if neg_rate == 1:
    #         while True:
    #             n_item = rd.choice(items)
    #             if n_item not in train_user_list[u]:
    #                 break
    #     else:
    #         n_item = randint_choice(items, size=neg_rate, exclusion=train_user_list[u])
    #     neg_item_list.append(n_item)

    for u in range(num_user):
        user_list.append(u)
        # pos_item_list.append(pos_item_array[u])
        pos_item = rd.choice(train_user_list[u])
        pos_item_list.append(pos_item)

        if neg_rate == 1:
            while True:
                n_item = rd.choice(items)
                if n_item not in train_user_list[u]:
                    break
        else:
            n_item = randint_choice(items, size=neg_rate, exclusion=train_user_list[u])

        # n_item = randint_choice(items, size=neg_rate, exclusion=train_user_list[u])
        neg_item_list.append(n_item)

    return np.array(user_list), np.array(pos_item_list), np.array(neg_item_list)

def negative_sampling_full(num_user, num_item, pos_user_array, pos_item_array, neg_rate):

    train_mat = coo_matrix((np.ones(pos_user_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True)
    pos = pos_item_array
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_item_array.shape[0]), replace=True)
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx]
    neg = neg[idx]
    pos_label = np.ones(pos.shape)
    neg_label = np.zeros(neg.shape)
    # return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
    #        np.concatenate([pos_label, neg_label], axis=0)
    return np.array(user_pos), np.array(user_neg), np.array(pos), np.array(neg), np.array(pos_label), np.array(neg_label)

def negative_sampling(num_user, num_item, pos_user_array, pos_item_array, neg_rate):

    train_mat = coo_matrix((np.ones(pos_user_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True)
    pos = pos_item_array
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_item_array.shape[0]), replace=True)
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx]
    neg = neg[idx]
    pos_label = np.ones(pos.shape)
    neg_label = np.zeros(neg.shape)
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0)

def negative_sampling_vae(num_user, num_item, pos_user_array, pos_item_array, neg_rate, rec):

    train_mat = coo_matrix((np.ones(pos_user_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True)
    pos = pos_item_array
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_item_array.shape[0]), replace=True)
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx]
    neg = neg[idx]
    pos_rec = rec[user_pos, pos]
    neg_rec = rec[user_neg, neg]
    pos_label = np.ones(pos.shape)
    neg_label = np.zeros(neg.shape)
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0), np.concatenate([pos_rec, neg_rec], axis=0)

def positive_sampling(num_user, num_item, pos_user_array, pos_item_array):
    user_pos = pos_user_array
    pos = pos_item_array
    pos_label = np.ones(pos.shape)

    return user_pos, pos, pos_label

def negative_sampling_boost(num_user, num_item, pos_user_array, pos_item_array, neg_rate, sample_weights):

    train_mat = coo_matrix((np.ones(pos_user_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True)
    pos = pos_item_array
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_item_array.shape[0]), replace=True)
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx]
    neg = neg[idx]
    pos_weights = sample_weights[user_pos, pos]
    neg_weights = sample_weights[user_neg, neg]
    pos_label = np.ones(pos.shape)
    neg_label = np.zeros(neg.shape)
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0), np.concatenate([pos_weights, neg_weights], axis=0)

def negative_sampling_gb(num_user, num_item, pos_user_array, pos_item_array, neg_rate, prev_res):

    train_mat = coo_matrix((np.ones(pos_user_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()


    train_hat = train_mat - prev_res
    # train_hat = -((train_mat - 1) * np.exp(sample_weights) + train_mat) / (np.exp(sample_weights) + 1)

    user_pos = pos_user_array
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True)
    pos = pos_item_array
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_item_array.shape[0]), replace=True)
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx]
    neg = neg[idx]
    # pos_weights = sample_weights[user_pos, pos]
    # neg_weights = sample_weights[user_neg, neg]
    # pos_label = np.ones(pos.shape)
    # neg_label = np.zeros(neg.shape)
    pos_label = train_hat[user_pos, pos]
    neg_label = train_hat[user_neg, neg]
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0)

def negative_sampling_gb_new(num_user, num_item, pos_user_array, pos_item_array, neg_rate, prev_res):

    train_mat = coo_matrix((np.ones(pos_user_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()


    # train_hat = np.abs(train_mat - prev_res)
    train_hat = np.square(train_mat - prev_res)
    # train_hat = -((train_mat - 1) * np.exp(sample_weights) + train_mat) / (np.exp(sample_weights) + 1)

    user_pos = pos_user_array
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True)
    pos = pos_item_array
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_item_array.shape[0]), replace=True)
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx]
    neg = neg[idx]
    # pos_weights = sample_weights[user_pos, pos]
    # neg_weights = sample_weights[user_neg, neg]
    # pos_label = np.ones(pos.shape)
    # neg_label = np.zeros(neg.shape)
    pos_label = train_hat[user_pos, pos]
    neg_label = train_hat[user_neg, neg]
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0)

def negative_sampling_ada_gb(num_user, num_item, pos_user_array, pos_item_array, neg_rate, sample_weights, prev_res):

    train_mat = coo_matrix((np.ones(pos_user_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()


    train_hat = train_mat - prev_res
    # train_hat = -((train_mat - 1) * np.exp(sample_weights) + train_mat) / (np.exp(sample_weights) + 1)

    user_pos = pos_user_array
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True)
    pos = pos_item_array
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_item_array.shape[0]), replace=True)
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx]
    neg = neg[idx]
    pos_weights = sample_weights[user_pos, pos]
    neg_weights = sample_weights[user_neg, neg]
    # pos_label = np.ones(pos.shape)
    # neg_label = np.zeros(neg.shape)
    pos_label = train_hat[user_pos, pos]
    neg_label = train_hat[user_neg, neg]
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0), np.concatenate([pos_weights, neg_weights], axis=0)

def negative_sampling_better(num_user, num_item, train_array, negative_array, neg_rate):
    pos_user_array = train_array[:, 0]
    pos_item_array = train_array[:, 1]
    # pos_prop_array = position_bias[train_array[:, 2]] * feedback_expose_prob
    pos_prop_array = position_bias[train_array[:, 2]]
    num_pos = len(pos_user_array)

    neg_user_array = negative_array[:, 0]
    neg_item_array = negative_array[:, 1]
    neg_prop_array = position_bias[negative_array[:, 2]]
    num_neg = len(neg_user_array)

    user_pos = pos_user_array.reshape((-1, 1))
    pos = pos_item_array.reshape((-1, 1))
    pos_label = np.ones(pos.shape)
    pos_prop = pos_prop_array.reshape((-1, 1))

    num_neg_sample = int(num_pos * neg_rate)
    neg_idx = np.random.choice(np.arange(num_neg), num_neg_sample, replace=True)
    user_neg = neg_user_array[neg_idx].reshape((-1, 1))
    neg = neg_item_array[neg_idx].reshape((-1, 1))
    neg_label = np.zeros(neg.shape)
    neg_prop = neg_prop_array[neg_idx].reshape((-1, 1))

    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0), np.concatenate([pos_prop, neg_prop], axis=0)

def negative_sampling_AutoRec(num_row, num_col, row_array, col_array, neg):
    row = np.tile(row_array, neg + 1).reshape(row_array.shape[0] * (neg + 1))
    pos = col_array.reshape((col_array.shape[0], 1))
    neg = np.random.choice(np.arange(num_col), size=(neg * col_array.shape[0]),
                           replace=True).reshape((col_array.shape[0] * neg, 1))
    col = np.concatenate([pos, neg], axis=0)
    mask = coo_matrix((np.ones(row.shape[0]), (row, col.reshape(col.shape[0]))),
                      shape=(num_row, num_col)).toarray()
    return mask


def test_model(num_u, Rec, like, test_like, precision_queue, recall_queue, ndcg_queue, n_user_queue):
    precision = np.array([0.0, 0.0, 0.0, 0.0])
    recall = np.array([0.0, 0.0, 0.0, 0.0])
    ndcg = np.array([0.0, 0.0, 0.0, 0.0])

    user_num = num_u

    for i in range(num_u):
        Rec[i, like[i]] = -100000.0

    for u in range(num_u):  # iterate each user
        scores = Rec[u, :]
        top_iid = np.argpartition(scores, -20)[-20:]
        top_iid = top_iid[np.argsort(scores[top_iid])[-1::-1]]

        # calculate the metrics
        if not len(test_like[u]) == 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top_iid, test_like[u])
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            user_num -= 1
    precision_queue.put(precision)
    recall_queue.put(recall)
    ndcg_queue.put(ndcg)
    n_user_queue.put(user_num)


def MP_test_model_all(Rec, test_like, train_like, n_workers=10):
    m = Manager()
    precision_queue = m.Queue(maxsize=n_workers)
    recall_queue = m.Queue(maxsize=n_workers)
    ndcg_queue = m.Queue(maxsize=n_workers)
    n_user_queue = m.Queue(maxsize=n_workers)
    processors = []

    num_user = Rec.shape[0]

    num_user_each = int(num_user / n_workers)
    for i in range(n_workers):
        if i < n_workers - 1:
            p = Process(target=test_model, args=(num_user_each,
                                                 Rec[num_user_each * i: num_user_each * (i + 1)],
                                                 train_like[num_user_each * i: num_user_each * (i + 1)],
                                                 test_like[num_user_each * i: num_user_each * (i + 1)],
                                                 precision_queue,
                                                 recall_queue,
                                                 ndcg_queue,
                                                 n_user_queue))
            processors.append(p)
        else:
            p = Process(target=test_model, args=(num_user - num_user_each * i,
                                                 Rec[num_user_each * i: num_user],
                                                 train_like[num_user_each * i: num_user],
                                                 test_like[num_user_each * i: num_user],
                                                 precision_queue,
                                                 recall_queue,
                                                 ndcg_queue,
                                                 n_user_queue))
            processors.append(p)
        p.start()
    print('!!!!!!!!!!!!!!!!!test start!!!!!!!!!!!!!!!!!!')

    for p in processors:
        p.join()
    precision = precision_queue.get()
    while not precision_queue.empty():
        tmp = precision_queue.get()
        precision += tmp
    recall = recall_queue.get()
    while not recall_queue.empty():
        tmp = recall_queue.get()
        recall += tmp
    ndcg = ndcg_queue.get()
    while not ndcg_queue.empty():
        tmp = ndcg_queue.get()
        ndcg += tmp
    n_user = n_user_queue.get()
    while not n_user_queue.empty():
        tmp = n_user_queue.get()
        n_user += tmp

    # compute the average over all users
    precision /= n_user
    recall /= n_user
    ndcg /= n_user

    # print('precision_%d\t[%.7f],\t||\t precision_%d\t[%.7f],\t||\t precision_%d\t[%.7f],\t||\t precision_%d\t[%.7f]'
    #       % (k_set[0], precision[0], k_set[1], precision[1], k_set[2], precision[2], k_set[3], precision[3]))
    #
    # print('recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f]'
    #       % (k_set[0], recall[0], k_set[1], recall[1], k_set[2], recall[2], k_set[3], recall[3]))
    #
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[2] == 0 else 0
    f_measure_20 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[3] == 0 else 0
    # print('f_measure_%d\t[%.7f],\t||\t f_measure_%d\t[%.7f],\t||\t f_measure_%d\t[%.7f],\t||\t f_measure_%d\t[%.7f]'
    #       % (k_set[0], f_measure_1, k_set[1], f_measure_5, k_set[2], f_measure_10, k_set[3], f_measure_15))
    f_score = np.array([f_measure_1, f_measure_5, f_measure_10, f_measure_20])
    #
    # print('ndcg_%d     \t[%.7f],\t||\t ndcg_%d     \t[%.7f],\t||\t ndcg_%d     \t[%.7f],\t||\t ndcg_%d     \t[%.7f]'
    #       % (k_set[0], ndcg[0], k_set[1], ndcg[1], k_set[2], ndcg[2], k_set[3], ndcg[3]))
    print('ndcg_%d     \t[%.7f]' % (k_set[3], ndcg[3]))

    return precision, recall, f_score, ndcg
    # return ndcg[3]


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def relu(x):
    return np.maximum(x, 0)


# calculate NDCG@k
def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg


# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_precision_recall_ndcg(new_user_prediction, test):
    dcg_list = []

    # compute the number of true positive items at top k
    count_1, count_5, count_10, count_20 = 0, 0, 0, 0
    for i in range(k_set[3]):
        if i < k_set[0] and new_user_prediction[i] in test:
            count_1 += 1.0
        if i < k_set[1] and new_user_prediction[i] in test:
            count_5 += 1.0
        if i < k_set[2] and new_user_prediction[i] in test:
            count_10 += 1.0
        if new_user_prediction[i] in test:
            count_20 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, k_set[0])
    ndcg_tmp_5 = NDCG_at_k(dcg_list, idcg_list, k_set[1])
    ndcg_tmp_10 = NDCG_at_k(dcg_list, idcg_list, k_set[2])
    ndcg_tmp_20 = NDCG_at_k(dcg_list, idcg_list, k_set[3])

    # precision@k
    precision_1 = count_1 * 1.0 / k_set[0]
    precision_5 = count_5 * 1.0 / k_set[1]
    precision_10 = count_10 * 1.0 / k_set[2]
    precision_20 = count_20 * 1.0 / k_set[3]

    l = len(test)
    if l == 0:
        l = 1
    # recall@k
    recall_1 = count_1 / l
    recall_5 = count_5 / l
    recall_10 = count_10 / l
    recall_20 = count_20 / l

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_5, precision_10, precision_20]), \
           np.array([recall_1, recall_5, recall_10, recall_20]), \
           np.array([ndcg_tmp_1, ndcg_tmp_5, ndcg_tmp_10, ndcg_tmp_20])


def print_sorted_dict(dictionary):
    tmp = []
    for key, value in [(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get)]:
        tmp.append(value)
        print("# %s: %s" % (key, value))
    rstd = np.std(tmp) / np.mean(tmp)
    print("# relative std = " + str(rstd))
    return rstd

def ranking_analysis_save(Rec, train_like, test_like, train_pop, test_pop, save_file):
    Rec = copy.copy(Rec)

    num_user = Rec.shape[0]
    num_item = Rec.shape[1]

    count1_pos = np.zeros(num_item)
    count2_pos = np.zeros(num_item)
    count3_pos = np.zeros(num_item)
    count4_pos = np.zeros(num_item)

    I_rank_pos = np.zeros(num_item)

    attention_table = np.zeros(num_item)
    for t in range(num_item):
        attention_table[t] = 1 / np.log2(2 + t)

    for u in range(num_user):
        Rec[u, train_like[u]] = -100000000.0

    item_pop = train_pop
    user_pop = np.zeros(num_user)
    for u in range(num_user):
        user_pop[u] = len(train_like[u])

    CC_user_rank_pos = []

    for u in tqdm.tqdm(range(num_user)):  # iterate each user
        u_pop = user_pop[u]
        if u_pop == 0:
            continue

        u_test = test_like[u]
        u_pred = Rec[u, :]

        top_item_idx_no_train = np.arange(num_item)
        u_top = (np.array([top_item_idx_no_train, u_pred[top_item_idx_no_train]])).T
        u_top = sorted(u_top, key=itemgetter(1), reverse=True)

        # calculate metrics for user based item-popularity bias
        if len(u_test) > 2:
            pos_pop_list = []
            pos_rank = []
            pos_attention = []
            pos_score = []
            for t in range(int(num_item - u_pop)):
                cur_id = int(u_top[t][0])
                cur_score = u_top[t][1]
                if cur_id in u_test:
                    cur_i_pop = item_pop[cur_id]
                    if cur_i_pop == 0:
                        continue
                    pos_pop_list.append(cur_i_pop)
                    pos_rank.append(t)
                    pos_attention.append(attention_table[t])
                    pos_score.append(cur_score)
            pos_pop_list = np.array(pos_pop_list)
            # pos_rank = np.array(pos_rank).astype(np.float)
            pos_rank = np.array(pos_rank).astype(float)

            if np.std(pos_pop_list) == 0:
                continue
            CC_user_rank_pos_tmp = stats.spearmanr(pos_pop_list + 1e-7, pos_rank + 1e-7)
            # CC_user_rank_pos_tmp = stats.pearsonr(pos_pop_list + 1e-7, pos_rank + 1e-7)

            CC_user_rank_pos.append(CC_user_rank_pos_tmp)

        if not len(u_test) == 0:
            for t in range(int(num_item - u_pop)):
                cur_id = int(u_top[t][0])
                if cur_id in u_test:
                    cur_i_pop = item_pop[cur_id]
                    if cur_i_pop == 0:
                        continue
                    I_rank_pos[cur_id] += t
                    if t < top4:
                        count4_pos[cur_id] += 1.
                        if t < top3:
                            count3_pos[cur_id] += 1.
                            if t < top2:
                                count2_pos[cur_id] += 1.
                                if t < top1:
                                    count1_pos[cur_id] += 1.

    # ranking probability
    prob1_pos = count1_pos / (test_pop + 1e-7)
    prob2_pos = count2_pos / (test_pop + 1e-7)
    prob3_pos = count3_pos / (test_pop + 1e-7)
    prob4_pos = count4_pos / (test_pop + 1e-7)
    prob_pos = [prob1_pos, prob2_pos, prob3_pos, prob4_pos]

    result_dict = {}
    result_dict['prob_pos'] = prob_pos
    result_dict['I_rank_pos'] = I_rank_pos
    result_dict['CC_user_rank_pos'] = CC_user_rank_pos

    with open(save_file, 'wb') as f:
        pickle.dump(result_dict, f)

    return

def ranking_analysis_load(train_pop, test_pop, load_file):
    result_dict = {}
    with open(load_file, 'rb') as f:
        result_dict = pickle.load(f)

    test_nonzero_idx = np.where(test_pop > 0)[0]
    test_pop = test_pop[test_nonzero_idx]
    train_pop = train_pop[test_nonzero_idx]

    prob_pos = result_dict['prob_pos']
    prob1_pos = prob_pos[0]
    prob2_pos = prob_pos[1]
    prob3_pos = prob_pos[2]
    prob4_pos = prob_pos[3]
    prob1_pos = prob1_pos[test_nonzero_idx]
    prob2_pos = prob2_pos[test_nonzero_idx]
    prob3_pos = prob3_pos[test_nonzero_idx]
    prob4_pos = prob4_pos[test_nonzero_idx]

    I_rank_pos = result_dict['I_rank_pos']
    I_rank_pos = I_rank_pos[test_nonzero_idx]
    CC_user_rank_pos = np.array(result_dict['CC_user_rank_pos'])

    CC_user_rank_pos_avg = np.mean(CC_user_rank_pos[:, 0])

    # user based item-popularity bias
    print('')
    print('!' * 50 + 'User based item-popularity bias' + '!' * 50)
    print('#' * 100)
    print('# (PRU) Average CC between item pop and RANK given positive = ' + str(-CC_user_rank_pos_avg))
    print('!' * 120)
    print('')

    # ranking probability
    CC1_pos = stats.spearmanr(prob1_pos + 1e-7, train_pop + 1e-7)
    CC2_pos = stats.spearmanr(prob2_pos + 1e-7, train_pop + 1e-7)
    CC3_pos = stats.spearmanr(prob3_pos + 1e-7, train_pop + 1e-7)
    CC4_pos = stats.spearmanr(prob4_pos + 1e-7, train_pop + 1e-7)

    # CC1_pos = stats.pearsonr(prob1_pos + 1e-7, train_pop + 1e-7)
    # CC2_pos = stats.pearsonr(prob2_pos + 1e-7, train_pop + 1e-7)
    # CC3_pos = stats.pearsonr(prob3_pos + 1e-7, train_pop + 1e-7)
    # CC4_pos = stats.pearsonr(prob4_pos + 1e-7, train_pop + 1e-7)

    print('')
    print('!' * 50 + 'Ranking probability' + '!' * 50)
    print('#' * 100)
    print('# CC of ranking @ ' + str(top1) + ' given positive = ' + str(CC1_pos))
    print('# CC of ranking @ ' + str(top2) + ' given positive = ' + str(CC2_pos))
    print('# CC of ranking @ ' + str(top3) + ' given positive = ' + str(CC3_pos))
    print('# CC of ranking @ ' + str(top4) + ' given positive = ' + str(CC4_pos))
    print('!' * 120)
    print('')

    # item based
    I_rank_pos = I_rank_pos / (test_pop + 1e-7)

    CC_I_rank_pos = stats.spearmanr(I_rank_pos + 1e-7, train_pop + 1e-7)

    print('')
    print('!' * 50 + 'item based item-popularity bias' + '!' * 50)
    print('#' * 100)
    # need a negative sign
    # print('# (PRI) SCC of item rank for item given positive = ' + str(-CC_I_rank_pos))
    print('# (PRI) SCC of item rank for item given positive = ' + str(CC_I_rank_pos))
    print('!' * 120)
    print('')

    return

def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    # a = np.arange(high)
    a = high
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
    if p is not None:
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample