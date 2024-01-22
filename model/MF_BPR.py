import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import linalg as LA

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool
from utils import MP_Utility

import copy
from past.builtins import range

import pickle
import argparse
import pandas as pd
from scipy.sparse import csr_matrix, rand as sprand
from tqdm import tqdm
from time import strftime

from evaluation.backend import HoldoutEvaluator
# from statistics import mean

# np.random.seed(0)
# torch.manual_seed(0)


class MF_BPR(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MF_BPR, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.emb_dim = model_conf['emb_dim']
        self.neg_sample = model_conf['neg_sample_rate']

        self.batch_size = model_conf['batch_size']
        self.regularization = model_conf['reg']
        self.lr = model_conf['lr']
        help_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name)
        help_dir = os.path.join(help_dir, 'bias_scores')
        self.train_user_list = np.load(help_dir + '/train_user_list.npy', allow_pickle=True)
        self.train_df = dataset.train_df
        self.device = device
        self.loss_function = torch.nn.MSELoss()
        # self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users, self.num_items,
        #                                                                             self.train_df[0],
        #                                                                             self.train_df[1],
        #                                                                             self.neg_sample_rate)
        print('******************** BPR ********************')
        self.embed_user = torch.nn.Embedding(self.num_users, self.emb_dim)  # , sparse=True
        # self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.embed_item = torch.nn.Embedding(self.num_items, self.emb_dim)  # , sparse=True
        # self.item_factors.weight.data.uniform_(-0.05, 0.05)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        print('P: ', self.embed_user)
        print('Q: ', self.embed_item)
        # self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() + LA.norm(self.item_factors.weight.data, 'fro').item())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.regularization)
        self.time = strftime('%Y%m%d-%H%M')

        print('********************* BPR Initialization Done *********************')
        self.to(self.device)

    def forward(self, users, pos_items, neg_items):
        # # Get the dot product per row
        # u = self.user_factors(user)
        # v = self.item_factors(item)
        # x = (u * v).sum(1)
        # return x
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        # neg_scores = torch.sum(torch.mul(users_emb.unsqueeze(1), neg_emb), dim=1)
        # neg_scores = torch.sum(neg_scores, dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.regularization * regularizer

        return mf_loss, reg_loss

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        # users = np.arange(self.num_users)
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)
        best_result = None

        start = time()
        for epoch_itr in range(1, num_epochs + 1):
            self.train()
            epoch_cost = 0.
            # epoch_cost1 = 0.
            # epoch_cost2 = 0.
            self.user_list, self.p_items, self.n_items = MP_Utility.neg_sampling_help_bc(self.num_users,
                                                                                         self.num_items,
                                                                                         self.train_df[0],
                                                                                         self.train_df[1],
                                                                                         self.neg_sample,
                                                                                         self.train_user_list)
            # start_time = time() * 1000.0
            batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                # tmp_cost, tmp_cost1, tmp_cost2 = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                #                                                   self.label_list[batch_idx])

                users = self.user_list[batch_idx]
                pos_items = self.p_items[batch_idx]
                neg_items = self.n_items[batch_idx]

                tmp_cost = self.train_model_per_batch(torch.tensor(users).to(self.device),
                                                        torch.tensor(pos_items).to(self.device),
                                                        torch.tensor(neg_items).to(self.device))
                epoch_cost += tmp_cost
                # epoch_cost1 += tmp_cost1
                # epoch_cost2 += tmp_cost2
                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, tmp_cost))
            epoch_train_time = time() - epoch_train_start
            epoch_info = ['epoch=%3d' % epoch_itr, 'loss=%.3f' % epoch_cost, 'train time=%.2f' % epoch_train_time]

            # self.train_model_help(epoch_itr)

            ## evaluation
            if (epoch_itr >= test_from and epoch_itr % test_step == 0) or epoch_itr == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                # test_score = evaluator.evaluate(self)
                test_score = evaluator.evaluate_vali(self)
                # test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
                #
                # print(test_score_str)
                updated, should_stop = early_stop.step(test_score, epoch_itr)

                # test_score_output = evaluator.evaluate(self)

                test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if k.startswith('NDCG')]
                # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        # torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        # if self.anneal_cap == 1: print(self.anneal)
                        best_result = test_score_output

                        # ndcg_test_all = evaluator.evaluate(self, mean=False)

                        # print(ndcg_test_all['DG@20'])
                        # print(mean(ndcg_test_all['DG@20']))
                        # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name,
                        #                               'mainstream_scores')
                        self.make_records()
                        similarity_file = os.path.join(similarity_dir, 'MF_BPR_scores')
                        if not os.path.exists(similarity_file):
                            os.mkdir(similarity_file)
                        with open(os.path.join(similarity_file, self.time + '_bpr_scores.npy'), 'wb') as f:
                            np.save(f, ndcg_test_all)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch_itr % print_step == 0:
                logger.info(', '.join(epoch_info))

            total_train_time = time() - start

        return best_result, total_train_time

    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        # g_droped = self.Graph
        #
        # for layer in range(self.n_layers):
        #     all_emb = torch.sparse.mm(g_droped, all_emb)
        #     embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items


    def train_model_per_batch(self, user_input, pos_item_input, neg_item_input):
        # reset gradients
        self.optimizer.zero_grad()

        loss = 0

        mf_loss, reg_loss = self.forward(user_input, pos_item_input, neg_item_input)
        loss = mf_loss + reg_loss

        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        # self.eval()

        # return (total_loss, total_loss1, total_loss2)
        return loss

    def get_rec(self):
        P, Q = self.embed_user.weight, self.embed_item.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        Rec = np.matmul(P, Q.T)
        return Rec

    def make_records(self):  # record all the results' details into files
        P, Q = self.embed_user.weight, self.embed_item.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)
        similarity_file = os.path.join(similarity_dir, 'PC_BPR_saves')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        with open(os.path.join(similarity_file,'P_MF_BPR.npy'), 'wb') as f:
            np.save(f, P)
        with open(os.path.join(similarity_file,'Q_MF_BPR.npy'), 'wb') as f:
            np.save(f, Q)
        # return P, Q

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            # eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            # P = torch.Tensor(self.user_list[user_ids]).int()
            # Q = torch.Tensor(self.item_list[user_ids]).int()
            Rec = self.get_rec()
            eval_output = Rec[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output