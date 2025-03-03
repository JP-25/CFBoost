import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
# from dataloader.DataBatcher import DataBatcher_pos_neg
from utils import Tool
from utils import MP_Utility
from experiment import EarlyStop
from time import strftime


class BC_Loss(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(BC_Loss, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_df = dataset.train_df

        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']

        self.emb_dim = model_conf['emb_dim']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        help_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name)
        help_dir = os.path.join(help_dir, 'bias_scores')
        self.train_user_list = np.load(help_dir + '/train_user_list.npy', allow_pickle=True)
        self.user_pop = np.load(help_dir + '/user_activeness.npy', allow_pickle=True)
        self.item_pop = np.load(help_dir + '/item_popularity.npy', allow_pickle=True)
        self.neg_sample = model_conf['neg_sample_rate']
        self.tau1 = model_conf['tau1']
        self.tau2 = model_conf['tau2']
        self.w_lambda = model_conf['w_lambda']
        self.freeze_epoch = model_conf['freeze_epoch']

        self.embed_user = nn.Embedding(self.num_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.num_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        # nn.init.xavier_uniform_(self.embed_user.weight)
        # nn.init.xavier_uniform_(self.embed_item.weight)

        self.w = nn.Embedding(self.emb_dim, 1)
        self.w_user = nn.Embedding(self.emb_dim, 1)
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_user.weight)
        # nn.init.xavier_uniform_(self.w.weight)
        # nn.init.xavier_uniform_(self.w_user.weight)

        self.embed_user_pop = nn.Embedding(self.num_users, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.num_items, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
        # nn.init.xavier_uniform_(self.embed_user_pop.weight)
        # nn.init.xavier_uniform_(self.embed_item_pop.weight)


        self.lr = model_conf['lr']

        self.device = device

        self.time = strftime('%Y%m%d-%H%M')

        # self.p_items, self.n_items, self.pop_pos, self.pop_neg = MP_Utility.neg_sampling_helper(self.num_users,
        #                                                                                         self.num_items,
        #                                                                                         self.train_user_list,
        #                                                                                         self.neg_sample,
        #                                                                                         self.item_pop)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

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

    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):
        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)

        users_pop_emb = F.normalize(users_pop_emb, dim=-1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim=-1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim=-1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1),
                                   neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim=1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        pos_ratings = torch.cos(
            torch.arccos(torch.clamp(pos_ratings, -1 + 1e-7, 1 - 1e-7)) + (1 - torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim=1)

        loss1 = (1 - self.w_lambda) * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                       0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1 / self.batch_size

        regularizer2 = 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                       0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2 = regularizer2 / self.batch_size
        reg_loss = self.reg * (regularizer1 + regularizer2)

        reg_loss_freeze = self.reg * (regularizer2)
        reg_loss_norm = self.reg * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        # prepare dataset
        # dataset.set_eval_data('valid')
        users = np.arange(self.num_users)
        items = np.arange(self.num_items)

        train_matrix = dataset.train_matrix.toarray()
        # train_matrix = torch.FloatTensor(train_matrix)
        best_result = None
        # best_epoch = -1

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0

            self.user_list, self.p_items, self.n_items = MP_Utility.neg_sampling_help_bc(self.num_users,
                                                                                      self.num_items,
                                                                                      self.train_df[0],
                                                                                      self.train_df[1],
                                                                                      self.neg_sample,
                                                                                      self.train_user_list)

            batch_loader = DataBatcher(np.arange(len(self.user_list)), batch_size=self.batch_size, drop_remain=False, shuffle=True)
            # batch_overall = DataBatcher_pos_neg(users, train_user_list=self.train_user_list, items=items, user_pop=self.user_pop, item_pop=self.item_pop, neg_sample = self.neg_sample, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()

            # for b, (batch_idx, p_items, n_items, pop_u, pop_pos, pop_neg) in enumerate(batch_overall):
            for b, batch_idx in enumerate(batch_loader):

                # # batch_matrix = train_matrix[batch_idx].to(self.device)
                # batch_idx = torch.tensor(batch_idx).to(self.device)
                # p_items = torch.tensor(p_items).to(self.device)
                # n_items = torch.tensor(n_items).to(self.device)
                # pop_u = torch.LongTensor(pop_u).to(self.device)
                # pop_pos = torch.LongTensor(pop_pos).to(self.device)
                # pop_neg = torch.LongTensor(pop_neg).to(self.device)

                users = self.user_list[batch_idx]
                pos_items = self.p_items[batch_idx]
                neg_items = self.n_items[batch_idx]
                pop_u = self.user_pop[users]
                pop_pos = self.item_pop[pos_items]
                pop_neg = self.item_pop[neg_items]
                # batch_loss = self.train_model_per_batch(batch_idx, p_items, n_items)
                batch_loss = self.train_model_per_batch(torch.tensor(users.astype(int)).to(self.device),
                                                        torch.tensor(pos_items.astype(int)).to(self.device),
                                                        torch.tensor(neg_items.astype(int)).to(self.device),
                                                        torch.LongTensor(pop_u.astype(float)).to(self.device),
                                                        torch.LongTensor(pop_pos.astype(float)).to(self.device),
                                                        torch.LongTensor(pop_neg.astype(float)).to(self.device),
                                                        epoch)
                # batch_loss = self.train_model_per_batch(batch_idx, p_items, n_items, pop_u, pop_pos, pop_neg, epoch)

                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]
            similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
            if not os.path.exists(similarity_dir):
                os.mkdir(similarity_dir)

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate_vali(self)
                updated, should_stop = early_stop.step(test_score, epoch)

                # test_score_output = evaluator.evaluate(self)

                test_score_output, ndcg_test_all = evaluator.evaluate_full_boost(self)
                # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if
                                  k.startswith('NDCG')]

                # # used to draw graph for 5 groups of user changes
                # s_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'mainstream_scores')
                # s_file = os.path.join(s_dir, 'MultVAE_scores_distribution_more_epoch')
                # if not os.path.exists(s_file):
                #     os.mkdir(s_file)
                # ndcg_test_all = evaluator.evaluate(self, mean=False)
                # with open(os.path.join(s_file, str(epoch) + '_epoch.npy'), 'wb') as f:
                #     np.save(f, ndcg_test_all)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        # torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        # save scores for all users
                        # rec = self.predict_all()
                        best_result = test_score_output

                        # ndcg_test_all = evaluator.evaluate(self, mean=False)

                        # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name,
                        #                               'mainstream_scores')
                        similarity_file = os.path.join(similarity_dir, 'BC_LOSS_scores')
                        if not os.path.exists(similarity_file):
                            os.mkdir(similarity_file)
                        with open(os.path.join(similarity_file, self.time + '_bc_loss_scores.npy'), 'wb') as f:
                            np.save(f, ndcg_test_all)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return best_result, total_train_time

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)

    def train_model_per_batch(self, user, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, epoch):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = self.forward(user, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop)

        if epoch < self.freeze_epoch:
            loss = loss2 + reg_loss_freeze
        else:
            self.freeze_pop()
            loss = loss1 + loss2 + reg_loss

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        return loss

    def get_rec(self, user_ids):
        items = list(range(self.num_items))

        all_users, all_items = self.compute()
        users = all_users[torch.tensor(user_ids)]
        items = torch.transpose(all_items[torch.tensor(items)], 0, 1)

        # users, items = self.embed_user.weight[user_ids], self.embed_item.weight
        # items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items)

        # item_scores = torch.matmul(torch.transpose(items, 0, 1), self.w.weight)
        # user_scores = torch.matmul(users, self.w_user.weight)
        #
        # rubi_rating_both = (rate_batch - self.rubi_c) * (torch.sigmoid(user_scores)) * torch.transpose(
        #     torch.sigmoid(item_scores), 0, 1)
        #
        # # direct_minus = rate_batch - self.rubi_c * (torch.sigmoid(user_scores)) * torch.transpose(
        # #     torch.sigmoid(item_scores), 0, 1)

        return rate_batch.cpu().detach().numpy()



    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            # eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            # P = torch.Tensor(self.user_list[user_ids]).int()
            # Q = torch.Tensor(self.item_list[user_ids]).int()
            eval_output = self.get_rec(user_ids)
            # eval_output = eval_output[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output