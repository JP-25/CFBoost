import os
import math
import copy
import pickle
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from concurrent import futures
from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Logger #, set_random_seed
from sklearn.cluster import KMeans
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_distances
import warnings
from tqdm import tqdm
from time import strftime
from experiment import EarlyStop

warnings.filterwarnings("ignore")

# early_stop change
class LOCA_VAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(LOCA_VAE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        # CoLA conf.
        self.num_local = model_conf['num_local']
        self.anchor_selection = model_conf['anchor_selection']
        self.dist_type = model_conf['dist_type']
        self.kernel_type = model_conf['kernel_type']
        self.train_h = model_conf['train_h']
        self.test_h = model_conf['test_h']
        self.embedding_type = model_conf['embedding']
        self.model_type = 'MultVAE'

        self.user_embedding = self.load_embedding(self.embedding_type)
        self.kernel_matrix = self.build_kernel_matrix()
        self.candidate_users = []
        for kernel in self.kernel_matrix:
            self.candidate_users.append(kernel[2].nonzero()[0])
            # print("L47: ", kernel[2].nonzero())
            # print("L48: ", kernel[2].nonzero()[0])

        self.num_local_threads = model_conf['num_local_threads']

        # Local conf.
        self.model_conf = model_conf
        self.local_models = []
        self.local_dirs = []

        self.device = device

        self.share_memory()

    def train_single_model(self, local_num):
        # KEY: make it executed in independent threads.
        evaluator, early_stop, local_conf = self.common_object
        early_stop = copy.deepcopy(early_stop)

        logger = self.init_local_logger(local_num)

        logger.info('Local %d train start...' % local_num)
        # build local model
        train_weight = torch.tensor(self.kernel_matrix[local_num][0])  # vector
        test_weight = torch.tensor(self.kernel_matrix[local_num][2])  # vector
        local_model = LocalWrapper(local_num, train_weight, test_weight, self.candidate_users[local_num],
                                   self.dataset, self.model_type, self.model_conf, self.device)

        # pass optimizer, evaluator, weight matrix with conf.
        local_best_score, local_train_time = local_model.train_model(self.dataset, evaluator, early_stop, logger, local_conf)

        # train done.
        logger.info('Local %d done...' % local_num)
        return local_best_score, local_train_time

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'bias_scores')
        similarity_file = os.path.join(similarity_dir, 'loca_vae_folder')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        path_ = os.path.join(similarity_file, 'loca_vae_scores_' + strftime('%Y%m%d-%H%M'))
        if not os.path.exists(path_):
            os.mkdir(path_)

        self.base_dir = logger.log_dir
        logger.info("Train coverage : %.5f (Average), %.5f (Max), %.5f (Min)" % (
            np.mean(self.train_coverage), max(self.train_coverage), min(self.train_coverage)))
        logger.info("Test coverage : %.5f (Average), %.5f (Max), %.5f (Min)" % (
            np.mean(self.test_coverage), max(self.test_coverage), min(self.test_coverage)))

        self.common_object = (evaluator, early_stop, config)

        total_train_time = 0.0
        train_start = time()
        # train all models
        if self.num_local_threads > 1:
            with futures.ProcessPoolExecutor(max_workers=self.num_local_threads) as exe:
                ret = list(exe.map(self.train_single_model, list(range(self.num_local))))
        else:
            for i in range(self.num_local):
                local_best_score, local_train_time = self.train_single_model(i)
                total_train_time += local_train_time

        total_train_time = time() - train_start
        # test_score = evaluator.evaluate(self)

        test_score, ndcg_test_all = evaluator.evaluate_full_boost(self)
        test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
        logger.info(', '.join(test_score_str))

        # ndcg_test_all = evaluator.evaluate(self, mean=False)
        with open(os.path.join(path_, str(self.num_local) + '_loca_vae_test_scores.npy'), 'wb') as f:
            np.save(f, ndcg_test_all)
        return test_score, total_train_time

    def init_local_logger(self, local_num):
        exp_dir = os.path.join(self.base_dir, 'local_%d' % local_num)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        logger = Logger(exp_dir)
        # self.exp_logger.append(logger)
        return logger

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        batch_pos_matrix = eval_pos_matrix[user_ids]

        # ======local output
        eval_output = torch.zeros((self.num_users, self.num_items), dtype=torch.float32)
        weights_sum = torch.zeros((self.num_users, 1), dtype=torch.float32)
        for local_num in range(self.num_local):
            local_dir = os.path.join(self.base_dir, 'local_%d' % local_num)
            train_weight = torch.tensor(self.kernel_matrix[local_num][0])  # vector
            test_weight = torch.tensor(self.kernel_matrix[local_num][2])  # vector
            local_model = LocalWrapper(local_num, train_weight, test_weight, self.candidate_users[local_num], self.dataset, self.model_type, self.model_conf, self.device)
            local_model.restore(local_dir)

            cand_users = self.candidate_users[local_num]
            cand_eval_users = [u for u in user_ids if u in cand_users]
            # local_pred: (# cand_eval_users, # items)
            local_pred = local_model.predict(cand_eval_users, eval_pos_matrix, eval_items)

            weights = test_weight
            local_weights = weights[cand_eval_users].view(-1, 1)
            eval_output[cand_eval_users] += torch.FloatTensor(local_pred) * local_weights
            # eval_output[cand_users] += local_pred[cand_users] * weights.view(-1, 1)[cand_users]
            weights_sum[cand_eval_users] += local_weights

        eval_output = eval_output[user_ids]
        weights_sum = weights_sum[user_ids]

        eval_output /= weights_sum
        eval_output[torch.isnan(eval_output)] = 0.0  # float('-inf')

        # ====== global output
        with open(os.path.join(self.dataset.data_dir, self.dataset.data_name, 'output', self.model_type + '_output.p'), 'rb') as f:
            global_pred = pickle.load(f)[user_ids]
        # global_pred = np.zeros_like(eval_output)
        zero_mask = torch.eq(weights_sum, 0).float()

        # ====== aggregate
        eval_output = torch.Tensor(global_pred) * zero_mask + eval_output
        eval_output = eval_output.numpy()
        eval_output[batch_pos_matrix.nonzero()] = float('-inf')

        return eval_output

    def restore(self, log_dir):
        self.base_dir = log_dir
        pass

    def dist(self, a, anchor=None):
        # anchor --> matrix
        if anchor is None:
            if np.sum(a @ a.T) == 0:
                return 999

            numer = a @ a.T
            norm = np.reshape(np.linalg.norm(a, axis=1), (-1, 1))
            denom = np.maximum(norm * norm.T, 1e-10)
            return (2 / math.pi) * np.arccos(np.clip(numer / denom, -1, 1))

        # anchor --> vector
        else:
            a_anchor = np.reshape(a[anchor], (1, -1))
            if np.sum(a_anchor @ a.T) == 0:
                return 999

            numer = a_anchor @ a.T  # (1, user)
            norm = np.reshape(np.linalg.norm(a, axis=1), (-1, 1))
            denom = np.maximum(norm[anchor] * norm.T, 1e-10)  # (1, user)
            return np.squeeze((2 / math.pi) * np.arccos(np.clip(numer / denom, -1, 1)))

    def kernel(self, a, h=0.8, kernel_type='Epanechnikov', anchor=None):
        if anchor is None:
            if kernel_type.lower() == 'epanechnikov':
                return (3 / 4) * np.maximum(1 - np.power(self.dist(a) / h, 2), 0)
            if kernel_type.lower() == 'uniform':
                return (self.dist(a) < h)
            if kernel_type.lower() == 'triangular':
                return max((1 - self.dist(a) / h), 0)
            if kernel_type.lower() == 'random':
                return np.random.uniform(0, 1) * (self.dist(a) < h)
        else:
            if kernel_type.lower() == 'epanechnikov':
                return (3 / 4) * np.maximum(1 - np.power(self.dist(a, anchor) / h, 2), 0)
            if kernel_type.lower() == 'uniform':
                return (self.dist(a, anchor) < h)
            if kernel_type.lower() == 'triangular':
                return max((1 - self.dist(a, anchor) / h), 0)
            if kernel_type.lower() == 'random':
                return np.random.uniform(0, 1) * (self.dist(a, anchor) < h)

    def kernel_weight_matrix(self, user_kernel, item_kernel):
        user_kernel = torch.FloatTensor(user_kernel).view(-1, 1)
        item_kernel = torch.FloatTensor(item_kernel).view(1, -1)
        return torch.matmul(user_kernel, item_kernel)

    def load_embedding(self, embedding_type):
        with open(os.path.join(self.dataset.data_dir, self.dataset.data_name, 'embedding', embedding_type + '_user.p'), 'rb') as f:
            embedding = pickle.load(f)
        return embedding

    def build_kernel_matrix(self):
        # for each local model
        if self.anchor_selection == 'kmeans':
            user_dist_with_centers = KMeans(n_clusters=self.num_local, random_state=0).fit_transform(self.user_embedding)
            user_anchors = np.argsort(user_dist_with_centers, axis=0)[0]
        elif self.anchor_selection == 'random':
            user_anchors = np.random.choice(self.num_users, size=self.num_local, replace=False)
        elif self.anchor_selection == 'coverage':
            user_anchors = np.zeros(self.num_local, dtype=int)
            W_mat = np.zeros((self.num_users, self.num_users), dtype=int)
            # if j is covered by i, W_mat[u,i] = 1.
            for u in tqdm(range(0, self.num_users, 10)):
                u_cover = np.nonzero(self.kernel(self.user_embedding, self.test_h, self.kernel_type, u))[0]
                W_mat[u, u_cover] = 1
        else:
            raise Exception("Choose correct self.anchor_selection")
        item_anchors = np.random.choice(self.num_items, size=self.num_local, replace=False)

        # for each local model
        kernel_ret = []
        self.train_coverage = []
        self.test_coverage = []
        for t in range(self.num_local):
            # select anchor
            if self.anchor_selection == 'coverage':
                user_anchors[t] = np.argmax(np.sum(W_mat, axis=1))  # maximum coverage becomes new anchor
                new_covered = np.nonzero(W_mat[user_anchors[t]])[0]  # elements which are covered
                W_mat[:, new_covered] = 0  # eliminate elements which are covered

            user_anchor_t = user_anchors[t]
            item_anchor_t = item_anchors[t]

            # train user kernel
            train_user_kernel_t = self.kernel(self.user_embedding, self.train_h, self.kernel_type, user_anchor_t)  # .astype(np.float32)
            train_item_kernel_t = np.ones(self.num_items)  # .astype(np.float32)
            train_coverage_size = (np.count_nonzero(train_user_kernel_t) * np.count_nonzero(train_item_kernel_t)) / (self.num_users * self.num_items)

            # test user kernel
            test_user_kernel_t = self.kernel(self.user_embedding, self.test_h, self.kernel_type, user_anchor_t)  # .astype(np.float32)
            test_item_kernel_t = np.ones(self.num_items)  # .astype(np.float32)
            test_coverage_size = (np.count_nonzero(test_user_kernel_t) * np.count_nonzero(test_item_kernel_t)) / (self.num_users * self.num_items)

            kernel_ret.append((train_user_kernel_t, train_item_kernel_t, test_user_kernel_t, test_item_kernel_t))
            # print("L259: ", (train_user_kernel_t, train_item_kernel_t, test_user_kernel_t, test_item_kernel_t))
            self.train_coverage.append(train_coverage_size)
            self.test_coverage.append(test_coverage_size)

            print("Anchor %3d coverage : %.5f (train), %.5f (test)" % (t, train_coverage_size, test_coverage_size))
        print("Train coverage : %.5f (Average), %.5f (Max), %.5f (Min)" % (
            np.mean(self.train_coverage), max(self.train_coverage), min(self.train_coverage)))
        print("Test coverage : %.5f (Average), %.5f (Max), %.5f (Min)" % (
            np.mean(self.test_coverage), max(self.test_coverage), min(self.test_coverage)))
        return kernel_ret


"""
Wrapper class for local model.
Local model can be any type.
"""


class LocalWrapper(BaseRecommender):
    def __init__(self, local_num, train_weight, test_weight, candidate_users, dataset, model_type, model_conf, device):
        super(LocalWrapper, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.model_type = model_type
        self.model_conf = model_conf
        self.device = device

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']

        self.local_num = local_num
        self.train_weight = train_weight
        self.test_weight = test_weight
        self.candidate_users = candidate_users
        self.local_model = self.build_model()
        # self.es = EarlyStop(10, 'mean')

        self.optimizer = self.local_model.optimizer

    def kernel_weight_matrix(self, user_kernel, item_kernel):
        user_kernel = torch.FloatTensor(user_kernel).view(-1, 1)
        item_kernel = torch.FloatTensor(item_kernel).view(1, -1)
        return torch.matmul(user_kernel, item_kernel)

    def build_model(self):
        # return actual model
        import model as m
        model = getattr(m, self.model_type)(self.dataset, self.model_conf, self.device)
        return model

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        # exp conf.
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        # prepare data
        users = np.arange(self.num_users)
        train_matrix = dataset.train_matrix

        # for each epoch
        start = time()
        best_result = None

        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            num_batches = len(batch_loader)

            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_train = np.where(self.train_weight[batch_idx] != 0)[0]
                if len(batch_train) == 0:
                    continue

                batch_input = torch.tensor(train_matrix[batch_idx].toarray(), requires_grad=False, device=self.device, dtype=torch.float)
                batch_weight = self.train_weight[batch_idx].to(self.device)  # TEMP

                batch_input = batch_input[batch_train]
                batch_weight = batch_weight[batch_train]

                batch_loss = self.local_model.train_model_per_batch(batch_input, batch_weight)

                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))

            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate_partial_vali(self, candidate_users=self.candidate_users)
                # test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]

                updated, should_stop = early_stop.step(test_score, epoch)
                test_score_output = evaluator.evaluate_partial(self, candidate_users=self.candidate_users)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]
                # test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output if
                #                   k.startswith('NDCG')]

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.local_model.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        best_result = test_score_output

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info('[Local %3d] ' % self.local_num + ', '.join(epoch_info))

        total_train_time = time() - start

        # return early_stop.best_score, total_train_time
        return best_result, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.local_model.eval()
        eval_output = self.local_model.predict(user_ids, eval_pos_matrix, eval_items)
        self.local_model.train()

        # apply weights
        return eval_output

    def restore(self, log_dir):
        self.local_model.restore(log_dir)
