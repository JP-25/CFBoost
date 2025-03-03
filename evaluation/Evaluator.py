import numpy as np
import os
from evaluation.backend import HoldoutEvaluator, predict_topk
from dataloader.DataBatcher import DataBatcher
from collections import defaultdict


class Evaluator:
    def __init__(self, eval_pos, eval_target, vali_target, eval_neg_candidates, ks, num_users=None, num_items=None, item_id=None):
        self.top_k = ks if isinstance(ks, list) else [ks]
        self.max_k = max(self.top_k)

        self.batch_size = 1024
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.eval_neg_candidates = eval_neg_candidates
        self.item_id = item_id

        self.vali_target = vali_target

        if num_users == None or num_items == None:
            self.num_users, self.num_items = eval_pos.shape
        else:
            self.num_users, self.num_items = num_users, num_items

        ks = sorted(ks) if isinstance(ks, list) else [ks]
        self.eval_runner = HoldoutEvaluator(ks, self.eval_pos, self.eval_target, self.eval_neg_candidates)
        self.eval_runner_vali = HoldoutEvaluator(ks, self.eval_pos, self.vali_target, self.eval_neg_candidates)

    ## for testing set
    def evaluate(self, model, mean=True):
        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = list(self.eval_target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None
        dg = defaultdict(list)
        for k in self.top_k:
            dg[k] = [0] * self.eval_pos.shape[1]

        for batch_user_ids in user_iterator:
            # need refactoring
            # print(score_cumulator)
            # print(dg)
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            # print(batch_topk)
            # score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
            score_cumulator = self.eval_runner.compute_metrics_new(batch_topk, batch_eval_target, dg, score_cumulator)
        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                if mean:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
                else:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].history
        # score_by_ks = score_cumulator['NDCG']
        # for k in score_by_ks:
        #     if mean:
        #         scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].mean
        #     else:
        #         scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].history
        # return
        model.train()
        return scores

    def evaluate_full_boost(self, model):
        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = list(self.eval_target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None
        dg = defaultdict(list)
        for k in self.top_k:
            dg[k] = [0] * self.eval_pos.shape[1]

        for batch_user_ids in user_iterator:
            # need refactoring
            # print(score_cumulator)
            # print(dg)
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            # print(batch_topk)
            # score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
            score_cumulator = self.eval_runner.compute_metrics_new(batch_topk, batch_eval_target, dg, score_cumulator)
        scores_mean = {}
        scores_all = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                scores_mean['%s@%d' % (metric, k)] = score_by_ks[k].mean
                scores_all['%s@%d' % (metric, k)] = score_by_ks[k].history
        model.train()
        return scores_mean, scores_all

    ## for validation set
    def evaluate_vali(self, model, mean=True):
        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = list(self.vali_target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None

        for batch_user_ids in user_iterator:
            # need refactoring
            batch_eval_target = {u: self.vali_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            score_cumulator = self.eval_runner_vali.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
        scores = {}
        # for metric in score_cumulator:
        #     score_by_ks = score_cumulator[metric]
        #     for k in score_by_ks:
        #         if mean:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
        #         else:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].history

        score_by_ks = score_cumulator['NDCG']
        for k in score_by_ks:
            if mean:
                scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].mean
            else:
                scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].history
        model.train()
        return scores

    # print out all ndcg and keep ndcg mean final results
    def evaluate_vali_MOE(self, model):
        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = list(self.vali_target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None

        for batch_user_ids in user_iterator:
            # need refactoring
            batch_eval_target = {u: self.vali_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            score_cumulator = self.eval_runner_vali.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
        scores_mean = {}
        scores_all = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                scores_mean['%s@%d' % (metric, k)] = score_by_ks[k].mean
                scores_all['%s@%d' % (metric, k)] = score_by_ks[k].history
        # return
        model.train()
        return scores_mean, scores_all

    ## for validation set partial
    def evaluate_partial_vali(self, model, candidate_users=None, mean=True):
        if candidate_users is None:
            print('Candidate users are not privided. Evaluate on all users')
            return self.evaluate(model)

        # Switch to eval mode
        model.eval()

        # eval users
        # eval_users = list(self.vali_target.keys())
        # user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None
        eval_users = candidate_users
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)


        for batch_user_ids in user_iterator:
            # need refactoring
            batch_eval_target = {u: self.vali_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            score_cumulator = self.eval_runner_vali.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
        # scores = {}
        # for metric in score_cumulator:
        #     score_by_ks = score_cumulator[metric]
        #     for k in score_by_ks:
        #         if mean:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
        #         else:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].history
        scores = {}
        score_by_ks = score_cumulator['NDCG']
        for k in score_by_ks:
            if mean:
                scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].mean
            else:
                scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].history
        # return
        model.train()
        return scores

    # if write model based on LOCA, modify the DG calculation like the previous part
    def evaluate_partial(self, model, candidate_users=None, mean=True):
        if candidate_users is None:
            print('Candidate users are not privided. Evaluate on all users')
            return self.evaluate(model)

        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = candidate_users
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)

        score_cumulator = None
        dg = defaultdict(list)
        for k in self.top_k:
            dg[k] = [0] * self.eval_pos.shape[1]

        for batch_user_ids in user_iterator:
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            #   make prediction
            batch_pred = model.predict(batch_user_ids, self.eval_pos)

            # compute metrics
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            # score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
            score_cumulator = self.eval_runner.compute_metrics_new(batch_topk, batch_eval_target, dg, score_cumulator)

        # scores = {}
        # for metric in score_cumulator:
        #     score_by_ks = score_cumulator[metric]
        #     for k in score_by_ks:
        #         if mean:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
        #         else:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].history
        scores = {}
        score_by_ks = score_cumulator['NDCG']
        for k in score_by_ks:
            if mean:
                scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].mean
            else:
                scores['%s@%d' % ('NDCG', k)] = score_by_ks[k].history

        # return
        model.train()
        return scores

    def update(self, eval_pos=None, eval_target=None, eval_neg_candidates=None):
        if eval_pos is not None:
            self.eval_pos = eval_pos
        if eval_target is not None:
            self.eval_target = eval_target
        if eval_neg_candidates is not None:
            self.eval_neg_candidates = eval_neg_candidates

    def evaluate_all(self, rec, mean=True):
        # score_cumulator = None
        # eval_users = list(self.eval_target.keys())
        # ndcg_scores = None
        #
        # # need refactoring
        # batch_eval_target = {u: self.eval_target[u] for u in eval_users}
        # eval_output = rec[eval_users, :]
        # eval_output[self.eval_pos[eval_users].nonzero()] = float('-inf')
        # # compute metrics
        # batch_topk = predict_topk(eval_output.astype(np.float32), self.max_k).astype(np.int64)
        # score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
        # ndcg_scores = self.eval_runner.compute_metrics_for_all_ndcg(batch_topk, batch_eval_target, ndcg_scores)
        # scores = {}
        # for metric in score_cumulator:
        #     score_by_ks = score_cumulator[metric]
        #     for k in score_by_ks:
        #         if mean:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
        #         else:
        #             scores['%s@%d' % (metric, k)] = score_by_ks[k].history
        # # return
        # return scores, ndcg_scores
        # eval users
        eval_users = list(self.eval_target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None
        dg = defaultdict(list)
        for k in self.top_k:
            dg[k] = [0] * self.eval_pos.shape[1]

        for batch_user_ids in user_iterator:
            # need refactoring
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            #   make prediction
            # batch_pred = model.predict(batch_user_ids, self.eval_pos)
            eval_output = rec[batch_user_ids, :]
            eval_output[self.eval_pos[batch_user_ids].nonzero()] = float('-inf')

            # compute metrics
            batch_topk = predict_topk(eval_output.astype(np.float32), self.max_k).astype(np.int64)
            # score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
            score_cumulator = self.eval_runner.compute_metrics_new(batch_topk, batch_eval_target, dg, score_cumulator)

        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                if mean:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
                else:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].history

        return scores


    def evaluate_all_boost(self, rec):
        eval_users = list(self.eval_target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None
        dg = defaultdict(list)
        for k in self.top_k:
            dg[k] = [0] * self.eval_pos.shape[1]

        for batch_user_ids in user_iterator:
            # need refactoring
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            #   make prediction
            # batch_pred = model.predict(batch_user_ids, self.eval_pos)
            eval_output = rec[batch_user_ids, :]
            eval_output[self.eval_pos[batch_user_ids].nonzero()] = float('-inf')

            # compute metrics
            batch_topk = predict_topk(eval_output.astype(np.float32), self.max_k).astype(np.int64)
            # score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
            score_cumulator = self.eval_runner.compute_metrics_new(batch_topk, batch_eval_target, dg, score_cumulator)

        scores_mean = {}
        scores_all = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                scores_mean['%s@%d' % (metric, k)] = score_by_ks[k].mean
                scores_all['%s@%d' % (metric, k)] = score_by_ks[k].history

        return scores_mean, scores_all

    def evaluate_vali_batch(self, rec, target, mean=True):
        # eval users
        self.eval_runner_vali_batch = HoldoutEvaluator([20], self.eval_pos, target, self.eval_neg_candidates)
        eval_users = list(target.keys())
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)
        score_cumulator = None

        for batch_user_ids in user_iterator:
            # need refactoring
            batch_eval_target = {u: target[u] for u in batch_user_ids}
            #   make prediction
            eval_output = rec[batch_user_ids, :]
            # eval_output[self.eval_pos[batch_user_ids].nonzero()] = float('-inf')

            # compute metrics
            batch_topk = predict_topk(eval_output.astype(np.float32), self.max_k).astype(np.int64)
            score_cumulator = self.eval_runner_vali_batch.compute_metrics(batch_topk, batch_eval_target, score_cumulator)
        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                if mean:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
                else:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].history
        # return
        return scores
