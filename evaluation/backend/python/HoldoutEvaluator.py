import math
from collections import OrderedDict
from collections import defaultdict
import numpy as np
import os
from time import strftime

from utils.Statistics import Statistics

class HoldoutEvaluator:
    def __init__(self, top_k, eval_pos, eval_target, eval_neg_candidates=None):
        self.top_k = top_k
        self.max_k = max(top_k)
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.eval_neg_candidates = eval_neg_candidates
        self.time = strftime('%Y%m%d-%H%M')

    def init_score_cumulator(self):
        score_cumulator = OrderedDict()
        # for metric in ['Prec', 'Recall', 'NDCG']:
        #     score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in self.top_k}
        # score_cumulator['NDCG'] = {k: Statistics('%s@%d' % ('NDCG', k)) for k in self.top_k}
        ## here
        for metric in ['NDCG', 'DG']:
            score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in self.top_k}
        return score_cumulator

    # def init_score_cumulator_ndcg(self):
    #     ndcg = OrderedDict()
    #     for k in self.top_k:
    #         ndcg.setdefault(k, [])
    #     return ndcg

    def compute_metrics(self, topk, target, score_cumulator=None):
        if score_cumulator is None:
            score_cumulator = self.init_score_cumulator()
        hits = []
        for idx, u in enumerate(target):
            pred_u = topk[idx]
            target_u = target[u]
            num_target_items = len(target_u)
            for k in self.top_k:
                pred_k = pred_u[:k]
                hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
                num_hits = len(hits_k)

                idcg_k = 0.0
                for i in range(1, min(num_target_items, k) + 1):
                    idcg_k += 1 / math.log(i + 1, 2)

                dcg_k = 0.0
                for idx, item in hits_k:
                    dcg_k += 1 / math.log(idx + 1, 2)

                if num_hits:
                    pass
                
                # prec_k = num_hits / k
                # # recall_k = num_hits / min(num_target_items, k) ### not used
                # recall_k = num_hits / num_target_items
                ndcg_k = dcg_k / idcg_k

                # score_cumulator['Prec'][k].update(prec_k)
                # score_cumulator['Recall'][k].update(recall_k)
                score_cumulator['NDCG'][k].update(ndcg_k)
            
                hits.append(len(hits_k))
        return score_cumulator

    def compute_metrics_new(self, topk, target, dg, score_cumulator=None):
        if score_cumulator is None:
            score_cumulator = self.init_score_cumulator()
        hits = []
        for idx, u in enumerate(target):
            pred_u = topk[idx]
            target_u = target[u]
            num_target_items = len(target_u)
            for k in self.top_k:
                pred_k = pred_u[:k]
                hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
                num_hits = len(hits_k)

                idcg_k = 0.0
                for i in range(1, min(num_target_items, k) + 1):
                    idcg_k += 1 / math.log(i + 1, 2)

                dcg_k = 0.0
                for idx, item in hits_k:
                    dcg_k += 1 / math.log(idx + 1, 2)
                    dg[k][item] += 1 / math.log(idx + 1, 2)

                if num_hits:
                    pass

                # for t, item in enumerate(pred_k):
                #     dg[k][item] += 1 / math.log(t + 2, 2)

                # for t, item in hits_k:
                #     dg[k][item] += 1 / math.log(t + 1, 2)

                # prec_k = num_hits / k
                # # recall_k = num_hits / min(num_target_items, k) ### not used
                # recall_k = num_hits / num_target_items
                ndcg_k = dcg_k / idcg_k

                # score_cumulator['Prec'][k].update(prec_k)
                # score_cumulator['Recall'][k].update(recall_k)
                score_cumulator['NDCG'][k].update(ndcg_k)
                score_cumulator['DG'][k].over_write(dg[k])

                hits.append(len(hits_k))
        return score_cumulator

    # def compute_metrics_for_all_ndcg(self, topk, target, ndcg_scores=None):
    #     if ndcg_scores is None:
    #         ndcg_scores = self.init_score_cumulator_ndcg()
    #     hits = []
    #     for idx, u in enumerate(target):
    #         pred_u = topk[idx]
    #         target_u = target[u]
    #         num_target_items = len(target_u)
    #         for k in self.top_k:
    #             pred_k = pred_u[:k]
    #             hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
    #             num_hits = len(hits_k)
    #
    #             idcg_k = 0.0
    #             for i in range(1, min(num_target_items, k) + 1):
    #                 idcg_k += 1 / math.log(i + 1, 2)
    #
    #             dcg_k = 0.0
    #             for idx, item in hits_k:
    #                 dcg_k += 1 / math.log(idx + 1, 2)
    #
    #             if num_hits:
    #                 pass
    #
    #             # prec_k = num_hits / k
    #             # # recall_k = num_hits / min(num_target_items, k)
    #             # recall_k = num_hits / num_target_items
    #             ndcg_k = dcg_k / idcg_k
    #
    #             ndcg_scores.setdefault(k, []).append(ndcg_k)
    #
    #             hits.append(len(hits_k))
    #     return ndcg_scores