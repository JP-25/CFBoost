import torch
import numpy as np
import random as rd
from utils import MP_Utility

class BatchSampler:
    def __init__(self, data_size, batch_size, drop_remain=False, shuffle=False):
        self.data_size = data_size
        self.batch_size = batch_size
        self.drop_remain = drop_remain
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.data_size)
        else:
            perm = range(self.data_size)

        batch_idx = []
        for idx in perm:
            batch_idx.append(idx)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []
        if len(batch_idx) > 0 and not self.drop_remain:
            yield batch_idx

    def __len__(self):
        if self.drop_remain:
            return self.data_size // self.batch_size
        else:
            return int(np.ceil(self.data_size / self.batch_size))

class DataBatcher:
    def __init__(self, *data_source, batch_size, drop_remain=False, shuffle=False):
        self.data_source = list(data_source)
        self.batch_size = batch_size
        self.drop_remain = drop_remain
        self.shuffle = shuffle

        for i, d in enumerate(self.data_source):
            if isinstance(d, list):
                self.data_source[i] = np.array(d)

        self.data_size = len(self.data_source[0])
        if len(self.data_source)> 1:
            flag = np.all([len(src) == self.data_size for src in self.data_source])
            if not flag:
                raise ValueError("All elements in data_source should have same lengths")

        self.sampler = BatchSampler(self.data_size, self.batch_size, self.drop_remain, self.shuffle)
        self.iterator = iter(self.sampler)

        self.n = 0

    def __next__(self):
        batch_idx = next(self.iterator)
        batch_data = tuple([data[batch_idx] for data in self.data_source])

        if len(batch_data) == 1:
            batch_data = batch_data[0]
        return batch_data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.sampler)



# class BatchSampler_pos_neg:
#     def __init__(self, data_size, train_user_list, items, user_pop, item_pop, neg_sample, batch_size, drop_remain=False, shuffle=False):
#         self.data_size = data_size
#         self.batch_size = batch_size
#         self.drop_remain = drop_remain
#         self.shuffle = shuffle
#         self.train_user_list = train_user_list
#         self.items = items
#         self.user_pop = user_pop
#         self.item_pop = item_pop
#         self.neg_sample = neg_sample
#
#     def __iter__(self):
#         if self.shuffle:
#             perm = np.random.permutation(self.data_size)
#         else:
#             perm = range(self.data_size)
#
#         batch_idx = []
#         pos_items = []
#         neg_items = []
#         u_pop = []
#         pos_pop = []
#         neg_pop = []
#
#         for idx in perm:
#             batch_idx.append(idx)
#             u_pop.append(self.user_pop[idx])
#             ## edit part
#             if not self.train_user_list[idx]:
#                 pos_items.append(0) # this case barely show
#             else:
#                 pos_i = rd.choice(self.train_user_list[idx])
#                 pos_items.append(pos_i)
#                 pos_pop.append(self.item_pop[pos_i])
#             if self.neg_sample == 1:
#                 while True:
#                     neg_item = rd.choice(self.items)
#                     if neg_item not in self.train_user_list[idx]:
#                         neg_items.append(neg_item)
#                         break
#                 neg_pop.append(self.item_pop[neg_item])
#             else:
#                 neg_idx = MP_Utility.randint_choice(self.items, size=self.neg_sample, exclusion=self.train_user_list[idx])
#                 neg_items.append(neg_idx)
#                 neg_pop.append(self.item_pop[neg_idx])
#
#             if len(batch_idx) == self.batch_size:
#                 yield batch_idx, pos_items, neg_items, u_pop, pos_pop, neg_pop
#                 batch_idx = []
#                 pos_items = []
#                 neg_items = []
#                 u_pop = []
#                 pos_pop = []
#                 neg_pop = []
#         if len(batch_idx) > 0 and not self.drop_remain:
#             yield batch_idx, pos_items, neg_items, u_pop, pos_pop, neg_pop
#
#     def __len__(self):
#         if self.drop_remain:
#             return self.data_size // self.batch_size
#         else:
#             return int(np.ceil(self.data_size / self.batch_size))
#
# class DataBatcher_pos_neg:
#     def __init__(self, *data_source, train_user_list, items, user_pop, item_pop, neg_sample, batch_size, drop_remain=False, shuffle=False):
#         self.data_source = list(data_source)
#         self.batch_size = batch_size
#         self.drop_remain = drop_remain
#         self.shuffle = shuffle
#         # self.train_user_list = train_user_list
#         # self.items = items
#         # self.user_pop = user_pop
#         # self.item_pop = item_pop
#
#         for i, d in enumerate(self.data_source):
#             if isinstance(d, list):
#                 self.data_source[i] = np.array(d)
#
#         self.data_size = len(self.data_source[0])
#         if len(self.data_source) > 1:
#             flag = np.all([len(src) == self.data_size for src in self.data_source])
#             if not flag:
#                 raise ValueError("All elements in data_source should have same lengths")
#
#         self.sampler = BatchSampler_pos_neg(self.data_size, train_user_list, items, user_pop, item_pop, neg_sample, self.batch_size,
#                              self.drop_remain, self.shuffle)
#         self.iterator = iter(self.sampler)
#
#         self.n = 0
#
#     def __next__(self):
#         batch_idx, pos, neg, u_pop, pos_pop, neg_pop = next(self.iterator)
#         batch_data = tuple([data[batch_idx] for data in self.data_source])
#
#         if len(batch_data) == 1:
#             batch_data = batch_data[0]
#         return batch_data, np.array(pos), np.array(neg), np.array(u_pop), np.array(pos_pop), np.array(neg_pop)
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return len(self.sampler)