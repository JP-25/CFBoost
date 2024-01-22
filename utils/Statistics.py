import numpy as np

class Statistics:
    def __init__(self, name='AVG'):
        self.name = name
        self.history = []
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.history.append(val)
        self.sum += val
        self.cnt += 1

    def over_write(self, arr):
        # if len(self.history) == 0:
        #     self.history = arr
        # else:
        #     self.history = (np.array(self.history) + np.array(self.history)).tolist()
        self.history = arr
        # print(self.history == arr)
        # print(len(self.history))
        # print(self.history)

    @property
    def mean_std(self):
        # mean = self.sum / self.cnt
        mean = np.mean(self.history)
        std = np.std(self.history)
        return mean, std

    @property
    def mean(self):
        # return self.sum / self.cnt
        return np.mean(self.history)

    @property
    def std(self):
        return np.std(self.history)