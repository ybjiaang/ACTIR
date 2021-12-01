import numpy as np
import torch 
from torch import nn
import math
from torch.autograd import Variable
import pandas as pd

# following this paper: https://arxiv.org/pdf/1910.00270.pdf
# x: temperature, feeling temperature, wind speed, humidity 
# y: cnt
class BikeSharingDataset(object):
    def __init__(self, cvs_dir = "dataset/Bike-Sharing-Dataset/hour.csv", test_season = 0, test_finetune_size = 10, test_unlabled_size=32, year = 1):
        super(BikeSharingDataset, self).__init__()
        self.cvs_dir = cvs_dir
        self.num_total_envs = 4
        self.num_train_evns = 3
        self.test_season = test_season
        self.test_finetune_size = test_finetune_size
        self.test_unlabled_size = test_unlabled_size
        self.year = year

        self.read_files()
        self.input_dim = self.test_data[0].shape[1]

    def read_files(self):
        data= pd.read_csv(self.cvs_dir, usecols = ['season', 'yr', 'temp', 'atemp', 'hum', 'windspeed', 'cnt'])
        data_array = data.to_numpy()
        self.train_data_by_season = []
        self.val_data_by_season = []

        for i in range(4):
            season_data = data_array[data_array[:,0] == i + 1, 1:]
            season_data = season_data[season_data[:,0] == self.year, 1:]
            total_num = season_data.shape[0]
            season_data_permutated = torch.Tensor(season_data[np.random.permutation(total_num)])
            if i == self.test_season:
                self.test_data_finetune = (season_data_permutated[:self.test_finetune_size, :-1], season_data_permutated[:self.test_finetune_size, -2:-1])
                self.test_data_unlabled = (season_data_permutated[self.test_finetune_size: self.test_finetune_size + self.test_unlabled_size, :-1], 
                    season_data_permutated[self.test_finetune_size: self.test_finetune_size + self.test_unlabled_size, -2:-1])
                self.test_data = (season_data_permutated[: self.test_finetune_size + self.test_unlabled_size, :-1], 
                    season_data_permutated[: self.test_finetune_size + self.test_unlabled_size, -2:-1])
            else:
                train_num = int(total_num * 0.8)
                train_x = season_data_permutated[:train_num, :-1]
                train_y = season_data_permutated[:train_num, -2:-1]
                val_x = season_data_permutated[train_num:, :-1]
                val_y = season_data_permutated[train_num:, -2:-1]
                self.train_data_by_season.append((train_x, train_y))
                self.val_data_by_season.append((val_x, val_y))

    def sample_envs(self, env_ind=0, train_val_test = 0):
        # train
        if train_val_test == 0:
            return self.train_data_by_season[env_ind]

        # val
        if train_val_test == 1:
            return self.val_data_by_season[env_ind]

        if train_val_test == 2:
            return self.test_data_finetune, self.test_data_unlabled, self.test_data


if __name__ == '__main__':
    env = BikeSharingDataset()