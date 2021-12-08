import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm
from torch.autograd import grad

from misc import batchify, HSICLoss, env_batchify

class HSIC():
  def __init__(self, model, loss_fn, config):
    self.model = copy.deepcopy(model)
    self.config = config

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    self.bias_adjustment = torch.nn.Parameter(torch.zeros(1), requires_grad = False) 

  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(n_outer_loop)):
      for train in env_batchify(train_dataset, batch_size):
        loss = 0
        for env_ind in range(n_train_envs):
          x, y = train[env_ind]
          f_beta, _ = self.model(x)
          error = HSICLoss(f_beta, y - f_beta)
          loss += error

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      if t % 10 == 0 and self.config.verbose:
        print(loss.item()/(n_train_envs*batch_size))

    # calculate adjustment
    total_num_y = 0
    for env_ind in range(n_train_envs):
        for x, y in batchify(train_dataset[env_ind], batch_size):
          total_num_y += y.shape[0]
          self.bias_adjustment += torch.sum(y, dim = 0)

    self.bias_adjustment /= total_num_y

  def test(self, test_dataset, batch_size = 32):
    # print(self.model.etas[0])
    self.model.eval()
    loss = 0
    batch_num = 0
    base_var = 0
    
    for x, y in batchify(test_dataset, batch_size):
      f_beta, _ = self.model(x)

      loss += self.criterion(f_beta + self.bias_adjustment, y) 
      base_var += torch.var(f_beta - y, unbiased=False)
      batch_num += 1

    print(f"Bse Test loss {loss.item()/batch_num} " + f"Bse Var {base_var.item()/batch_num}")
    return loss.item()/batch_num
