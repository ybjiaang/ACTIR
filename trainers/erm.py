import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm

from misc import batchify, env_batchify

class ERM():
  def __init__(self, model, loss_fn, config):
    self.model = copy.deepcopy(model)
    self.config = config
    self.classification = self.config.classification

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)


  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(self.config.n_outer_loop)):
      loss_print = 0
      count = 0
      for train in env_batchify(train_dataset, batch_size, self.config):
        loss = 0
        for env_ind in range(n_train_envs):
          x, y = train[env_ind]
          f_beta, _ = self.model(x)
          loss += self.criterion(f_beta, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_print += loss
        count += 1
    
      if t % 1 == 0 and self.config.verbose:
        print(loss.item()/(n_train_envs*batch_size))
        print(loss_print.item()/count)

  def test(self, test_dataset, batch_size = 32):
    
    self.model.eval()
    loss = 0
    total = 0
    
    for x, y in batchify(test_dataset, batch_size, self.config):
      f_beta, _ = self.model(x)
      if self.classification:
        _, predicted = torch.max(f_beta.data, 1)
        loss += (predicted == y).sum()
      else:
        loss += self.criterion(f_beta, y) * y.size(0) 

      total += y.size(0)
    print(f"Bse Test Error {loss.item()/total} ")
    return loss.item()/total
