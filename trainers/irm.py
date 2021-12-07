import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm
from torch.autograd import grad

from misc import batchify

class IRM():
  def __init__(self, model, loss_fn, config, reg_lambda=0.1):
    self.model = copy.deepcopy(model)
    self.config = config

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.optimizer = torch.optim.Adam(self.model.Phi.parameters(), lr=1e-2)

    self.reg_lambda = reg_lambda


  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(n_outer_loop)):

      loss = 0
      penalty = 0
      for env_ind in range(n_train_envs):
        for x, y in batchify(train_dataset[env_ind], batch_size):
          f_beta, _ = self.model(x)
          error = self.criterion(f_beta, y)
          penalty += grad(error, self.model.beta, create_graph=True)[0].pow(2).mean()
          loss += error

      self.optimizer.zero_grad()
      (self.reg_lambda * penalty + loss).backward()
      self.optimizer.step()

      if t % 10 == 0 and self.config.verbose:
        print(loss.item()/(n_train_envs*batch_size))

  
  def test(self, test_dataset, batch_size = 32):
    # print(self.model.etas[0])
    self.model.eval()
    loss = 0
    batch_num = 0
    base_var = 0
    
    for x, y in batchify(test_dataset, batch_size):
      f_beta, _ = self.model(x)

      loss += self.criterion(f_beta, y) 
      base_var += torch.var(f_beta - y, unbiased=False)
      batch_num += 1

    print(f"Bse Test loss {loss.item()/batch_num} " + f"Bse Var {base_var.item()/batch_num}")
    return loss.item()/batch_num
