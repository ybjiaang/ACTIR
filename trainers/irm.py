import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm
from torch.autograd import grad

from misc import batchify, env_batchify, mean_confidence_interval

class IRM():
  def __init__(self, model, loss_fn, config, reg_lambda=0.1):
    self.model = copy.deepcopy(model)
    self.config = config
    self.classification = self.config.classification

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.optimizer = torch.optim.Adam(self.model.Phi.parameters(), lr=config.lr)

    self.reg_lambda = self.config.irm_reg_lambda 


  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(self.config.n_outer_loop)):
      for train in env_batchify(train_dataset, batch_size, self.config):
        loss = 0
        penalty = 0
        for env_ind in range(n_train_envs):
          x, y = train[env_ind]
          f_beta, _ = self.model(x)
          scale = torch.tensor(1.).to(self.config.device).requires_grad_()
          error = self.criterion(f_beta * scale, y)
          # penalty += grad(error, self.model.beta, create_graph=True)[0].pow(2).mean()
          penalty += grad(error, scale, create_graph=True)[0].pow(2).mean()
          loss += error

        self.optimizer.zero_grad()
        (self.reg_lambda * penalty + loss).backward()
        self.optimizer.step()

      if t % 10 == 0 and self.config.verbose:
        print(loss.item()/(n_train_envs*batch_size))

  
  def test(self, test_dataset, batch_size = 32):
    self.model.eval()
    loss = 0
    total = 0
    
    all_prediction = []
    for x, y in batchify(test_dataset, batch_size, self.config):
      f_beta, _ = self.model(x)
      if self.classification:
        _, predicted = torch.max(f_beta.data, 1)
        correct_or_not = predicted == y
        loss += (correct_or_not).sum()
        all_prediction += (correct_or_not).cpu().numpy().tolist()
      else:
        loss += self.criterion(f_beta, y) * y.size(0) 

      total += y.size(0)
    print(f"Bse Test Error {loss.item()/total} ")
    print(f"Bse Test Std {np.std(np.array(all_prediction).astype(int))} ")
    print(mean_confidence_interval(np.array(all_prediction).astype(int)))
    return loss.item()/total
