import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm

from misc import batchify, maml_batchify

# only update the linear layer for maml, so not the full maml
class LinearMAML():
  def __init__(self, model, loss_fn, config):
    self.model = copy.deepcopy(model)
    self.config = config

    # define loss
    self.criterion = loss_fn

    # define optimizer
    self.param_to_update_inner_loop = self.model.beta
    self.fast_update_lr = 1e-2
    self.n_inner_update = 10
    self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

  def meta_update(self, train_batch, train_query_batch):
    n_train_envs = len(train_batch)
    
    loss_query = 0
    for env_ind in range(n_train_envs):
      x, y = train_batch[env_ind]
      x_query, y_query = train_query_batch[env_ind]
      f_beta, _ = self.model(x)
      loss = self.criterion(f_beta, y)
      grad = torch.autograd.grad(loss, self.param_to_update_inner_loop)

      fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, self.param_to_update_inner_loop)))

      for k in range(1, self.n_inner_update):
        f_beta, _ = self.model(x, fast_beta = fast_weights)
        loss = self.criterion(f_beta, y)
        grad = torch.autograd.grad(loss, fast_weights)
        fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, fast_weights)))

      f_beta_query, _ = self.model(x_query, fast_beta = fast_weights)
      loss_query += self.criterion(f_beta_query, y_query)

    loss_query = loss_query / n_train_envs

    self.meta_optimizer.zero_grad()
    loss_query.backward()
    self.meta_optimizer.step()

    return loss_query 

  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100):

    self.model.train()
    for t in tqdm(range(n_outer_loop)):
      train_spt_set = []
      train_query_set = []

      for train_spt_set, train_query_set in maml_batchify(train_dataset, batch_size):
        loss = self.meta_update(train_spt_set, train_query_set)
        
      if t % 10 == 0 and self.config.verbose:
        print(loss.item())

  
  def test(self, test_dataset, batch_size = 32, input_model = None):
    
    fast_weights = None
    test_model = self.model
    if input_model is not None:
      test_model, fast_weights = input_model

    test_model.eval()
    loss = 0
    batch_num = 0
    
    for x, y in batchify(test_dataset, batch_size):
      f_beta, _ = test_model(x, fast_beta = fast_weights)
      
      loss += self.criterion(f_beta, y) 
      batch_num += 1
      
    print(f"Bse Test loss {loss.item()/batch_num}")
    
    return loss.item()/batch_num


  def finetune_test(self, test_finetune_dataset, batch_size = 32):
    model = copy.deepcopy(self.model)
    param_to_update_inner_loop  = model.beta

    loss = 0
    for x, y in batchify(test_finetune_dataset, batch_size):
      f_beta, _ = model(x)
      loss += self.criterion(f_beta, y)

    grad = torch.autograd.grad(loss, param_to_update_inner_loop)
    fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, param_to_update_inner_loop)))

    for k in range(1, self.n_inner_update):
      loss = 0
      for x, y in batchify(test_finetune_dataset, batch_size):
        f_beta, _ = model(x, fast_beta = fast_weights)
        loss += self.criterion(f_beta, y)

      grad = torch.autograd.grad(loss, fast_weights)
      fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, fast_weights)))

    return (model, fast_weights)