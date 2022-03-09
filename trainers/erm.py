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

    self.fine_inner_lr = 1e-2

    # optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)


  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(self.config.n_outer_loop)):
      loss_print = 0
      count = 0
      accuracy_count = 0
      total = 0
      base_accuracy_count = 0
      for train in env_batchify(train_dataset, batch_size, self.config):
        loss = 0
        for env_ind in range(n_train_envs):
          x, y = train[env_ind]
          f_beta, _ = self.model(x)
          loss += self.criterion(f_beta, y)

          if self.classification:
            _, base_predicted = torch.max(f_beta.data, 1)
            base_accuracy_count += (base_predicted == y).sum()
            _, predicted = torch.max((f_beta + f_eta).data, 1)
            accuracy_count += (predicted == y).sum()
            total += y.size(0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_print += loss
        count += 1
    
      if t % 1 == 0 and self.config.verbose:
        print(loss.item()/(n_train_envs*batch_size))
        print(loss_print.item()/count)
        if self.classification:
          print(f"Bse Test Error {base_accuracy_count.item()/total} ")
          print(f"Test loss {accuracy_count.item()/total} ")

  def test(self, test_dataset, input_model = None, batch_size = 32, print_flag = True):

    test_model = self.model
    if input_model is not None:
      test_model = input_model

    test_model.eval()
    loss = 0
    total = 0

    for x, y in batchify(test_dataset, batch_size, self.config):
      f_beta, _ = test_model(x)
      if self.classification:
        _, predicted = torch.max(f_beta.data, 1)
        loss += (predicted == y).sum()
      else:
        loss += self.criterion(f_beta, y) * y.size(0) 

      total += y.size(0)
    if print_flag:
      print(f"Bse Test Error {loss.item()/total} ")
    if input_model is None:
        torch.save(self.model, "./erm.model") 
    return loss.item()/total


  def finetune_test(self, test_finetune_dataset, batch_size = 32):
    model = copy.deepcopy(self.model)
    param_to_update_inner_loop  = model.beta

    self.test_inner_optimizer = torch.optim.Adam([param_to_update_inner_loop], lr=self.fine_inner_lr)

    model.train()
    for i in range(self.config.n_finetune_loop):
      batch_num = 0
      for x, y in batchify(test_finetune_dataset, batch_size, self.config):
        loss = 0
        batch_num += 1

        f_beta, _ = model(x)
        loss += self.criterion(f_beta, y) 

        self.test_inner_optimizer.zero_grad()
        loss.backward()
        self.test_inner_optimizer.step()

    return model
