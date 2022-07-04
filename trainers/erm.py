import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm
import os

from misc import batchify, env_batchify, mean_confidence_interval

class ERM():
  def __init__(self, model, loss_fn, config):
    self.model = copy.deepcopy(model)
    self.config = config
    self.classification = self.config.classification

    # define loss
    self.criterion = loss_fn

    self.fine_inner_lr = config.fine_tune_lr

    # optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    # model save and load path
    self.model_path = config.model_save_dir + "/erm.tar"
    self.emb_path = config.model_save_dir + "/erm_emb"

    if self.config.save_test_phi:
      if not os.path.exists(self.emb_path):
        os.makedirs(self.emb_path)

  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(self.config.n_outer_loop)):
      loss_print = 0
      count = 0
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
            total += y.size(0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_print += loss
        count += 1
    
      if t % 1 == 0 and self.config.verbose:
        print(loss_print.item()/count)
        if self.classification:
          print(f"Bse Test Acc {base_accuracy_count.item()/total} ")

  def test(self, test_dataset, rep_learning_flag = False, input_model = None, batch_size = 1024, print_flag = True):

    test_model = self.model
    if input_model is not None:
      test_model = input_model

    test_model.eval()
    loss = 0
    total = 0

    save_tensor_idx = 0
    all_prediction = []
    for x, y in batchify(test_dataset, batch_size, self.config):
      f_beta, phi = test_model(x, rep_learning = rep_learning_flag)

      if self.config.save_test_phi:
        nb_tensors = len(phi)
        for i in range(nb_tensors):
          torch.save({'phi':phi[i].detach().cpu(), 'y': y[i].detach().cpu()}, f"{self.emb_path}/tensor{save_tensor_idx}.pt")
          save_tensor_idx += 1

      if self.classification:
        _, predicted = torch.max(f_beta.data, 1)
        correct_or_not = predicted == y
        loss += (correct_or_not).sum()
        all_prediction += (correct_or_not).cpu().numpy().tolist()
      else:
        loss += self.criterion(f_beta, y) * y.size(0) 

      total += y.size(0)

    if print_flag:
      print(f"Bse Test Acc {loss.item()/total} ")
      print(f"Bse Test Std {np.std(np.array(all_prediction).astype(int))} ")
      print(mean_confidence_interval(np.array(all_prediction).astype(int)))
    return loss.item()/total

  def save_model(self):
    torch.save({
            'epoch': self.config.n_outer_loop,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model_path)

  def finetune_test(self, test_finetune_dataset, rep_learning_flag = False, batch_size = 100):
    model = copy.deepcopy(self.model)
    if len(test_finetune_dataset) == 0:
      return model
    param_to_update_inner_loop  = model.beta
    self.test_inner_optimizer = torch.optim.Adam([param_to_update_inner_loop], lr=self.fine_inner_lr)

    model.train()
    for i in range(self.config.n_finetune_loop):
      batch_num = 0
      for x, y in batchify(test_finetune_dataset, batch_size, self.config):
        loss = 0
        batch_num += 1

        f_beta, _ = model(x, rep_learning = rep_learning_flag)
        loss += self.criterion(f_beta, y) 

        self.test_inner_optimizer.zero_grad()
        loss.backward()
        self.test_inner_optimizer.step()

    return model
