import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm
import os

from misc import batchify, maml_batchify, mean_confidence_interval

# only update the linear layer for maml, so not the full maml
class LinearMAML():
  def __init__(self, model, loss_fn, config):
    self.model = copy.deepcopy(model)
    self.config = config
    self.classification = self.config.classification

    # define loss
    self.criterion = loss_fn
    self.fine_inner_lr = config.fine_tune_lr

    # define optimizer
    self.param_to_update_inner_loop = [self.model.beta]
    self.fast_update_lr = config.lr
    self.n_inner_update = 10
    self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    # model save and load path
    self.model_path = config.model_save_dir + "/maml.tar"
    self.emb_path = config.model_save_dir + "/maml_emb"

    if self.config.save_test_phi:
      if not os.path.exists(self.emb_path):
        os.makedirs(self.emb_path)

  def meta_update(self, train_batch, train_query_batch):
    n_train_envs = len(train_batch)
    loss_query = 0
    for env_ind in range(n_train_envs):
      x, y = train_batch[env_ind]

      f_beta, _ = self.model(x)
      loss = self.criterion(f_beta, y)
      grad = torch.autograd.grad(loss, self.param_to_update_inner_loop)

      fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, self.param_to_update_inner_loop)))

      for k in range(1, self.n_inner_update):
        f_beta, _ = self.model(x, fast_beta = fast_weights)
        loss = self.criterion(f_beta, y)
        grad = torch.autograd.grad(loss, fast_weights)
        fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, fast_weights)))

      x_query, y_query = train_query_batch[env_ind]

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
    for t in tqdm(range(self.config.n_outer_loop)):

      for train_spt_set, train_query_set in maml_batchify(train_dataset, batch_size, self.config):
        loss = self.meta_update(train_spt_set, train_query_set)
        
      if t % 10 == 0 and self.config.verbose:
        print(loss.item())
    
    return loss.item()

  def save_model(self):
    torch.save({
            'epoch': self.config.n_outer_loop,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            }, self.model_path)
  
  def test(self, test_dataset, rep_learning_flag = False, batch_size = 1024, input_model = None, print_flag=True):
    
    fast_weights = None
    test_model = self.model
    if input_model is not None:
      test_model, fast_weights = input_model

    test_model.eval()
    loss = 0
    total = 0

    all_prediction = []
    save_tensor_idx = 0
    for x, y in batchify(test_dataset, batch_size, self.config):
      f_beta, phi = test_model(x, fast_beta = fast_weights, rep_learning = rep_learning_flag)

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


  def finetune_test(self, test_finetune_dataset, rep_learning_flag = False, batch_size = 100):
    model = copy.deepcopy(self.model)
    if len(test_finetune_dataset) == 0:
      return (model, [model.beta])
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

    fast_weights = [model.beta]

    return (model, fast_weights)    

