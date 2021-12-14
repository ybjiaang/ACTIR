import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm

from misc import batchify, HSICLoss, ConditionalHSICLoss, env_batchify

class AdaptiveInvariantNNTrainer():
  def __init__(self, model, loss_fn, reg_lambda, config, causal_dir = True):
    self.model = copy.deepcopy(model)
    self.config = config
    self.causal_dir = causal_dir

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.model.freeze_all_but_etas()
    self.inner_optimizer = torch.optim.Adam(self.model.etas.parameters(), lr=1e-2)
    self.test_inner_optimizer = torch.optim.SGD(self.model.etas.parameters(), lr=1e-3)

    self.model.freeze_all_but_phi()
    self.outer_optimizer = torch.optim.Adam(self.model.Phi.parameters(),lr=1e-2)

    # self.model.freeze_all_but_beta()
    # self.outer_optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-2)

    self.reg_lambda = reg_lambda

    # during test, use the first eta
    self.eta_test_ind = 0
    
  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 20):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(n_outer_loop)):
      for train in env_batchify(train_dataset, batch_size):
        self.model.freeze_all_but_etas()
        for _ in range(n_inner_loop):
          for env_ind in range(n_train_envs):
            loss = 0
            x, y = train[env_ind]
            f_beta, f_eta, _ = self.model(x, env_ind)
            if self.causal_dir:
              hsic_loss = HSICLoss(f_beta, f_eta)
              loss += self.criterion(f_beta + f_eta, y) +self.reg_lambda * hsic_loss
              # loss += self.criterion(f_beta + f_eta, y) + self.reg_lambda * torch.pow(torch.mean(f_beta * f_eta), 2) # + 0.1 * torch.mean(f_eta * f_eta)
            else:
              hsic_loss = ConditionalHSICLoss(f_beta, f_eta, y)
              loss += self.criterion(f_beta + f_eta, y) + self.reg_lambda * hsic_loss

            self.inner_optimizer.zero_grad()
            loss.backward()
            # for p in self.model.etas.parameters():
            #   print(p.grad)
            self.inner_optimizer.step()
            # print(self.criterion(f_beta + f_eta, y).item(), hsic_loss.item())

        # update phi
        self.model.freeze_all_but_phi()
        phi_loss = 0
        for env_ind in range(n_train_envs):
            x, y = train[env_ind]
            f_beta, f_eta, _ = self.model(x, env_ind)
            phi_loss += self.criterion(f_beta + f_eta, y) + self.criterion(f_beta, y)

        self.outer_optimizer.zero_grad()
        phi_loss.backward()
        self.outer_optimizer.step()

      if t % 10 == 0 and self.config.verbose:
        print(phi_loss.item()/(n_train_envs*batch_size))

  def test(self, test_dataset, batch_size = 32, print_flag = True):
    # print(self.model.etas[0])
    self.model.eval()
    loss = 0
    batch_num = 0
    base_loss = 0
    var = 0
    base_var = 0
    for x, y in batchify(test_dataset, batch_size):
      f_beta, f_eta, _ = self.model(x, self.eta_test_ind)

      loss += self.criterion(f_beta + f_eta, y) 
      var += torch.var(f_beta + f_eta - y, unbiased=False)
      base_loss += self.criterion(f_beta, y) 
      base_var += torch.var(f_beta - y, unbiased=False)
      batch_num += 1

    if print_flag:
        print(f"Bse Test loss {base_loss.item()/batch_num}, " + f"Bse Var {base_var.item()/batch_num}")
        print(f"Test loss {loss.item()/batch_num} " + f"Test Var {var.item()/batch_num}")
    
    return base_loss.item()/batch_num, loss.item()/batch_num

  def finetune_test(self, test_finetune_dataset, test_unlabeld_dataset = None, batch_size = 32,  n_loop = 20, projected_gd = False):
    self.model.freeze_all() # use this so that I can set etas to zeros when I call test again
    self.model.set_etas_to_zeros()
      
    if self.causal_dir:
      M = torch.zeros((self.model.phi_odim, self.model.phi_odim), requires_grad=False)
      mean_phi = torch.zeros((self.model.phi_odim), requires_grad=False)
      # Estimate covariance matrix
      total_num_entries = test_finetune_dataset[0].shape[0]
      if test_unlabeld_dataset == None:
        for i in range(test_finetune_dataset[0].shape[0]):
          x = test_finetune_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
          mean_phi += rep
        
        mean_phi /= total_num_entries
        M /= total_num_entries

        M = (M - torch.outer(mean_phi, mean_phi)) * total_num_entries / (total_num_entries - 1)

      else:
        total_num_entries = test_unlabeld_dataset[0].shape[0] + test_finetune_dataset[0].shape[0]
        for i in range(test_unlabeld_dataset[0].shape[0]):
          x = test_unlabeld_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
          mean_phi += rep

        for i in range(test_finetune_dataset[0].shape[0]):
          x = test_finetune_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
          mean_phi += rep
        
        mean_phi /= total_num_entries
        M /= total_num_entries

        M = (M - torch.outer(mean_phi, mean_phi)) * total_num_entries / (total_num_entries - 1)

    self.model.train()
    self.model.freeze_all_but_etas()

    # test set is ususally small
    for i in range(n_loop):
      loss = 0
      batch_num = 0
      for x, y in batchify(test_finetune_dataset, batch_size):
        batch_num += 1

        f_beta, f_eta, _ = self.model(x, self.eta_test_ind)
        loss += self.criterion(f_beta + f_eta, y) 

      self.test_inner_optimizer.zero_grad()
      loss.backward()
      self.test_inner_optimizer.step()
      # print(loss.item())

      if self.causal_dir:
        """ projected gradient descent """
        if projected_gd:
          with torch.no_grad():
            v = M @ self.model.beta 
            norm = v.T @ v
            alpha = self.model.etas[0].T @ v
            self.model.etas[0].sub_(alpha * v/norm)
        # print(self.model.etas[0].T @ M @ self.model.beta )

      # if i % 10 == 0:
      #     print(loss.item()/batch_num) 
