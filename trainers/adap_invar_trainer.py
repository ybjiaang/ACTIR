import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm

from misc import batchify, HSICLoss, ConditionalHSICLoss, env_batchify, DiscreteConditionalHSICLoss

class AdaptiveInvariantNNTrainer():
  def __init__(self, model, loss_fn, reg_lambda, config, causal_dir = True):
    self.model = copy.deepcopy(model)
    self.config = config
    self.causal_dir = causal_dir
    self.classification = self.config.classification
    self.num_class = config.num_class

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.model.freeze_all_but_etas()
    self.inner_optimizer = torch.optim.SGD(self.model.etas.parameters(), lr=1e-2)
    self.test_inner_optimizer = torch.optim.SGD(self.model.etas.parameters(), lr=1e-3)

    self.model.freeze_all_but_phi()
    self.outer_optimizer = torch.optim.Adam(self.model.Phi.parameters(),lr=1e-2)

    # self.model.freeze_all_but_beta()
    # self.outer_optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-2)

    self.reg_lambda = reg_lambda
    self.gamma = 0.9
    # during test, use the first eta
    self.eta_test_ind = 0

  def inner_loss(self, x, y, env_ind):
    f_beta, f_eta, _ = self.model(x, env_ind)
    if self.classification:
      if self.causal_dir:
        # reg_loss = 0
        # for i in range(self.num_class):
        #   reg_loss += HSICLoss(f_beta[:,[i]], f_eta[:,[i]])
        reg_loss = HSICLoss(f_beta, f_eta)
      else:
        # reg_loss = 0
        # for i in range(self.num_class):
        #   reg_loss += DiscreteConditionalHSICLoss(f_beta[:,[i]], f_eta[:,[i]], y)
        reg_loss = DiscreteConditionalHSICLoss(f_beta, f_eta, y)
        # print(reg_loss)
        # print(DiscreteConditionalHSICLoss(x[:,[0]], x[:,[1]] + x[:,[0]], y))
    else:
      if self.causal_dir:
        reg_loss = HSICLoss(f_beta, f_eta)
        # reg_loss = torch.pow(torch.mean(f_beta * f_eta), 2) # + 0.1 * torch.mean(f_eta * f_eta)
      else:
        reg_loss = ConditionalHSICLoss(f_beta, f_eta, y)
      
    loss = self.reg_lambda * self.criterion(f_beta + f_eta, y) + reg_loss

    return loss
    
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

            loss += self.inner_loss(x, y, env_ind)
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
            phi_loss += self.gamma * self.criterion(f_beta + f_eta, y) + (1 - self.gamma) * self.criterion(f_beta, y)

        self.outer_optimizer.zero_grad()
        phi_loss.backward()
        self.outer_optimizer.step()

      if t % 10 == 0 and self.config.verbose:
        print(phi_loss.item()/(n_train_envs*batch_size))

  def test(self, test_dataset, batch_size = 32, print_flag = True):
    # print(self.model.etas[0])
    self.model.eval()
    loss = 0
    total = 0
    base_loss = 0

    for x, y in batchify(test_dataset, batch_size):
      f_beta, f_eta, _ = self.model(x, self.eta_test_ind)

      if self.classification:
        _, base_predicted = torch.max(f_beta.data, 1)
        base_loss += (base_predicted == y).sum()
        _, predicted = torch.max((f_beta + f_eta).data, 1)
        loss += (predicted == y).sum()
      else:
        loss += self.criterion(f_beta + f_eta, y) * y.size(0)
        base_loss += self.criterion(f_beta, y) * y.size(0)
      total += y.size(0)

    if print_flag: 
        print(f"Bse Test Error {base_loss.item()/total} ")
        print(f"Test loss {loss.item()/total} ")
    
    return base_loss.item()/total, loss.item()/total

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
