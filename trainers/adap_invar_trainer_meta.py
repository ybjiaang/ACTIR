import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm

from misc import batchify, HSICLoss, ConditionalHSICLoss, maml_batchify, DiscreteConditionalHSICLoss, DiscreteConditionalExpecationTest

class AdaptiveInvariantNNTrainerMeta():
  def __init__(self, model, loss_fn, reg_lambda, config, causal_dir = True):
    self.model = copy.deepcopy(model)
    self.config = config
    self.causal_dir = causal_dir

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.model.freeze_all_but_etas()
    self.fast_update_lr = 1e-2
    self.test_inner_optimizer = torch.optim.SGD(self.model.etas.parameters(), lr=1e-3)

    self.model.freeze_all_but_phi()
    self.outer_optimizer = torch.optim.Adam(self.model.Phi.parameters(),lr=1e-2)
    # self.model.freeze_all_but_beta()
    # self.outer_optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-2)

    self.reg_lambda = reg_lambda

    # during test, use the first eta
    self.eta_test_ind = 0

  # Define training Loop
  def meta_update(self, train_batch, train_query_batch, n_inner_loop = 20):
    n_train_envs = len(train_batch)

    self.model.freeze_all_but_beta()
    loss_query = 0
    for env_ind in range(n_train_envs):
      parameter_to_update = [self.model.etas[env_ind]]
      x, y = train_batch[env_ind]
      x_query, y_query = train_query_batch[env_ind]
      loss = self.inner_loss(x, y, env_ind, parameter_to_update)
      grad = torch.autograd.grad(loss, parameter_to_update)

      fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, parameter_to_update)))

      for k in range(1, n_inner_loop):
        loss = self.inner_loss(x, y, env_ind, fast_weights)
        grad = torch.autograd.grad(loss, fast_weights)
        fast_weights = list(map(lambda p: p[1] - self.fast_update_lr * p[0], zip(grad, fast_weights)))

      f_beta, f_eta, _ = self.model(x_query, env_ind, fast_eta = fast_weights)
      loss_query += self.criterion(f_beta + f_eta, y_query) + self.criterion(f_beta, y_query)

    loss_query = loss_query / n_train_envs

    self.outer_optimizer.zero_grad()
    loss_query.backward()
    self.outer_optimizer.step()

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

      if self.causal_dir:
        with torch.no_grad():
          self.test_eta = torch.Tensor(self.model.etas[0].numpy()).clone().detach()

  def inner_loss(self, x, y, env_ind, fast_weight=None):
    f_beta, f_eta, _ = self.model(x, env_ind, fast_eta=fast_weight)
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
        reg_loss = HSICLoss(f_beta, f_eta) #+ 0.1 * torch.mean(f_eta * f_eta)
        # reg_loss = torch.pow(torch.mean(f_beta * f_eta), 2) +  torch.mean(f_eta * f_eta)
        # reg_loss = SampleCovariance(f_beta, f_eta)[0][0]
      else:
        # reg_loss = ConditionalHSICLoss(f_beta, f_eta, y)
        # reg_loss = DiscreteConditionalHSICLoss(f_beta, f_eta, y)
        reg_loss = DiscreteConditionalExpecationTest(f_beta, f_eta, y)
      
    loss = self.reg_lambda * self.criterion(f_beta + f_eta, y) + reg_loss
    
    return loss

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
    
    if self.causal_dir:
      return base_loss.item()/batch_num, loss.item()/batch_num
    else:
      return base_loss.item()/batch_num, loss.item()/batch_num

  def finetune_test(self, test_finetune_dataset, test_unlabeld_dataset = None, batch_size = 32,  n_loop = 20, projected_gd = False):
    if not self.causal_dir:
      self.model.freeze_all() # use this so that I can set etas to zeros when I call test again
      self.model.set_etas_to_zeros()
    else:
      self.model.freeze_all()
      self.model.etas[0].copy_(self.test_eta)
      
    if self.causal_dir:
      M = torch.zeros((self.model.phi_odim, self.model.phi_odim), requires_grad=False)

      # Estimate covariance matrix
      if test_unlabeld_dataset == None:
        for i in range(test_finetune_dataset[0].shape[0]):
          x = test_finetune_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
        
        M /= test_finetune_dataset[0].shape[0]

      else:
        for i in range(test_unlabeld_dataset[0].shape[0]):
          x = test_unlabeld_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
        
        M /= test_unlabeld_dataset[0].shape[0]

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
