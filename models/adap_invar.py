import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm

from misc import batchify

class AdaptiveInvariantNN(nn.Module):
  def __init__(self, n_batch_envs, input_dim):
    super(AdaptiveInvariantNN, self).__init__()

    self.n_batch_envs = n_batch_envs
    self.input_dim = input_dim
    self.phi_odim = 3

    # Define \Phi
    self.Phi = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, self.phi_odim)
        )

    # Define \beta
    self.beta = torch.nn.Parameter(torch.zeros(self.phi_odim, 1), requires_grad = False) 
    self.beta[0,0] = 1.0

    # Define \eta
    self.etas = nn.ParameterList([torch.nn.Parameter(torch.zeros(self.phi_odim, 1), requires_grad = True) for i in range(n_batch_envs)]) 

  def forward(self, x, env_ind):
    rep = self.Phi(x)

    f_beta = rep @ self.beta
    f_eta = rep @ self.etas[env_ind]

    return f_beta, f_eta, rep

  def sample_base_classifer(self, x):
    x_tensor = torch.Tensor(x)
    return self.Phi(x_tensor) @ self.beta

  """ used to free and check var """
  def freeze_all_but_etas(self):
    for para in self.parameters():
      para.requires_grad = False

    for eta in self.etas:
      eta.requires_grad = True

  def set_etas_to_zeros(self):
    # etas are only temporary and should be set to zero during test
    for eta in self.etas:
      eta.zero_()

  def freeze_all_but_phi(self):
    for para in self.parameters():
      para.requires_grad = True

    for eta in self.etas:
      eta.requires_grad = False
    
    self.beta.requires_grad = False

  def freeze_all(self):
    for para in self.parameters():
      para.requires_grad = False

  def check_var_with_required_grad(self):
    """ Check what paramters are required grad """
    for name, param in self.named_parameters():
      if param.requires_grad:print(name)



class AdaptiveInvariantNNTrainer():
  def __init__(self, model, loss_fn, reg_lambda):
    self.model = copy.deepcopy(model)

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.model.freeze_all_but_etas()
    self.inner_optimizer = torch.optim.SGD(self.model.etas.parameters(), lr=1e-2)
    self.test_inner_optimizer = torch.optim.SGD(self.model.etas.parameters(), lr=1e-2)

    self.model.freeze_all_but_phi()
    self.outer_optimizer = torch.optim.Adam(self.model.Phi.parameters(), lr=1e-2)

    self.reg_lambda = reg_lambda

  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 10):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(n_outer_loop)):

      # update indivual etas
      self.model.freeze_all_but_etas()
      for env_ind in range(n_train_envs):
        for _ in range(n_inner_loop):
          loss = 0
          for x, y in batchify(train_dataset[env_ind], batch_size):
            f_beta, f_eta, _ = self.model(x, env_ind)
            loss += self.criterion(f_beta + f_eta, y) + self.reg_lambda * torch.mean(torch.pow(f_beta * f_eta, 2))

          self.inner_optimizer.zero_grad()
          loss.backward()
          self.inner_optimizer.step()

      # update phi
      self.model.freeze_all_but_phi()
      phi_loss = 0
      # for batch in (n_batches):
      for env_ind in range(n_train_envs):
        for x, y in batchify(train_dataset[env_ind], batch_size):
          f_beta, f_eta, _ = self.model(x, env_ind)
          phi_loss += self.criterion(f_beta + f_eta, y) 

      self.outer_optimizer.zero_grad()
      phi_loss.backward()
      self.outer_optimizer.step()

      if t % 10 == 0:
        print(phi_loss.item()/(n_train_envs*batch_size))


  def test(self, test_dataset, batch_size = 32, test_unlabeld_dataset = None, n_loop = 100):
    self.model.freeze_all() # use this so that I can set etas to zeros when I call test again
    self.model.set_etas_to_zeros()

    M = torch.zeros((self.model.phi_odim, self.model.phi_odim), requires_grad=False)
    # Estimate covariance matrix
    if test_unlabeld_dataset == None:
      for i in range(test_dataset[0].shape[0]):
        x = test_dataset[0][i]
        self.model.eval()
        _, _, rep = self.model(x, 0)
        M += rep.T @ rep
      
      M /= test_dataset[0].shape[0]
    else:
      raise Exception('Dont know how to use test_unlabeld_dataset yet')

    self.model.train()
    self.model.freeze_all_but_etas()

    # test set is ususally small
    for i in range(n_loop):
      loss = 0
      for x, y in batchify(test_dataset, batch_size):
        f_beta, f_eta, _ = self.model(x, 0)

        loss += self.criterion(f_beta + f_eta, y) 

      self.test_inner_optimizer.zero_grad()
      loss.backward()
      self.test_inner_optimizer.step()

      """ projected gradient descent """
      with torch.no_grad():
        v = M @ self.model.beta 
        norm = v.T @ v
        alpha = self.model.etas[0].T @ v
        self.model.etas[0].sub_(alpha * v/norm)

      if i % 10 == 0:
          print(loss.item()/test_dataset[0].shape[0]) 


