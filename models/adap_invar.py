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
    self.phi_odim = 8

    # Define \Phi
    self.Phi = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, self.phi_odim)
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

    # during test, use the first eta
    self.eta_test_ind = 0
    
  def pairwise_distances(self, x):
      #x should be two dimensional
      instances_norm = torch.sum(x**2,-1).reshape((-1,1))
      return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

  def GaussianKernelMatrix(self, x, sigma=1):
      pairwise_distances_ = self.pairwise_distances(x)
      return torch.exp(-pairwise_distances_ /sigma)

  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(n_outer_loop)):

      # update indivual etas
      self.model.freeze_all_but_etas()
      for env_ind in range(n_train_envs):
        for _ in range(n_inner_loop):
          loss = 0
          reg_loss = 0
          batch_num = 0
          for x, y in batchify(train_dataset[env_ind], batch_size):
            batch_num += 1
            f_beta, f_eta, _ = self.model(x, env_ind)
            m = x.shape[0]
            K = self.GaussianKernelMatrix(x)
            L = self.GaussianKernelMatrix(y)
            H = torch.eye(m) - 1.0/m * torch.ones((m,m))
            HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
            # loss += self.criterion(f_beta + f_eta, y) + self.reg_lambda * HSIC
            loss += self.criterion(f_beta + f_eta, y) + self.reg_lambda * torch.pow(torch.mean(f_beta * f_eta), 2) # + 0.1 * torch.mean(f_eta * f_eta)
            # reg_loss += torch.mean(torch.pow(f_beta * f_eta, 2))

          self.inner_optimizer.zero_grad()
          loss.backward()
          self.inner_optimizer.step()
        # print(reg_loss.item()/batch_num)
        # print(loss.item()/batch_num)

      # update phi
      self.model.freeze_all_but_phi()
      phi_loss = 0
      for env_ind in range(n_train_envs):
        for x, y in batchify(train_dataset[env_ind], batch_size):
          f_beta, f_eta, _ = self.model(x, env_ind)
          phi_loss += self.criterion(f_beta + f_eta, y)

      self.outer_optimizer.zero_grad()
      phi_loss.backward()
      self.outer_optimizer.step()

      if t % 10 == 0:
        print(phi_loss.item()/(n_train_envs*batch_size))

  
  def test(self, test_dataset, batch_size = 32):
    # print(self.model.etas[0])
    self.model.eval()
    loss = 0
    batch_num = 0
    base_loss = 0
    for x, y in batchify(test_dataset, batch_size):
      f_beta, f_eta, _ = self.model(x, self.eta_test_ind)

      loss += self.criterion(f_beta + f_eta, y) 
      base_loss += self.criterion(f_beta, y) 
      batch_num += 1

    print(f"Test loss {loss.item()/batch_num}")
    print(f"Bse Test loss {base_loss.item()/batch_num}")
    return loss.item()/batch_num


  def finetune_test(self, test_finetune_dataset, test_unlabeld_dataset = None, batch_size = 32,  n_loop = 100, projected_gd = True):
    self.model.freeze_all() # use this so that I can set etas to zeros when I call test again
    self.model.set_etas_to_zeros()

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


