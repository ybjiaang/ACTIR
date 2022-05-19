import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm

from misc import batchify

class AdaptiveInvariantNN(nn.Module):
  def __init__(self, n_batch_envs, input_dim, Phi, config, out_dim=1, phi_dim = None):
    super(AdaptiveInvariantNN, self).__init__()

    self.n_batch_envs = n_batch_envs
    self.input_dim = input_dim
    self.classification = config.classification
    
    # Define \Phi
    self.Phi = copy.deepcopy(Phi)

    if not phi_dim:
      self.phi_odim = self.Phi[-1].out_features
    else:
      self.phi_odim = phi_dim
    
    # Define \beta
    self.beta = torch.nn.Parameter(torch.zeros(self.phi_odim, out_dim), requires_grad = False) 
    for i in range(out_dim):
      self.beta[i,i] = 1.0

    # Define \eta
    self.etas = nn.ParameterList([torch.nn.Parameter(torch.zeros(self.phi_odim, out_dim), requires_grad = True) for i in range(n_batch_envs)]) 

    self.softmax_layer = nn.Softmax(dim=-1)

  def forward(self, x, env_ind, rep_learning = False, fast_eta = None):
    if rep_learning:
      rep = x
    else:
      rep = self.Phi(x)

    f_beta = rep @ self.beta
    if fast_eta is None:
      f_eta = rep @ self.etas[env_ind]
    else:
      f_eta = rep @ fast_eta[0]

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

  def freeze_all_but_beta(self):
    for para in self.parameters():
      para.requires_grad = True
    
    self.beta.requires_grad = False

  def freeze_all(self):
    for para in self.parameters():
      para.requires_grad = False

  def free_all(self):
    for para in self.parameters():
      para.requires_grad = True

  def check_var_with_required_grad(self):
    """ Check what paramters are required grad """
    for name, param in self.named_parameters():
      if param.requires_grad:print(name)

