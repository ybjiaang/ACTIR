import numpy as np
import torch 
from torch import nn
import copy

class BaseClass(nn.Module):
  def __init__(self, n_batch_envs, input_dim, Phi):
    super(BaseClass, self).__init__()

    self.n_batch_envs = n_batch_envs
    self.input_dim = input_dim
    
    # Define \Phi
    self.Phi = copy.deepcopy(Phi)

    self.phi_odim = self.Phi[-1].out_features
    
    # Define \beta
    self.beta = torch.nn.Parameter(torch.zeros(self.phi_odim, 1))
    self.beta[0,0] = 1.0

  def forward(self, x, env_ind):
    rep = self.Phi(x)

    f_beta = rep @ self.beta

    return f_beta, rep