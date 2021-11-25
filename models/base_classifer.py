import numpy as np
import torch 
from torch import nn
import copy

class BaseClass(nn.Module):
  def __init__(self, input_dim, Phi):
    super(BaseClass, self).__init__()

    self.input_dim = input_dim
    
    # Define \Phi
    self.Phi = copy.deepcopy(Phi)

    self.phi_odim = self.Phi[-1].out_features
    
    # Define \beta
    init_beta_numpy = np.zeros((self.phi_odim, 1))
    init_beta_numpy[0,0] = 1.0
    init_beta = torch.Tensor(init_beta_numpy)
    self.beta = torch.nn.Parameter(init_beta)

  def forward(self, x, fast_beta = None):
    rep = self.Phi(x)

    if fast_beta is None:
      f_beta = rep @ self.beta
    else:
      f_beta = rep @ fast_beta[0]

    return f_beta, rep