import numpy as np
import torch 
from torch import nn
import copy

class BaseClass(nn.Module):
  def __init__(self, input_dim, Phi, out_dim=1):
    super(BaseClass, self).__init__()

    self.input_dim = input_dim
    
    # Define \Phi
    self.Phi = copy.deepcopy(Phi)

    self.phi_odim = self.Phi[-1].out_features
    
    # Define \beta
    init_beta_numpy = np.zeros((self.phi_odim, out_dim))
    for i in range(out_dim):
      init_beta_numpy[i,i] = 1.0
    init_beta = torch.Tensor(init_beta_numpy)
    self.beta = torch.nn.Parameter(init_beta)

  def forward(self, x, fast_beta = None):
    rep = self.Phi(x)

    if fast_beta is None:
      f_beta = rep @ self.beta
    else:
      f_beta = rep @ fast_beta[0]

    return f_beta, rep

  def sample_base_classifer(self, x):
    x_tensor = torch.Tensor(x)
    return self.Phi(x_tensor) @ self.beta