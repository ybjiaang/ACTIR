import numpy as np
import torch 
from torch import nn
import math
from torch.autograd import Variable

# generating data
class Envs(object):
  def __init__(self):
    pass
  
  def sample_dataset(self):
    pass

class CausalAdditiveNoSpurious(Envs):
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1, d_u = 1, d_x_y = 1, guassian_normalized_weight = False):
    super(CausalAdditiveNoSpurious, self).__init__()
    # dimensions
    self.d_x_z_perp = d_x_z_perp
    self.d_x_y_perp = d_x_y_perp
    self.d_u = d_u
    self.d_x_y = d_x_y

    # weight vectors
    if guassian_normalized_weight:
      self.w_x_z_perp = np.random.randn(d_x_z_perp, 1)
      self.w_x_z_perp = self.w_x_z_perp/np.linalg.norm(self.w_x_z_perp, axis=0)
    else:
      self.w_x_z_perp = np.random.uniform(low = -5, high = 5, size=(d_x_z_perp, 1)) 
    
    #input_dim 
    self.input_dim = self.d_x_z_perp + self.d_x_y_perp + self.d_x_y

    self.env_means = [0.2, 2, 1.0, 10.0]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2

    # generate all the weight vector of u for different enviroment
    self.w_u_all = np.zeros((self.num_total_envs, self.d_u, 1))
    for i in range(self.num_total_envs):
      if guassian_normalized_weight:
        self.w_u_all[i,:,:] = np.random.randn(self.d_u, 1)
        self.w_u_all[i,:,:] = self.w_u_all[i,:,:]/np.linalg.norm(self.w_u_all[i,:,:], axis=0)
      else:
        self.w_u_all[i,:,:] = np.random.uniform(low = -5, high = 5, size=(self.d_u, 1))
  
  def sample_envs(self, env_ind, n = 100):
    """ 
    sample data from our enviroment sets
    """
    x_z_perp = np.random.randn(n, self.d_x_z_perp)
    u = np.random.randn(n, self.d_u) + self.env_means[env_ind]
    if env_ind == self.num_total_envs + 1:
      u = np.random.exponential(size=(n, self.d_u)) + self.env_means[env_ind]

    x_y_perp = self.phi_x_y_perp(u)

    # weight vector for u
    w_u = self.w_u_all[env_ind,:,:]

    y = self.fn_y(x_z_perp, self.w_x_z_perp, u, w_u)

    if self.d_x_y != 0:
      x_y = y * self.env_means[env_ind] + np.random.randn(n, 1) * 0.1
      return torch.Tensor(np.concatenate([x_z_perp, x_y_perp, x_y], axis=1)), torch.Tensor(y)

    else:
      return torch.Tensor(np.concatenate([x_z_perp, x_y_perp], axis=1)), torch.Tensor(y)

  def phi_base(self, x):
    return np.sin(np.pi * x)
  
  def phi_u(self, x):
    return np.cos(np.pi * x)

  def phi_x_y_perp(self, x):
    n = x.shape[0]
    return 1 / (1 + np.exp(-x)) + np.random.randn(n, 1)*0.1
  
  def fn_y(self, x_z_perp, w_x_z_perp, u, w_u):
    n = x_z_perp.shape[0]
    return self.phi_base(x_z_perp) @ w_x_z_perp + self.phi_u(u) @ w_u + np.random.randn(n, 1)*0.1

  def sample_base_classifer(self, x):
    # x.shape = [n, self.d_x_z_per]
    return self.phi_base(x[:,:self.d_x_z_perp]) @ self.w_x_z_perp


class CausalHiddenAdditiveNoSpurious(CausalAdditiveNoSpurious):
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1, d_u = 1):
    super(CausalHiddenAdditiveNoSpurious, self).__init__(d_x_z_perp, d_x_y_perp, d_u)
    # change input dim
    self.input_dim = self.d_x_z_perp

  def sample_envs(self, env_ind, n = 100):
    """ 
    sample data from our enviroment sets
    """
    x_z_perp = np.random.randn(n, self.d_x_z_perp)
    u = np.random.randn(n, self.d_u) + self.env_means[env_ind]

    x_y_perp = self.phi_x_y_perp(u) 

    # weight vector for u
    w_u = self.w_u_all[env_ind,:,:]

    x = x_z_perp + x_y_perp + np.random.randn(n, 1)*0.1

    y = self.fn_y(x, self.w_x_z_perp, u, w_u)

    return torch.Tensor(x), torch.Tensor(y)

  def sample_base_classifer(self, x):
    raise Exception("This does not work")


class AntiCausal(CausalAdditiveNoSpurious):
  def __init__(self, d_y = 1, d_u_perp = 1, d_x_y_u = 1):
    super(AntiCausal, self).__init__()
    self.d_y = d_y
    self.d_u_perp = d_u_perp
    self.d_x_y_u = d_x_y_u

    self.input_dim = self.d_u_perp + self.d_x_y_u

  def sample_envs(self, env_ind, n = 100):
    y = np.random.randn(n, self.d_y)

    x_u_perp = self.phi_base(y) + np.random.randn(n, 1)*0.1

    x_y_u = self.phi_x_y_perp(y) * self.env_means[env_ind] + x_u_perp + np.random.randn(n, 1)*0.1
    # if env_ind == self.num_total_envs + 1:
    #    x_y_u = self.phi_u(y) + x_u_perp + np.random.randn(n, 1)*0.1

    return torch.Tensor(np.concatenate([x_u_perp, x_y_u], axis=1)), torch.Tensor(y)

  def sample_base_classifer(self, x):
    raise Exception("This does not work")