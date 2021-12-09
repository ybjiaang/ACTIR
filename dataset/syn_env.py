import numpy as np
import torch 
from torch import nn
import math
from torch.autograd import Variable

# generating data
class Envs(object):
  def __init__(self):
    pass
  
  def sample_envs(self):
    pass

class CausalAdditiveNoSpurious(Envs):
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1, d_u = 1, d_x_y = 0, guassian_normalized_weight = False):
    super(CausalAdditiveNoSpurious, self).__init__()
    # dimensions
    self.d_x_z_perp = d_x_z_perp
    self.d_x_y_perp = d_x_y_perp
    self.d_u = d_u
    self.d_x_y = d_x_y
    self.sigma = 0.1

    # weight vectors
    if guassian_normalized_weight:
      self.w_x_z_perp = np.random.randn(d_x_z_perp, 1)
      self.w_x_z_perp = self.w_x_z_perp / d_x_z_perp
    else:
      self.w_x_z_perp = np.random.uniform(low = 1, high = 2, size=(d_x_z_perp, 1)) 
    
    #input_dim 
    self.input_dim = self.d_x_z_perp + self.d_x_y + self.d_x_y_perp 

    self.env_means = [0.2, 5.0, 4.0, 2.0]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2

    # generate all the weight vector of u for different enviroment
    self.w_u_all = np.zeros((self.num_total_envs, self.d_u, 1))
    for i in range(self.num_total_envs):
      if guassian_normalized_weight:
        self.w_u_all[i,:,:] = np.random.randn(self.d_u, 1)
        self.w_u_all[i,:,:] = self.w_u_all[i,:,:]/self.d_u
      else:
        self.w_u_all[i,:,:] = np.random.uniform(low = -1.0, high = 1.0, size=(self.d_u, 1))

    self.w_u_y_all = np.zeros((self.num_total_envs, self.d_u, self.d_x_y_perp))
    for i in range(self.num_total_envs):
      if guassian_normalized_weight:
        self.w_u_y_all[i,:,:] = np.random.randn(self.d_u, self.d_x_y_perp)
        self.w_u_y_all[i,:,:] = self.w_u_y_all[i,:,:]/self.d_u
      else:
        self.w_u_y_all[i,:,:] = np.random.uniform(low = -1.0, high = 1.0, size=(self.d_u, self.d_x_y_perp))
  
    if guassian_normalized_weight:
      self.w_x_y_all = np.random.randn(self.d_x_y_perp, 1)
      self.w_x_y_all = self.w_x_y_all/self.d_x_y_perp
    else:
      self.w_x_y_all = np.random.uniform(low = -1.0, high = 1.0, size=(self.d_x_y_perp, 1)) 

  def sample_envs(self, env_ind, n = 100):
    """ 
    sample data from our enviroment sets
    """
    # x_z_perp = np.random.uniform(low = -1, high = 1, size=(n, self.d_x_z_perp))
    # u = np.random.uniform(low = -1, high = 1, size=(n, self.d_u)) + self.env_means[env_ind]
    x_z_perp =  np.random.randn(n, self.d_x_z_perp)
    u =  np.random.randn(n, self.d_u) * np.sqrt(self.env_means[env_ind])

    x_y_perp = self.phi_base(u) @ self.w_u_y_all[env_ind, :, :] + np.random.randn(n, self.d_x_y_perp) * np.sqrt(self.env_means[env_ind])

    # weight vector for u
    w_u = self.w_u_all[env_ind,:,:]

    y = self.fn_y(x_z_perp, self.w_x_z_perp, u, w_u, env_ind) + x_y_perp @ self.w_x_y_all

    return torch.Tensor(np.concatenate([x_z_perp, x_y_perp], axis=1)), torch.Tensor(y)

  def phi_base(self, x):
    return np.sin(np.pi * x)
  
  def phi_u(self, x):
    return np.cos(np.pi * x) * x

  def phi_x_y_perp(self, x):
    n = x.shape[0]
    return 1 / (1 + np.exp(-x)) 
  
  def fn_y(self, x_z_perp, w_x_z_perp, u, w_u, env_ind):
    n = x_z_perp.shape[0]
    return self.fn_y_base(x_z_perp, w_x_z_perp) + self.phi_u(u) @ w_u + np.random.randn(n, 1) * self.sigma
  
  def fn_y_base(self, x_z_perp, w_x_z_perp):
    return self.phi_base(x_z_perp) @ w_x_z_perp 

  def sample_base_classifer(self, x):
    # x.shape = [n, self.d_x_z_per]
    return self.fn_y_base(x[:,:self.d_x_z_perp], self.w_x_z_perp) 

class AntiCausal(CausalAdditiveNoSpurious):
  def __init__(self, d_y = 1, d_u_perp = 1, d_x_y_u = 1):
    super(AntiCausal, self).__init__()
    self.d_y = d_y
    self.d_u_perp = d_u_perp
    self.d_x_y_u = d_x_y_u

    self.input_dim = self.d_u_perp + self.d_x_y_u

    self.env_means = [0.2, 2.0, 4.0, 5.0]
    self.env_means[-1] = np.random.uniform(low=0.1, high=5)

  def sample_envs(self, env_ind, n = 100):
    # y = np.random.uniform(low = -3, high = 3, size=(n, self.d_y))
    y = np.random.randn(n, self.d_y)

    x_u_perp = y * 0.5 + np.random.randn(n, self.d_u_perp)

    x_y_u = y * 0.5 + np.random.randn(n, self.d_x_y_u) * np.sqrt(self.env_means[env_ind])

    return torch.Tensor(np.concatenate([x_u_perp, x_y_u], axis=1)), torch.Tensor(y)

  def sample_base_classifer(self, x):
    raise Exception("This does not work")
