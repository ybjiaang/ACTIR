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
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1, d_u = 1):
    super(CausalAdditiveNoSpurious, self).__init__()
    # dimensions
    self.d_x_z_perp = d_x_z_perp
    self.d_x_y_perp = d_x_y_perp
    self.d_u = d_u

    # weight vectors
    self.w_x_z_perp = np.random.randn(d_x_z_perp, 1)
    self.w_x_z_perp = self.w_x_z_perp/np.linalg.norm(self.w_x_z_perp, axis=0)
    
    #input_dim 
    self.input_dim = self.d_x_z_perp + self.d_x_y_perp

    self.env_means = [0.5, -1.0, 1.5, -2.0, -2.5, 3.0]
    # self.env_means = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2

    # generate all the weight vector of u for different enviroment
    self.w_u_all = np.zeros((self.num_total_envs, self.d_u, 1))
    for i in range(self.num_total_envs):
      self.w_u_all[i,:,:] = np.random.randn(self.d_u, 1)
      self.w_u_all[i,:,:] = self.w_u_all[i,:,:]/np.linalg.norm(self.w_u_all[i,:,:], axis=0)
  
  def sample_envs(self, env_ind, n = 100):
    """ 
    sample data from our enviroment sets
    """
    x_z_perp = np.random.randn(n, self.d_x_z_perp)
    u = np.random.randn(n, self.d_u) + self.env_means[env_ind]
    # x_z_perp = np.random.uniform(low= - 5, high = 5, size=(n, self.d_x_z_perp))
    # u =  np.random.uniform(low= - 5, high = 5, size=(n, self.d_u)) + self.env_means[env_ind]

    x_y_perp = self.phi_x_y_perp(u)

    # weight vector for u
    w_u = self.w_u_all[env_ind,:,:]

    y = self.fn_y(x_z_perp, self.w_x_z_perp, u, w_u)

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
