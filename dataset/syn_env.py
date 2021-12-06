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

# [[2.573583]] [[[ 2.17644781]]

#  [[-2.05382188]]

#  [[ 0.80540998]]

#  [[-0.21336871]]]


# [[2.72848549]] [[[ 0.57159361]]

#  [[ 0.15256034]]

#  [[ 0.28758005]]

#  [[-1.81879821]]]

# [[2.93074311]] [[[-0.08186377]]

#  [[-0.73037747]]

#  [[ 0.69007207]]

#  [[-0.02080297]]]

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
      self.w_x_z_perp = np.random.uniform(low = 4, high = 5, size=(d_x_z_perp, 1)) 
    
    #input_dim 
    self.input_dim = self.d_x_z_perp + self.d_x_y_perp + self.d_x_y

    self.env_means = [0.1, 2, 5, 10]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2

    # generate all the weight vector of u for different enviroment
    self.w_u_all = np.zeros((self.num_total_envs, self.d_u, 1))
    for i in range(self.num_total_envs):
      if guassian_normalized_weight:
        self.w_u_all[i,:,:] = np.random.randn(self.d_u, 1)
        self.w_u_all[i,:,:] = self.w_u_all[i,:,:]/np.linalg.norm(self.w_u_all[i,:,:], axis=0)
      else:
        self.w_u_all[i,:,:] = np.random.uniform(low = -1, high = 1, size=(self.d_u, 1))
    print(self.w_x_z_perp, self.w_u_all)
  
  def sample_envs(self, env_ind, n = 100):
    """ 
    sample data from our enviroment sets
    """
    # x_z_perp = np.random.uniform(low = -1, high = 1, size=(n, self.d_x_z_perp))
    # x_y_perp = np.random.uniform(low = -1, high = 1, size=(n, self.d_u))  
    x_z_perp =  np.random.randn(n, self.d_x_z_perp)
    u =  np.random.randn(n, self.d_u) * self.env_means[env_ind]
    x_y_perp = self.phi_u(u)

    # weight vector for u
    w_u = self.w_u_all[env_ind,:,:]

    if env_ind == self.num_total_envs - 1:
      y = self.fn_y_base(x_z_perp, self.w_x_z_perp)
    else:
      y = self.fn_y(x_z_perp, self.w_x_z_perp, x_y_perp, w_u) + x_y_perp

    if self.d_x_y != 0:
      x_y = self.phi_base(y) + u #np.random.randn(n, self.d_x_y) * self.env_means[env_ind]
      return torch.Tensor(np.concatenate([x_z_perp, x_y_perp, x_y], axis=1)), torch.Tensor(y)
    else:
      return torch.Tensor(np.concatenate([x_z_perp, x_y_perp], axis=1)), torch.Tensor(y)

  def phi_base(self, x):
    return np.cos(np.pi * x)
  
  def phi_u(self, x):
    return np.sin(np.pi * x)

  def phi_x_y_perp(self, x):
    n = x.shape[0]
    return 1 / (1 + np.exp(-x))
  
  def fn_y(self, x_z_perp, w_x_z_perp, u, w_u):
    n = x_z_perp.shape[0]
    return self.phi_base(x_z_perp) @ w_x_z_perp + u @ w_u + np.random.randn(n, 1)*0.1
  
  def fn_y_base(self, x_z_perp, w_x_z_perp):
    return self.phi_base(x_z_perp) @ w_x_z_perp

  def sample_base_classifer(self, x):
    # x.shape = [n, self.d_x_z_per]
    return self.phi_base(x[:,:self.d_x_z_perp]) @ self.w_x_z_perp

class AntiCausal(CausalAdditiveNoSpurious):
  def __init__(self, d_y = 1, d_u_perp = 1, d_x_y_u = 1):
    super(AntiCausal, self).__init__()
    self.d_y = d_y
    self.d_u_perp = d_u_perp
    self.d_x_y_u = d_x_y_u

    self.input_dim = self.d_u_perp + self.d_x_y_u

    self.env_means = [0.1, 5, 2.5, 1]
    self.env_means[-1] = np.random.uniform(low=0.1, high=5)

  def sample_envs(self, env_ind, n = 100):
    y = np.random.uniform(low = -3, high = 3, size=(n, self.d_y))

    x_u_perp = y * 0.5 + np.random.randn(n, self.d_u_perp)*0.1

    x_y_u = y * self.env_means[env_ind] + np.random.randn(n, self.d_x_y_u)*0.1

    return torch.Tensor(np.concatenate([x_u_perp, x_y_u], axis=1)), torch.Tensor(y)

  def sample_base_classifer(self, x):
    raise Exception("This does not work")