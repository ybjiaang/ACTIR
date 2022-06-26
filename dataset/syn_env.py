from re import U
import numpy as np
import torch 
from torch import nn
import math
from torch.autograd import Variable
from sklearn import preprocessing

# importing sys
import sys

sys.path.insert(0, '../')

# from misc import DiscreteConditionalExpecationTest

# generating data
class Envs(object):
  def __init__(self):
    pass
  
  def sample_envs(self):
    pass

class CausalControlDataset(Envs):
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1):
    super(CausalControlDataset, self).__init__()
    self.d_x_z_perp = d_x_z_perp
    self.d_x_y_perp = d_x_y_perp
    self.env_means = [0.95, 0.8, 0.2, 0.1]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2
    self.input_dim = self.d_x_z_perp + self.d_x_y_perp 

  def sample_envs(self, env_ind, n = 100):
    u = np.random.randn(n, 1) 
    factor = np.random.binomial(1, self.env_means[env_ind], (n,1))
    z = u * factor + (- u) * (1-factor) 
    x_z_perp = np.random.randn(n, 1)
    x_y_perp = z
    y = self.phi_base(x_z_perp) + u + np.random.randn(n, 1) * 0.1

    return torch.Tensor(np.concatenate([x_z_perp, x_y_perp], axis=1)), torch.Tensor(y)
  
  def sample_base_classifer(self, x):
   return self.phi_base(x[:,:self.d_x_z_perp])

  def phi_base(self, x):
    return x * x
  
  def phi_u(self, x):
    return np.cos(np.pi * x) * x

class CausalControlDescentDataset(Envs):
  def __init__(self, d_x_z_parent = 1, d_x_y_child = 1):
    super(CausalControlDescentDataset, self).__init__()
    self.d_x_z_parent = d_x_z_parent
    self.d_x_y_child = d_x_y_child
    self.env_means = [0.95, 0.8, 0.2, 0.1]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2
    self.input_dim = self.d_x_z_parent + self.d_x_y_child 
    self.num_class = 2

  def sample_envs(self, env_ind, n = 100):
    def xor(a, b):
      return 1 - np.abs(a-b) # Assumes both inputs are either 0 or 1
      
    x_y_parent = np.random.binomial(1, 0.5, (n,1))

    factor = np.random.binomial(1, 0.75, (n,1))
    y = xor(x_y_parent, factor)

    factor = np.random.binomial(1, self.env_means[env_ind], (n,1))
    x_y_perp = xor(y, factor)

    x_y_perp = xor(x_y_perp, x_y_parent)

    return torch.Tensor(np.concatenate([x_y_parent, x_y_perp], axis=1)), torch.squeeze(torch.Tensor(y).long())
  
  def sample_base_classifer(self, x):
   raise Exception("This does not work")

  def phi_base(self, x):
    return x * x
  
  def phi_u(self, x):
    return np.cos(np.pi * x) * x

class AntiCausalControlDataset(Envs):
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1):
    super(AntiCausalControlDataset, self).__init__()
    self.d_x_z_perp = d_x_z_perp
    self.d_x_y_perp = d_x_y_perp
    self.env_means = [0.95, 0.7, 0.6, 0.1]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2
    self.input_dim = self.d_x_z_perp + self.d_x_y_perp
    self.num_class = 2

  def sample_envs_z(self, env_ind = 2, n = 100):
    y = np.random.binomial(1, 0.5, (n,1))
    factor = np.random.binomial(1, 0.75, (n,1))
    x_z_perp = (2*y - 1) * (2* factor - 1)
    factor = np.random.binomial(1, self.env_means[env_ind], (n,1))
    z = (2*y - 1) * (2* factor - 1)
    x_y_perp = z
    
    return torch.Tensor(np.concatenate([x_z_perp, x_y_perp], axis=1)), torch.Tensor(z) #torch.Tensor(y) 

  def sample_envs(self, env_ind, n = 100):
    y = np.random.binomial(1, 0.5, (n,1))
    factor = np.random.binomial(1, 0.75, (n,1))
    x_z_perp = (2*y - 1) * (2* factor - 1)
    factor = np.random.binomial(1, self.env_means[env_ind], (n,1))
    z = (2*y - 1) * (2* factor - 1)
    x_y_perp = z
    
    return torch.Tensor(np.concatenate([x_z_perp, x_y_perp], axis=1)), torch.squeeze(torch.Tensor(y).long()) #torch.Tensor(y) 
  
  def sample_base_classifer(self, x):
   raise Exception("This does not work")

  def phi_base(self, x):
    return np.cos(x)
  
  def phi_u(self, x):
    return np.cos(np.pi * x) * x

  def z_range(self):
    return [-1, 1]

class AntiCausalControlDatasetMultiClass(Envs):
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1):
    super(AntiCausalControlDatasetMultiClass, self).__init__()
    self.d_x_z_perp = d_x_z_perp
    self.d_x_y_perp = d_x_y_perp
    self.env_means = [0.95, 0.85, 0.8, 0.2, 0.1]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2
    self.input_dim = self.d_x_z_perp + self.d_x_y_perp
    self.num_class = 3

  def sample_envs(self, env_ind, n = 100):
    y = np.random.randint(self.num_class, size=(n,1))
    factor = np.random.binomial(1, 0.75, (n,1))
    x_z_perp = (y) * factor + (1 - factor) * np.random.randint(self.num_class, size=(n,1)) 
    # x_z_perp = (y * (y >=2)) * factor + (1 - factor) * np.random.randint(self.num_class, size=(n,1)) 

    factor = np.random.binomial(1, self.env_means[env_ind], (n,1))
    z = (y) * factor + (1 - factor) * np.random.randint(self.num_class, size=(n,1))
    # z = (y * (y <2)) * factor + (1 - factor) * np.random.randint(self.num_class, size=(n,1)) 

    x_y_perp = z

    normalized_arr = np.concatenate([x_z_perp, x_y_perp], axis=1).astype(np.float)
    normalized_arr -= normalized_arr.mean(axis=0)
    normalized_arr /= normalized_arr.max(axis=0)
    
    return torch.Tensor(normalized_arr), torch.squeeze(torch.Tensor(y).long()) #torch.Tensor(y) 
  
  def sample_base_classifer(self, x):
   raise Exception("This does not work")

  def phi_base(self, x):
    return np.cos(x)
  
  def phi_u(self, x):
    return np.cos(np.pi * x) * x

if __name__ == '__main__':
  env = CausalControlDescentDataset()
  x, y = env.sample_envs(0, n = 1000)
  x, y = env.sample_envs(3, n = 1000)
  # print(torch.pow(torch.sum(DiscreteConditionalExpecationTest(x[:,[0]], x[:,[1]], y)),2))
