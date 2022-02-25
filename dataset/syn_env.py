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

class AntiCausalControlDatasetMultiClass(Envs):
  def __init__(self, d_x_z_perp = 1, d_x_y_perp = 1):
    super(AntiCausalControlDatasetMultiClass, self).__init__()
    self.d_x_z_perp = d_x_z_perp
    self.d_x_y_perp = d_x_y_perp
    self.env_means = [0.9, 0.8, 0.2, 0.1]
    self.num_total_envs = len(self.env_means)
    self.num_train_evns = self.num_total_envs - 2
    self.input_dim = self.d_x_z_perp + self.d_x_y_perp
    self.num_class = 5

  def sample_envs(self, env_ind, n = 100):
    y = np.random.randint(self.num_class, size=(n,1))
    factor = np.random.binomial(1, 0.65, (n,1))
    x_z_perp = y * factor + (1 - factor) * np.random.randint(self.num_class, size=(n,1))
    # x_z_perp = (2*y - 1) * (2* factor - 1)
    # x_z_perp = y * (2* factor - 1)
    factor = np.random.binomial(1, self.env_means[env_ind], (n,1))
    z = y * factor + (1 - factor) * np.random.randint(self.num_class, size=(n,1))
    # z = (2*y - 1) * (2* factor - 1)
    # z = y * (2* factor - 1)
    x_y_perp = z
    
    return torch.Tensor(np.concatenate([x_z_perp, x_y_perp], axis=1)),torch.Tensor(y) #torch.squeeze(torch.Tensor(y).long()) #torch.Tensor(y) #
  
  def sample_base_classifer(self, x):
   raise Exception("This does not work")

  def phi_base(self, x):
    return np.cos(x)
  
  def phi_u(self, x):
    return np.cos(np.pi * x) * x

if __name__ == '__main__':
  env = AntiCausalControlDatasetMultiClass()
  x, y = env.sample_envs(0, n = 100)
  print(x[:,0])
  print(x[:,1])
  print(y)
