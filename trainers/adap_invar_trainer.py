import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm
from torch.autograd import grad

from misc import batchify, HSICLoss, ConditionalCovaraince, ConditionalHSICLoss, env_batchify, DiscreteConditionalExpecationTest, DiscreteConditionalHSICLoss, printModelParam, SampleCovariance

class AdaptiveInvariantNNTrainer():
  def __init__(self, model, loss_fn, reg_lambda, config, causal_dir = True):
    self.model = copy.deepcopy(model)
    self.config = config
    self.causal_dir = causal_dir
    self.classification = self.config.classification
    self.inner_gd = True
    self.num_class = config.num_class

    # define loss
    self.criterion = loss_fn

    # optimizer
    self.model.freeze_all_but_etas()
    self.adam_inner_optimizers = []
    for i in range(len(self.model.etas)):
      self.adam_inner_optimizers.append(torch.optim.Adam([self.model.etas[i]],lr=1e-2))
    # self.inner_optimizer = torch.optim.SGD(self.model.etas.parameters(), lr=1e-2)
    self.test_inner_optimizer = torch.optim.Adam(self.model.etas.parameters(), lr=1e-2)

    # self.model.freeze_all_but_phi()
    # self.outer_optimizer = torch.optim.Adam(self.model.Phi.parameters(),lr=1e-2)

    self.model.freeze_all_but_beta()
    self.outer_optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-2)

    self.reg_lambda = reg_lambda
    self.reg_lambda_2 = config.reg_lambda_2
    self.gamma = config.gamma
    # during test, use the first eta
    self.eta_test_ind = 0

    self.softmax = nn.Softmax(dim=1)

  def calculate_eta(self, x, y, b):
    """ 
    x.shape = n, fea
    y.shape = n, 1
    b = fea, 1
    """
    M = SampleCovariance(x, x, mean_centering = False)
    if self.causal_dir:
      M_reg = SampleCovariance(x, x, mean_centering = True)
    else:
      M_reg = ConditionalCovaraince(x,y) 
      # M_reg = SampleCovariance(x, x, mean_centering = True)
    LHS = torch.mean(x * y, dim = 0, keepdim=True).T - M @ b
    # print(LHS)
    RHS = M + self.reg_lambda * (M_reg @ b) @ (M_reg @ b).T + 0.001 * torch.eye(x.shape[1])
    # print(RHS)
    # print(torch.linalg.svd(RHS))

    # eta = torch.linalg.lstsq(RHS, LHS).solution
    eta = torch.inverse(RHS) @ LHS
    # print(eta)
    # exit(0)
    return eta

  def reg_loss(self, f_beta, f_eta, y, end_ind):
    
    if self.classification:
      if self.causal_dir:
        # reg_loss = 0
        # for i in range(self.num_class):
        #   reg_loss += HSICLoss(f_beta[:,[i]], f_eta[:,[i]])
        reg_loss = HSICLoss(f_beta, f_eta)
      else:
        # reg_loss = 0
        # for i in range(self.num_class):
        #   reg_loss += DiscreteConditionalHSICLoss(f_beta[:,[i]], f_eta[:,[i]], y)
        reg_loss = DiscreteConditionalHSICLoss(f_beta, f_eta, y)
        # print(reg_loss)
        # print(DiscreteConditionalHSICLoss(x[:,[0]], x[:,[1]] + x[:,[0]], y))
    else:
      if self.causal_dir:
        reg_loss = HSICLoss(f_beta, f_eta) #+ 0.1 * torch.mean(f_eta * f_eta)
        # reg_loss = torch.pow(torch.mean(f_beta * f_eta), 2) +  torch.mean(f_eta * f_eta)
        # reg_loss = SampleCovariance(f_beta, f_eta)[0][0]
      else:
        # reg_loss = ConditionalHSICLoss(f_beta, f_eta, y)
        # reg_loss = DiscreteConditionalHSICLoss(f_beta, f_eta, y)
        reg_loss = DiscreteConditionalExpecationTest(f_beta, f_eta, y) # pow does not work
        # reg_loss = DiscreteConditionalHSICLoss(f_beta, f_eta, y)
    
    return reg_loss

  def inner_loss(self, x, y, env_ind):
    f_beta, f_eta, _ = self.model(x, env_ind)
      
    loss = self.criterion(f_beta + f_eta, y) + self.reg_lambda * self.reg_loss(f_beta, f_eta, y, env_ind)

    return loss
  
  def contraint_loss(self, f_beta, f_eta, y, env_ind):
    loss = self.criterion(f_beta + f_eta, y) + self.reg_lambda * self.reg_loss(f_beta, f_eta, y, env_ind)

    return loss
    
  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.train()

    for t in tqdm(range(n_outer_loop)):
      for train in env_batchify(train_dataset, batch_size):

        # update phi
        # self.model.freeze_all_but_phi()
        self.model.freeze_all_but_beta()
        phi_loss = 0
        for env_ind in range(n_train_envs):
            x, y = train[env_ind]
            f_beta, f_eta, _ = self.model(x, env_ind)
            contraint_loss = self.contraint_loss(f_beta, f_eta, y, env_ind)
            phi_loss += self.gamma * self.criterion(f_beta + f_eta, y) + (1 - self.gamma) * self.criterion(f_beta, y) 
            phi_loss += self.reg_lambda_2 * grad(contraint_loss, self.model.etas[env_ind], create_graph=True)[0].pow(2).mean()

        self.outer_optimizer.zero_grad()
        phi_loss.backward()
        self.outer_optimizer.step()

        # update eta
        # if self.inner_gd:
        #   self.model.freeze_all_but_etas()
        #   for env_ind in range(n_train_envs):
        #     for k in range(n_inner_loop):
        #       x, y = train[env_ind]

        #       loss = self.inner_loss(x, y, env_ind)
        #       self.adam_inner_optimizers[env_ind].zero_grad()
        #       loss.backward()
        #       self.adam_inner_optimizers[env_ind].step()
        # else:
        #   self.model.freeze_all()
        #   for env_ind in range(n_train_envs):
        #     x, y = train[env_ind]
        #     _, _, phi_x = self.model(x, env_ind)
        #     self.model.etas[env_ind].data = self.calculate_eta(phi_x, y, self.model.beta)
        # print(self.model.etas[0])

      if t % 10 == 0 and self.config.verbose:
        print(phi_loss.item()/(n_train_envs*batch_size))

  def test(self, test_dataset, batch_size = 32, print_flag = True):
    # print(self.model.etas[0])
    self.model.eval()
    loss = 0
    total = 0
    base_loss = 0

    for x, y in batchify(test_dataset, batch_size):
      f_beta, f_eta, _ = self.model(x, self.eta_test_ind)

      if self.classification:
        _, base_predicted = torch.max(f_beta.data, 1)
        base_loss += (base_predicted == y).sum()
        _, predicted = torch.max((f_beta + f_eta).data, 1)
        loss += (predicted == y).sum()
      else:
        loss += self.criterion(f_beta + f_eta, y) * y.size(0)
        base_loss += self.criterion(f_beta, y) * y.size(0)
      total += y.size(0)

    if print_flag: 
        print(f"Bse Test Error {base_loss.item()/total} ")
        print(f"Test loss {loss.item()/total} ")
    
    return base_loss.item()/total, loss.item()/total

  def finetune_test(self, test_finetune_dataset, test_unlabeld_dataset = None, batch_size = 32,  n_loop = 20, projected_gd = False):
    self.model.freeze_all() # use this so that I can set etas to zeros when I call test again
    self.model.set_etas_to_zeros()
      
    if self.causal_dir:
      M = torch.zeros((self.model.phi_odim, self.model.phi_odim), requires_grad=False)
      mean_phi = torch.zeros((self.model.phi_odim), requires_grad=False)
      # Estimate covariance matrix
      total_num_entries = test_finetune_dataset[0].shape[0]
      if test_unlabeld_dataset == None:
        for i in range(test_finetune_dataset[0].shape[0]):
          x = test_finetune_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
          mean_phi += rep
        
        mean_phi /= total_num_entries
        M /= total_num_entries

        M = (M - torch.outer(mean_phi, mean_phi)) * total_num_entries / (total_num_entries - 1)

      else:
        total_num_entries = test_unlabeld_dataset[0].shape[0] + test_finetune_dataset[0].shape[0]
        for i in range(test_unlabeld_dataset[0].shape[0]):
          x = test_unlabeld_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
          mean_phi += rep

        for i in range(test_finetune_dataset[0].shape[0]):
          x = test_finetune_dataset[0][i]
          self.model.eval()
          _, _, rep = self.model(x, self.eta_test_ind)
          M += torch.outer(rep, rep)
          mean_phi += rep
        
        mean_phi /= total_num_entries
        M /= total_num_entries

        M = (M - torch.outer(mean_phi, mean_phi)) * total_num_entries / (total_num_entries - 1)

    self.model.train()
    self.model.freeze_all_but_etas()

    # test set is ususally small
    for i in range(n_loop):
      loss = 0
      batch_num = 0
      for x, y in batchify(test_finetune_dataset, batch_size):
        batch_num += 1

        f_beta, f_eta, _ = self.model(x, self.eta_test_ind)
        loss += self.criterion(f_beta + f_eta, y) 

      self.test_inner_optimizer.zero_grad()
      loss.backward()
      self.test_inner_optimizer.step()
      # print(loss.item())

      if self.causal_dir:
        """ projected gradient descent """
        if projected_gd:
          with torch.no_grad():
            v = M @ self.model.beta 
            norm = v.T @ v
            alpha = self.model.etas[0].T @ v
            self.model.etas[0].sub_(alpha * v/norm)
        # print(self.model.etas[0].T @ M @ self.model.beta )

      # if i % 10 == 0:
      #     print(loss.item()/batch_num) 
