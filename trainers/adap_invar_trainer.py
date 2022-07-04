import numpy as np
import torch 
from torch import nn
import copy
from tqdm import tqdm
from torch.autograd import grad
import torch.autograd as autograd
import os

from misc import mean_confidence_interval, batchify, HSICLoss, DiscreteConditionalLinearHSICLoss, ConditionalCovaraince, ConditionalHSICLoss, env_batchify, DiscreteConditionalExpecationTest, DiscreteConditionalHSICLoss, printModelParam, SampleCovariance

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
    
    self.fine_tune_lr = config.fine_tune_lr
    self.test_inner_optimizer = torch.optim.Adam(self.model.etas.parameters(), lr=config.fine_tune_lr)

    self.model.freeze_all_but_beta()
    self.outer_optimizer = torch.optim.Adam(self.model.parameters(),lr=config.lr) 

    self.reg_lambda = reg_lambda
    self.reg_lambda_2 = config.reg_lambda_2
    self.gamma = config.gamma
    # during test, use the first eta
    self.eta_test_ind = 0

    self.softmax_layer = nn.Softmax(dim=-1)

    # model save and load path
    self.model_path = config.model_save_dir + "/adap_invar.tar"
    self.emb_path = config.model_save_dir + "/adap_invar_emb"

    if self.config.save_test_phi:
      if not os.path.exists(self.emb_path):
        os.makedirs(self.emb_path)

  def reg_loss(self, f_beta, f_eta, y, end_ind):
    
    if self.classification:
      if self.causal_dir:
        reg_loss = HSICLoss(f_beta, f_eta)
      else:
        # currently, only this reguarlizer is used !!!
        reg_loss = torch.sum(torch.abs(DiscreteConditionalExpecationTest(f_beta, f_eta, y)))
    else:
      if self.causal_dir:
        reg_loss = SampleCovariance(f_beta, f_eta)[0][0]
      else:
        reg_loss = torch.sum(torch.abs(DiscreteConditionalExpecationTest(f_beta, f_eta, y)))
    
    return reg_loss
  
  def contraint_loss(self, f_beta, f_eta, y, env_ind):

    reg_loss = self.reg_loss(f_beta, f_eta, y, env_ind)
    loss = self.criterion(f_beta + f_eta, y) + self.reg_lambda * reg_loss

    return loss

  def gradient_penalty(self, f_beta, f_eta, y, env_ind):
    grad_1 = autograd.grad(self.contraint_loss(f_beta[0::2], f_eta[0::2], y[0::2], env_ind), [self.model.etas[env_ind]], create_graph=True)[0]
    grad_2 = autograd.grad(self.contraint_loss(f_beta[1::2], f_eta[1::2], y[1::2], env_ind), [self.model.etas[env_ind]], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)

    return result
    
  # Define training Loop
  def train(self, train_dataset, batch_size, n_outer_loop = 100, n_inner_loop = 30):
    n_train_envs = len(train_dataset)

    self.model.freeze_all_but_beta()
    self.model.train()

    for t in tqdm(range(self.config.n_outer_loop)):
      for train in env_batchify(train_dataset, batch_size, self.config):

        phi_loss = 0
        loss = 0
        total = 0
        base_loss = 0
        for env_ind in range(n_train_envs):
            x, y = train[env_ind]
            f_beta, f_eta, _ = self.model(x, env_ind)

            contraint_loss = self.contraint_loss(f_beta, f_eta, y, env_ind)
            phi_loss += self.gamma * self.criterion(f_beta + f_eta, y) + (1 - self.gamma) * self.criterion(f_beta, y) 
            gradient = grad(contraint_loss, self.model.etas[env_ind], create_graph=True)[0].pow(2).mean()
            phi_loss += self.reg_lambda_2 * gradient
                        
            if self.classification:
              _, base_predicted = torch.max(f_beta.data, 1)
              base_loss += (base_predicted == y).sum()
              _, predicted = torch.max((f_beta + f_eta).data, 1)
              loss += (predicted == y).sum()
              total += y.size(0)

        self.outer_optimizer.zero_grad()
        phi_loss.backward()
        self.outer_optimizer.step()
      
      if self.config.verbose:
        print(phi_loss.item()/(n_train_envs*batch_size))
        if self.classification:
          print(f"Bse Test Acc {base_loss.item()/total} ")
          print(f"Test Acc {loss.item()/total} ")

  def get_activation(self, test_dataset):
    self.model.eval()
    ret_list = []
    with torch.no_grad():
      for x, z in batchify(test_dataset, self.config.batch_size, self.config):
        f_beta, f_eta, phi = self.model(x, self.eta_test_ind)
        phi_cpu = phi.detach().cpu().numpy()
        x_cpu = x.detach().cpu().numpy()
        z_cpu = z.detach().cpu().numpy()
        combined = np.concatenate((phi_cpu, z_cpu), axis=1)
        ret_list.append(combined)
      return np.concatenate(ret_list, axis=0)


  def test(self, test_dataset, rep_learning_flag = False, input_model = None, batch_size = 1024, print_flag = True):

    test_model = self.model
    if input_model is not None:
      test_model = input_model

    test_model.eval()
    loss = 0
    total = 0
    base_loss = 0

    save_tensor_idx = 0
    base_all_prediction = []
    all_predicition = []
    for x, y in batchify(test_dataset, batch_size, self.config):
      f_beta, f_eta, phi = test_model(x, self.eta_test_ind, rep_learning = rep_learning_flag)

      if self.config.save_test_phi:
        nb_tensors = len(phi)
        for i in range(nb_tensors):
          torch.save({'phi':phi[i].detach().cpu(), 'y': y[i].detach().cpu()}, f"{self.emb_path}/tensor{save_tensor_idx}.pt")
          save_tensor_idx += 1

      if self.classification:
        _, base_predicted = torch.max(f_beta.data, 1)
        base_correct_or_not = base_predicted == y
        base_loss += (base_correct_or_not).sum()
        base_all_prediction += (base_correct_or_not).cpu().numpy().tolist()

        _, predicted = torch.max((f_beta + f_eta).data, 1)
        correct_or_not = predicted == y
        loss += (correct_or_not).sum()
        all_predicition += (correct_or_not).cpu().numpy().tolist()
      else:
        loss += self.criterion(f_beta + f_eta, y) * y.size(0)
        base_loss += self.criterion(f_beta, y) * y.size(0)

      total += y.size(0)

    if print_flag: 
        print(f"Bse Test Acc {base_loss.item()/total} ")
        print(f"Bse Test Std {np.std(np.array(base_all_prediction).astype(int))} ")
        print(mean_confidence_interval(np.array(base_all_prediction).astype(int)))

        # print(f"Test Acc {loss.item()/total} ")
        # print(f"Test Std {np.std(np.array(all_predicition).astype(int))} ")
        # print(mean_confidence_interval(np.array(all_predicition).astype(int)))
    

    return base_loss.item()/total, loss.item()/total

  def save_model(self):
    torch.save({
            'epoch': self.config.n_outer_loop,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.outer_optimizer.state_dict(),
            }, self.model_path)

  def finetune_test(self, test_finetune_dataset, test_unlabeld_dataset = None, rep_learning_flag = False, batch_size = 100,  n_loop = 20, projected_gd = False):
    model = copy.deepcopy(self.model)

    if len(test_finetune_dataset) == 0:
      return model

    param_to_update_inner_loop  = model.beta

    self.test_inner_optimizer = torch.optim.Adam([param_to_update_inner_loop], lr=self.fine_tune_lr)
    model.beta.requires_grad=True
      
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

    model.train()

    for i in range(self.config.n_finetune_loop):
      batch_num = 0
      for x, y in batchify(test_finetune_dataset, batch_size, self.config):
        loss = 0
        batch_num += 1

        f_beta, f_eta, _ = model(x, self.eta_test_ind, rep_learning = rep_learning_flag)
        loss += self.criterion(f_beta, y) 

        self.test_inner_optimizer.zero_grad()
        loss.backward()
        self.test_inner_optimizer.step()


      if self.causal_dir and not self.classification:
        """ projected gradient descent """
        if projected_gd:
          with torch.no_grad():
            v = M @ self.model.beta 
            norm = v.T @ v
            alpha = self.model.etas[0].T @ v
            self.model.etas[0].sub_(alpha * v/norm)

      if i % 10 == 0 and self.config.verbose:
          print(loss.item()/batch_num) 
    return model
