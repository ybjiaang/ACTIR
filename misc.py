from matplotlib.pyplot import axes, axis
import numpy as np
import torch 
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

from torchsampler import ImbalancedDatasetSampler
import scipy.stats

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def mean_confidence_interval(data, confidence=0.90):
  a = data.ravel()
  a = 1.0 * a
  n = len(a)
  m, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
  return m, h, se

def create_DF(inp, x_values):
  df = pd.DataFrame(inp).melt()
  df.columns = ['num of finetuning points', 'finetuned accuary']

  df.loc[:, 'num of finetuning points'].replace(np.arange(len(x_values)), x_values, inplace=True)

  return df

class FolderDataset(Dataset):
  def __init__(self, folder):
    self.files = os.listdir(folder)
    self.folder = folder
    self.all_tensors = []
    for idx in range(len(self.files)):
      tensor_dict = torch.load(f"{self.folder}/{self.files[idx]}")
      self.all_tensors.append(tensor_dict)
  def __len__(self):
    return len(self.files)
  def __getitem__(self, idx):
    # tensor_dict = torch.load(f"{self.folder}/{self.files[idx]}")
    tensor_dict = self.all_tensors[idx]
    return tensor_dict['phi'], tensor_dict['y']

def printModelParam(model):
  for name, param in model.named_parameters():
    print(name, param.data)

def itr_merge(itrs, config):
  num_itrs = len(itrs)
  if num_itrs == 1:
    for v in itrs[0]:
      yield (v[0].to(config.device), v[1].to(config.device))
  else:
    # find longest dataset
    all_lens = []
    for i in range(num_itrs): 
      all_lens.append(len(itrs[i]))
    np_iterations = max(all_lens)

    loops = []
    for i in range(num_itrs):
      loops.append(iter(itrs[i]))

    for _ in range(np_iterations):
      v_list = []
      for i in range(num_itrs): 
        try:
          v = next(loops[i])
          v_list.append((v[0].to(config.device), v[1].to(config.device)))
        except StopIteration:
          loops[i] = iter(itrs[i])
          v = next(loops[i])
          v_list.append((v[0].to(config.device), v[1].to(config.device)))
      yield v_list


def maml_iter_merge(itrs, config):
  num_itrs = len(itrs)
  # find longest dataset
  all_lens = []
  for i in range(num_itrs): 
    all_lens.append(len(itrs[i]))
  np_iterations = max(all_lens)

  loops = []
  for i in range(num_itrs):
    loops.append(iter(itrs[i]))

  for _ in range(np_iterations):
    v_sqt_list = []
    v_qrt_set = []
    for i in range(num_itrs): 
      try:
        v = next(loops[i])
        v_sqt_list.append((v[0][0::2].to(config.device), v[1][0::2].to(config.device)))
        v_qrt_set.append((v[0][1::2].to(config.device), v[1][1::2].to(config.device)))
      except StopIteration:
        loops[i] = iter(itrs[i])
        v = next(loops[i])
        v_sqt_list.append((v[0][0::2].to(config.device), v[1][0::2].to(config.device)))
        v_qrt_set.append((v[0][1::2].to(config.device), v[1][1::2].to(config.device)))
    yield v_sqt_list, v_qrt_set

def batchify(dataset, batch_size, config):
  if config.torch_loader:
    if config.balanced_dataset:
      return itr_merge([torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(dataset), shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker)], config)
    else:
      return itr_merge([torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, worker_init_fn=seed_worker)], config)
  else:
    x, y = dataset
    total_length = len(x)
    nloops = np.ceil(total_length/batch_size).astype(int)

    def creatDataSet():
      for i in range(nloops):
        start = i*batch_size
        if (start + batch_size) >= total_length:
          yield x[start:].to(config.device), y[start:].to(config.device)
        else:
          yield x[start : start + batch_size].to(config.device), y[start : start + batch_size].to(config.device)

    return creatDataSet()

def env_batchify(dataset, batch_size, config):
  n_envs = len(dataset)
  if config.torch_loader:
    dataloaders = []
    for i in range(n_envs):
      if config.balanced_dataset:    
        dataloaders.append(torch.utils.data.DataLoader(dataset=dataset[i],  sampler=ImbalancedDatasetSampler(dataset[i]), drop_last=True, batch_size=batch_size, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker))
      else:
        dataloaders.append(torch.utils.data.DataLoader(dataset=dataset[i],  drop_last=True, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, worker_init_fn=seed_worker))
    return itr_merge(dataloaders, config)
    # return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers)

  else:
    all_lens = np.zeros(n_envs)
    for i, dataset_per_env in enumerate(dataset):
      all_lens[i] = len(dataset_per_env[0])

    total_length_min = np.min(all_lens)
    nloops = np.ceil(total_length_min/batch_size).astype(int)

    def creatDataSet():
      for i in range(nloops):
        start = i*batch_size
        train_sqt_set = []

        for env_ind in range(n_envs):
          x, y = dataset[env_ind]
          if (start + batch_size) >= all_lens[env_ind]:
            train_sqt_set.append((x[start:].to(config.device), y[start:].to(config.device)))
          else:
            train_sqt_set.append((x[start : start + batch_size].to(config.device), y[start : start + batch_size].to(config.device)))

        yield train_sqt_set
        
    return creatDataSet()

def maml_batchify(dataset, batch_size, config):
  n_envs = len(dataset)
  if config.torch_loader:
    dataloaders = []
    for i in range(n_envs):
      if config.balanced_dataset: 
        dataloaders.append(torch.utils.data.DataLoader(dataset=dataset[i],  sampler=ImbalancedDatasetSampler(dataset[i]), drop_last=True, batch_size=batch_size, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker))
      else:
        dataloaders.append(torch.utils.data.DataLoader(dataset=dataset[i],  drop_last=True, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, worker_init_fn=seed_worker))
    return maml_iter_merge(dataloaders, config)

  else:
    all_lens = np.zeros(n_envs)
    for i, dataset_per_env in enumerate(dataset):
      all_lens[i] = len(dataset_per_env[0])

    total_length_min = np.min(all_lens)
    nloops = np.ceil(total_length_min/batch_size).astype(int)

    def creatDataSet():
      for i in range(nloops):
        start = i*batch_size
        train_sqt_set = []
        train_qrt_set = []

        for env_ind in range(n_envs):
          x, y = dataset[env_ind]
          if (start + batch_size) >= all_lens[env_ind]:
            # assume we have at least two elements, otherwise, python would throw the index error
            train_sqt_set.append((x[start::2].to(config.device), y[start::2].to(config.device)))
            train_qrt_set.append((x[start+1::2].to(config.device), y[start+1::2].to(config.device)))

          else:
            train_sqt_set.append((x[start : start + batch_size//2].to(config.device), y[start : start + batch_size//2].to(config.device)))
            train_qrt_set.append((x[start + batch_size//2 : start + batch_size].to(config.device), y[start + batch_size//2 : start + batch_size].to(config.device)))

        yield train_sqt_set, train_qrt_set
      
    return creatDataSet()

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def LinearKernelMatrix(x):
  return torch.mm(x,x.t())

def SampleCovariance(x,y, mean_centering = True):
  n, _ = x.shape
  if mean_centering:
    mean_x = torch.mean(x, dim = 0)
    mean_y = torch.mean(y, dim = 0)
    estimate = (x - mean_x).T @ (y - mean_y) / (n - 1)
  else:
    estimate = x.T @ y / n
    
  return estimate

def ConditionalCovaraince(x, y):
  n, _ = x.shape
  if len(y.shape) > 1:
    temp_y = y[:,0]
  else:
    temp_y = y
  labels_in_batch_sorted, indices = torch.sort(temp_y)
  unique_ixs = 1 + (labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1]).nonzero()
  unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(temp_y)]

  estimate = 0
  for j in range(len(unique_ixs)-1):
    current_class_indices = unique_ixs[j], unique_ixs[j + 1]
    count = current_class_indices[1] - current_class_indices[0]
    if count < 2: 
      continue
    curr_class_slice = slice(*current_class_indices)
    curr_class_indices = indices[curr_class_slice].sort()[0]

    estimate += SampleCovariance(x[curr_class_indices, :], x[curr_class_indices, :]) * count

  return estimate / n

def HSICLoss(x, y, s_x=1, s_y=1, epsilon = 1e-6, cuda=False):
  m,_ = x.shape #batch size
  K = GaussianKernelMatrix(x,s_x)
  L = GaussianKernelMatrix(y,s_y)
  H = torch.eye(m) - 1.0/m * torch.ones((m,m))
  if cuda:
    H = H.double().cuda() 
  HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
  return HSIC

def LinearHSICLoss(x, y, cuda=True):
  m,_ = x.shape #batch size
  K = LinearKernelMatrix(x)
  L = LinearKernelMatrix(y)
  H = torch.eye(m) - 1.0/m * torch.ones((m,m))
  if cuda:
    H = H.double().cuda() 
  HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
  return HSIC

def DiscreteConditionalExpecationTest(x, y, z):
  n, _ = x.shape
  if len(z.shape) > 1:
    temp_z = z[:,0]
  else:
    temp_z = z
    
  labels_in_batch_sorted, indices = torch.sort(temp_z)
  unique_ixs = 1 + (labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1]).nonzero()
  unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(temp_z)]

  estimate = 0
  for j in range(len(unique_ixs)-1):
    current_class_indices = unique_ixs[j], unique_ixs[j + 1]
    count = current_class_indices[1] - current_class_indices[0]
    if count < 2: 
      continue
    curr_class_slice = slice(*current_class_indices)
    curr_class_indices = indices[curr_class_slice].sort()[0]

    y_cond_z = torch.mean(y[curr_class_indices, :], dim=0, keepdim=True)
    estimate += torch.sum(x[curr_class_indices, :] * (y[curr_class_indices, :] - y_cond_z), dim=0)
  
  return estimate/n

def DiscreteConditionalHSICLoss(x, y, z, s_x=1, s_y=1, epsilon = 1e-6, cuda=False):
  """ adapted https://github.com/nv-research-israel/causal_comp/blob/7b26f00bd7b28d0e4cb80147e2ce302ead5cde75/train.py#L329 """
  if len(z.shape) > 1:
    temp_z = torch.squeeze(z)
  else:
    temp_z = z
  labels_in_batch_sorted, indices = torch.sort(temp_z)
  unique_ixs = 1 + (labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1]).nonzero()
  unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(temp_z)]
  
  hisc_loss = 0
  num_classes_calculated = 0
  for j in range(len(unique_ixs)-1):
    current_class_indices = unique_ixs[j], unique_ixs[j + 1]
    count = current_class_indices[1] - current_class_indices[0]
    if count < 2: 
      continue
    curr_class_slice = slice(*current_class_indices)
    curr_class_indices = indices[curr_class_slice].sort()[0]

    hisc_loss += HSICLoss(x[curr_class_indices, :], y[curr_class_indices, :])
    num_classes_calculated += 1
  
  return hisc_loss/num_classes_calculated

def DiscreteConditionalLinearHSICLoss(x, y, z, cuda=False):
  """ adapted https://github.com/nv-research-israel/causal_comp/blob/7b26f00bd7b28d0e4cb80147e2ce302ead5cde75/train.py#L329 """
  if len(z.shape) > 1:
    temp_z = torch.squeeze(z)
  else:
    temp_z = z
  labels_in_batch_sorted, indices = torch.sort(temp_z)
  unique_ixs = 1 + (labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1]).nonzero()
  unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(temp_z)]
  
  hisc_loss = 0
  num_classes_calculated = 0
  for j in range(len(unique_ixs)-1):
    current_class_indices = unique_ixs[j], unique_ixs[j + 1]
    count = current_class_indices[1] - current_class_indices[0]
    if count < 2: 
      continue
    curr_class_slice = slice(*current_class_indices)
    curr_class_indices = indices[curr_class_slice].sort()[0]

    hisc_loss += LinearHSICLoss(x[curr_class_indices, :], y[curr_class_indices, :])
    num_classes_calculated += 1
  
  return hisc_loss/num_classes_calculated

def ConditionalHSICLoss(x, y, z, s_x=1, s_y=1, s_z = 1, epsilon = 1e-6, cuda=False):
  n,_ = x.shape #batch size

  Kx = GaussianKernelMatrix(x,s_x)
  Ky = GaussianKernelMatrix(y,s_y)
  Kz = GaussianKernelMatrix(z,s_z)
  Gx_tilde = Centering_GramMatrix(Kx * Kz)
  Gy_tilde = Centering_GramMatrix(Ky * Kz)
  Rx_tilde = Gx_tilde @ torch.inverse(Gx_tilde + n * epsilon * torch.eye(n))
  Ry_tilde = Gy_tilde @ torch.inverse(Gy_tilde + n * epsilon * torch.eye(n))
  G_z_center = Centering_GramMatrix(Kz)
  Rz = G_z_center @ torch.inverse(G_z_center + n * epsilon * torch.eye(n))
  HSIC = torch.trace(Ry_tilde @ Rx_tilde - 2.0 * Ry_tilde @ Rx_tilde @ Rz+ Ry_tilde @ Rz @ Rx_tilde @ Rz) 

  return HSIC

def Centering_GramMatrix(K_n):
  n, _ = K_n.shape #batch size
  H = 1.0/n * torch.ones((n,n))
  return K_n - H @ K_n - K_n @ H + H @ K_n @ H


def ConditionalLinearHSICLoss(x, y, z, epsilon = 1e-5, cuda=False):
  n,_ = x.shape #batch size

  Kx = LinearKernelMatrix(x)
  Ky = LinearKernelMatrix(y)
  Kz = LinearKernelMatrix(z)
  Gx_tilde = Centering_GramMatrix(Kx * Kz)
  Gy_tilde = Centering_GramMatrix(Ky * Kz)
  Rx_tilde = Gx_tilde @ torch.inverse(Gx_tilde + n * epsilon * torch.eye(n))
  Ry_tilde = Gy_tilde @ torch.inverse(Gy_tilde + n * epsilon * torch.eye(n))
  G_z_center = Centering_GramMatrix(Kz)
  Rz = G_z_center @ torch.inverse(G_z_center + n * epsilon * torch.eye(n))
  HSIC = torch.trace(Ry_tilde @ Rx_tilde - 2.0 * Ry_tilde @ Rx_tilde @ Rz+ Ry_tilde @ Rz @ Rx_tilde @ Rz) 

  return HSIC

def BaseLoss(test_dataset, env, criterion, batch_size):
    loss = 0
    batch_num = 0
    
    for x, y in batchify(test_dataset, batch_size):
      f_beta = env.sample_base_classifer(x)

      loss += criterion(f_beta, y) 
      batch_num += 1

    print(f"Bse Test loss {loss.item()/batch_num} ")
    return loss.item()/batch_num


def standalone_tunning_test(trainer, config, test_dataset, adaptive = False, n_fine_tune_points = 1):
  finetuned_losses = [ ]

  for i in range(config.n_fine_tune_tests):
    finetune_dataset = torch.utils.data.Subset(test_dataset,  np.random.choice(len(test_dataset), n_fine_tune_points, replace=False))
    if not adaptive:
      model = trainer.finetune_test(finetune_dataset, rep_learning_flag = True)
      finetuned_loss = trainer.test(test_dataset, input_model = model, rep_learning_flag = True, print_flag=False)
      finetuned_losses.append(finetuned_loss)
    else:
      model = trainer.finetune_test(finetune_dataset, rep_learning_flag = True)
      finetuned_loss, _ = trainer.test(test_dataset, input_model = model, print_flag=False, rep_learning_flag = True)
      finetuned_losses.append(finetuned_loss)

  print(sum(finetuned_losses) / len(finetuned_losses))
  return finetuned_losses

def fine_tunning_test(trainer, config, test_finetune_dataset, test_dataset, n_fine_tune_points = 1, test_unlabelled_dataset = None, run_proj_gd = False):
  # Finetuning tests
  finetuned_loss = 0.0
  if run_proj_gd:
    projected_gd_finetuned_loss = 0.0

  x, y = test_finetune_dataset
  n_total_finetune_datapoints = len(x)

  for i in range(config.n_fine_tune_tests):
    perm = np.random.permutation(n_total_finetune_datapoints)
    
    try:
      x_perm = x[perm]
      y_perm = y[perm]
    except:
      x_perm = x
      y_perm = y

    partical_test_finetune_dataset = (x_perm[:n_fine_tune_points], y_perm[:n_fine_tune_points])

    if test_unlabelled_dataset is not None:
      if run_proj_gd:
        # model = trainer.finetune_test(partical_test_finetune_dataset, test_unlabeld_dataset =test_unlabelled_dataset, projected_gd=True)
        # _, proj_gd_loss_this_epoch = trainer.test(test_dataset, print_flag=False)
        # projected_gd_finetuned_loss+=proj_gd_loss_this_epoch
        model = trainer.finetune_test(partical_test_finetune_dataset, test_unlabeld_dataset =test_unlabelled_dataset,  projected_gd=True)
        proj_gd_loss_this_epoch, _ = trainer.test(test_dataset, input_model = model, print_flag=False)
        finetuned_loss+=proj_gd_loss_this_epoch

      # trainer.finetune_test(partical_test_finetune_dataset, test_unlabelled_dataset)
      # _, gd_loss_this_epoch = trainer.test(test_dataset, print_flag=False)
      model = trainer.finetune_test(partical_test_finetune_dataset, test_unlabeld_dataset =test_unlabelled_dataset)
      gd_loss_this_epoch, _ = trainer.test(test_dataset, input_model = model, print_flag=False)
      finetuned_loss+=gd_loss_this_epoch
    else:
      model = trainer.finetune_test(partical_test_finetune_dataset)
      finetuned_loss+=trainer.test(test_dataset, input_model = model, print_flag=False)


  finetuned_loss /= config.n_fine_tune_tests

  if run_proj_gd:
    projected_gd_finetuned_loss /= config.n_fine_tune_tests
    print(finetuned_loss, projected_gd_finetuned_loss)
    return finetuned_loss, projected_gd_finetuned_loss
  else:
    print(finetuned_loss)
    return finetuned_loss

""" copy from https://github.com/p-lambda/wilds/blob/a7a452c80cad311cf0aabfd59af8348cba1b9861/examples/models/layers.py """
import torch.nn.functional as F

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x

""" copy from https://github.com/p-lambda/wilds/blob/a7a452c80cad311cf0aabfd59af8348cba1b9861/examples/models/initializer.py"""
def initialize_torchvision_model(name, d_out, **kwargs):
    import torchvision

    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model
