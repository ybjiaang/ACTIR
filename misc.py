import numpy as np
import torch 

def batchify(dataset, batch_size):
  x, y = dataset
  total_length = x.shape[0]
  nloops = np.ceil(total_length/batch_size).astype(int)

  def creatDataSet():
    for i in range(nloops):
      start = i*batch_size
      if (start + batch_size) >= total_length:
        yield x[start:], y[start:]
      else:
        yield x[start : start + batch_size], y[start : start + batch_size]

  return creatDataSet()

def env_batchify(dataset, batch_size):
  n_envs = len(dataset)
  all_lens = np.zeros(n_envs)
  for i, dataset_per_env in enumerate(dataset):
    all_lens[i] = dataset_per_env[0].shape[0]

  total_length_min = np.min(all_lens)
  nloops = np.ceil(total_length_min/batch_size).astype(int)

  def creatDataSet():
    for i in range(nloops):
      start = i*batch_size
      train_sqt_set = []

      for env_ind in range(n_envs):
        x, y = dataset[env_ind]
        if (start + batch_size) >= all_lens[env_ind]:
          train_sqt_set.append((x[start:], y[start:]))
        else:
          train_sqt_set.append((x[start : start + batch_size], y[start : start + batch_size]))

      yield train_sqt_set
      
  return creatDataSet()

def maml_batchify(dataset, batch_size):
  n_envs = len(dataset)
  all_lens = np.zeros(n_envs)
  for i, dataset_per_env in enumerate(dataset):
    all_lens[i] = dataset_per_env[0].shape[0]

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
          train_sqt_set.append((x[start::2], y[start::2]))
          train_qrt_set.append((x[start+1::2], y[start+1::2]))

        else:
          train_sqt_set.append((x[start : start + batch_size//2], y[start : start + batch_size//2]))
          train_qrt_set.append((x[start + batch_size//2 : start + batch_size], y[start + batch_size//2 : start + batch_size]))

      yield train_sqt_set, train_qrt_set
      
  return creatDataSet()

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)


def HSICLoss(x, y, s_x=1, s_y=1, cuda=False, return_matrix = False):
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    if cuda:
      H = H.double().cuda() 
    if return_matrix:
      return torch.mm(L,torch.mm(H,torch.mm(K,H)))
    else:
      HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
      return HSIC

def ConditionalHSICLoss(x, y, z, s_x=1, s_y=1, s_z = 1, cuda=False):
  m,_ = x.shape #batch size
  assert(s_z == 1)
  x_dd = torch.cat((x, z), dim=1)
  
  Simga_xdd_y = HSICLoss(x_dd, y, s_x + s_z, s_y, return_matrix = True)
  Sigma_xdd_z = HSICLoss(x_dd, z, s_x + s_z, s_z, return_matrix = True)
  Sigma_z_z = HSICLoss(z, z, s_z, s_z, return_matrix = True)
  Sigma_z_y = HSICLoss(z, y, s_z, s_y, return_matrix = True)

  HSIC = torch.trace(Simga_xdd_y - Sigma_xdd_z @ Sigma_z_y / (Sigma_z_z + 1e-5))/((m-1)**2)

  return HSIC

  

def fine_tunning_test(trainer, config, test_finetune_dataset, test_dataset, n_fine_tune_points = 1, test_unlabelled_dataset = None, run_proj_gd = False):
  # Finetuning tests
  finetuned_loss = 0.0
  if run_proj_gd:
    projected_gd_finetuned_loss = 0.0

  x, y = test_finetune_dataset
  n_total_finetune_datapoints = x.shape[0]

  for i in range(config.n_fine_tune_tests):
    perm = np.random.permutation(n_total_finetune_datapoints)
    
    x_perm = x[perm]
    y_perm = y[perm]

    partical_test_finetune_dataset = (x_perm[:n_fine_tune_points,:], y_perm[:n_fine_tune_points])

    if test_unlabelled_dataset is not None:
      if run_proj_gd:
        trainer.finetune_test(partical_test_finetune_dataset, test_unlabelled_dataset, projected_gd=True)
        _, proj_gd_loss_this_epoch = trainer.test(test_dataset, print_flag=False)
        projected_gd_finetuned_loss+=proj_gd_loss_this_epoch

      trainer.finetune_test(partical_test_finetune_dataset, test_unlabelled_dataset)
      _, gd_loss_this_epoch = trainer.test(test_dataset, print_flag=False)
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