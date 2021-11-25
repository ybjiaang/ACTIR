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


def HSICLoss(x, y, s_x=1, s_y=1, cuda=False):
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    if cuda:
      H = H.double().cuda() 
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC