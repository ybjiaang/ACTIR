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