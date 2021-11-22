import numpy as np

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