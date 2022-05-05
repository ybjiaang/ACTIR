import numpy as np
import torch 
from torchvision import datasets
import math


# follow https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
class ColorMnist(object):
    def __init__(self, test_finetune_size = 10, test_unlabled_size=100):
        super(ColorMnist, self).__init__()
        self.num_total_envs = 4
        self.num_train_evns = 2
        self.test_finetune_size = test_finetune_size
        self.test_unlabled_size = test_unlabled_size
        self.input_dim = 2 * 14 * 14
        self.num_class = 2

        mnist = datasets.MNIST('~/dataset/mnist', train=True, download=True)

        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:55000], mnist.targets[50000:55000])
        mnist_test = (mnist.data[55000:], mnist.targets[55000:])

        self.mnist_val = mnist_val
        
        # train data
        self.train_data_by_season = [
            self.make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.3),
            self.make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.05),
        ]

        # val data
        self.val_data_by_season = [
            self.make_environment(mnist_val[0], mnist_val[1], 0.8),
        ]

        # test data
        test_data_all = self.make_environment(mnist_test[0], mnist_test[1], 0.9)
        self.test_data_finetune = (test_data_all[0][:self.test_finetune_size], test_data_all[1][:self.test_finetune_size])
        self.test_data_unlabled = (test_data_all[0][self.test_finetune_size: self.test_finetune_size + self.test_unlabled_size], 
        test_data_all[1][self.test_finetune_size: self.test_finetune_size + self.test_unlabled_size])
        self.test_data = (test_data_all[0][self.test_finetune_size + self.test_unlabled_size:], test_data_all[1][self.test_finetune_size + self.test_unlabled_size:])
        

    def make_environment(self, images, labels, e, return_color = False):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))#[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0    
        
        if return_color:
            # return (images.reshape((-1, 2*14*14)).float() / 255., colors[:, None])
            return (images.float() / 255., colors[:, None])

        # return (images.reshape((-1, 2*14*14)).float() / 255., labels.long())
        return (images.float() / 255., labels.long())

    def sample_envs_z(self, env_ind = 2, n = 100):
        return self.make_environment(self.mnist_val[0], self.mnist_val[1], 0.5, return_color=True)

    def z_range(self):
        return [0, 1]

    def sample_envs(self, env_ind=0, train_val_test = 0):
        # train
        if train_val_test == 0:
            return self.train_data_by_season[env_ind]

        # val
        if train_val_test == 1:
            return self.val_data_by_season[env_ind]

        if train_val_test == 2:
            return self.test_data_finetune, self.test_data_unlabled, self.test_data



if __name__ == '__main__':
    env = ColorMnist()