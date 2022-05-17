# modified from here
# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

import numpy as np
from torchvision import transforms
import torch 
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
import os
import argparse
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultipleDomainDataset:

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class PACS(MultipleDomainDataset):
    def __init__(self, config, root = 'dataset/PACS', test_finetune_size = 1000):
        super().__init__()
        self.config = config
        self.num_train_evns = 3
        environments = [f.name for f in os.scandir(config.data_dir) if f.is_dir()]
        environments = sorted(environments)
        test_i = config.test_index
        assert config.val_index > 0
        val_i = (test_i + config.val_index) % 4
        if config.downsample:
            self.input_dim = 112 * 112 * 3
        else:
            self.input_dim = 224 * 224 * 3
        print("test id:" + str(test_i))
        print("val id:" + str(val_i))
        print(self.input_dim)

        if config.downsample:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((112,112)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]) 

        self.train_data_list = []
        self.val_data_list = []
        for i, environment in enumerate(environments):

            env_transform = transform

            path = os.path.join(config.data_dir, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)  
            # def make_weights_for_balanced_classes(images, nclasses):                        
            #     count = [0] * nclasses                                                      
            #     for item in images:                                                         
            #         count[item[1]] += 1                                                     
            #     weight_per_class = [0.] * nclasses   
            #     N = float(sum(count))  
            #     c_list = []  
            #     for c in count:
            #         c = c/N       
            #         c_list.append(c)                           
            #     print(c_list)                                          

            # print(make_weights_for_balanced_classes(env_dataset.imgs, len(env_dataset.classes)))
            
            print(len(env_dataset))
            if i == test_i:
                self.test_data_finetune = torch.utils.data.Subset(env_dataset, np.random.choice(len(env_dataset), test_finetune_size, replace=False))
                self.test_data_unlabled = self.test_data_finetune
                self.test_data_list = env_dataset
            elif i == val_i:
                self.val_data_list = env_dataset
                if not config.hyper_param_tuning:
                    self.train_data_list.append(env_dataset)
            else:
                self.train_data_list.append(env_dataset)
        if config.downsample:
            self.input_shape = (3, 112, 112,)
        else:
            self.input_shape = (3, 224, 224,)
        self.num_class = len(self.train_data_list[-1].classes)

    def sample_envs(self, env_ind=0, train_val_test = 0):
        # train
        if train_val_test == 0:
            return self.train_data_list[env_ind]

        # val
        if train_val_test == 1:
            return self.val_data_list[env_ind]

        if train_val_test == 2:
            return self.test_data_finetune, self.test_data_unlabled, self.test_data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_envs', type=int, default= 5, help='number of enviroments per training epoch')
    parser.add_argument('--batch_size', type=int, default= 128, help='batch size')
    parser.add_argument('--irm_reg_lambda', type=float, default= 573.6152510448682, help='regularization coeff for irm')
    parser.add_argument('--reg_lambda', type=float, default= 1000, help='regularization coeff for adaptive invariant learning')
    parser.add_argument('--reg_lambda_2', type=float, default= 0.4, help='second regularization coeff for adaptive invariant learning')
    parser.add_argument('--gamma', type=float, default= 0.9, help='interpolation parmameter')
    parser.add_argument('--phi_odim',  type=int, default= 3, help='Phi output size')

    # different models
    parser.add_argument('--model_name', type=str, default= "adp_invar", help='type of modesl. current support: adp_invar, erm')
    parser.add_argument('--compare_all_invariant_models', action='store_true', help='compare all invariant models')
    parser.add_argument('--classification', action='store_true', help='if the tast is classification, set this flag to enable correct prediction, labels has to be between [0, ..., n]')

    # finetune
    parser.add_argument('--run_fine_tune_test', action='store_true', help='run finetunning tests')
    parser.add_argument('--n_fine_tune_tests', type=int, default= 10, help='number of fine tunning tests')
    parser.add_argument('--n_fine_tune_points', nargs='+', type=int, help='how many points for finetuning')

    # dataset
    parser.add_argument('--dataset', type=str, default= "syn", help='type of experiment: syn, bike, color_mnist')
    
    parser.add_argument('--test_index', type=int, default= 3, help='which dataset to test')
    parser.add_argument('--val_index', type=int, default= 1, help='which dataset to val, it has to be strictly positive')
    parser.add_argument('--downsample', action='store_true', help='whether to downsample')
    parser.add_argument('--resnet_dim', type=int, default= 8, help='resnet dimension')

    # synthetic dataset specifics
    parser.add_argument('--causal_dir_syn', type=str, default= "anti", help='anti or causal')
    parser.add_argument('--syn_dataset_train_size', type=int, default= 1024, help='size of synthetic dataset per env')

    # bike sharing specifics
    parser.add_argument('--bike_test_season', type=int, default= 1, help='what season to test our model')
    parser.add_argument('--bike_year', type=int, default= 0, help='what year to test our model')

    # misc
    parser.add_argument('--print_base_graph', action='store_true', help='whether to print base classifer comparision graph, can only be used in 1 dimension')
    parser.add_argument('--verbose', action='store_true', help='verbose or not')
    parser.add_argument('--cvs_dir', type=str, default= "./test.cvs", help='path to the cvs file')
    parser.add_argument('--hyper_param_tuning', action='store_true', help='whether to do hyper-parameter tuning')

    parser.add_argument('--data_dir', type=str, default= "dataset/PACS", help='where to put data')
    args = parser.parse_args()
    env = PACS(args)
