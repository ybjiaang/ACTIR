# -*- coding: utf-8 -*-
"""Adaptive-Invariant.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c5CDAFfDlkuhe9HMwS5Al9RR6B_b3FQ-
"""

# Commented out IPython magic to ensure Python compatibility.
import torch, torchvision
from torch import nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import argparse
# %matplotlib inline
import csv
import os
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from dataset.syn_env import CausalControlDataset, AntiCausalControlDataset, AntiCausalControlDatasetMultiClass
from dataset.bike_env import BikeSharingDataset
from dataset.color_mnist import ColorMnist
from dataset.camelyon17 import Camelyon17
from dataset.vlcs import VLCS
from models.adap_invar import AdaptiveInvariantNN
from models.base_classifer import BaseClass
from trainers.adap_invar_trainer import AdaptiveInvariantNNTrainer
from trainers.adap_invar_trainer_meta import AdaptiveInvariantNNTrainerMeta
from trainers.erm import ERM
from trainers.irm import IRM
from trainers.hsic import HSIC
from trainers.maml import LinearMAML
from misc import create_DF, standalone_tunning_test, fine_tunning_test, BaseLoss, initialize_torchvision_model, FolderDataset

# seaborn stuff
err_sty = 'band'

def set_seed(seed):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class Identity(nn.Module):
  """An identity layer"""
  def __init__(self):
      super(Identity, self).__init__()

  def forward(self, x):
      return x

class ResNet(torch.nn.Module):
   """ResNet with the softmax chopped off and the batchnorm frozen"""
   def __init__(self, model):
      super(ResNet, self).__init__()
      self.network = model
      self.freeze_bn()
      
   def forward(self, x):
    """Encode x into a feature vector of size n_outputs."""
    return self.network(x)
    
   def train(self, mode=True):
    """
    Override the default train() to freeze the BN parameters
    """
    super().train(mode)
    self.freeze_bn()
  
   def freeze_bn(self):
    for m in self.network.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.eval()

if __name__ == '__main__':
  g = torch.Generator()
  g.manual_seed(0)
  set_seed(0)

  parser = argparse.ArgumentParser()

  parser.add_argument('--n_envs', type=int, default= 5, help='number of enviroments per training epoch')
  parser.add_argument('--batch_size', type=int, default= 128, help='batch size')
  parser.add_argument('--irm_reg_lambda', type=float, default= 10, help='regularization coeff for irm')
  parser.add_argument('--reg_lambda', type=float, default= 1,help='regularization coeff for adaptive invariant learning')
  parser.add_argument('--reg_lambda_2', type=float, default= 10, help='second regularization coeff for adaptive invariant learning')
  parser.add_argument('--gamma', type=float, default= 0.9, help='interpolation parmameter')
  parser.add_argument('--phi_odim',  type=int, default= 3, help='Phi output size')
  parser.add_argument('--n_outer_loop',  type=int, default= 100, help='outer loop size')
  parser.add_argument('--n_finetune_loop',  type=int, default= 20, help='finetune loop size')

  # different models
  parser.add_argument('--model_name', type=str, default= "adp_invar", help='type of modesl. current support: adp_invar, erm')
  parser.add_argument('--compare_all_invariant_models', action='store_true', help='compare all invariant models')
  parser.add_argument('--classification', action='store_true', help='if the tast is classification, set this flag to enable correct prediction, labels has to be between [0, ..., n]')

  # finetune
  parser.add_argument('--run_fine_tune_test', action='store_true', help='run finetunning tests')
  parser.add_argument('--n_fine_tune_tests', type=int, default= 10, help='number of fine tunning tests')
  parser.add_argument('--n_fine_tune_points', nargs='+', type=int, help='how many points for finetuning')

  # dataset
  parser.add_argument('--dataset', type=str, default= "syn", help='type of experiment: syn, bike, color_mnist, vlcs, camelyon17')
  
  # synthetic dataset specifics
  parser.add_argument('--causal_dir_syn', type=str, default= "anti", help='anti or causal or anti-multi')
  parser.add_argument('--syn_dataset_train_size', type=int, default= 1024, help='size of synthetic dataset per env')

  # bike sharing specifics
  parser.add_argument('--bike_test_season', type=int, default= 1, help='what season to test our model')
  parser.add_argument('--bike_year', type=int, default= 0, help='what year to test our model')

  # camelyon17 specifics
  parser.add_argument('--data_dir', type=str, default= "dataset/VLCS", help='where to put data')

  # standalone finetune test
  parser.add_argument('--run_fine_tune_test_standalone', action='store_true', help='run standalone finetunning tests')

  # misc
  parser.add_argument('--print_base_graph', action='store_true', help='whether to print base classifer comparision graph, can only be used in 1 dimension')
  parser.add_argument('--verbose', action='store_true', help='verbose or not')
  parser.add_argument('--cvs_dir', type=str, default= "./test.cvs", help='path to the cvs file')
  parser.add_argument('--model_save_dir', type=str, default= "./saved_model", help='where to save model')
  parser.add_argument('--hyper_param_tuning', action='store_true', help='whether to do hyper-parameter tuning')
  parser.add_argument('--save_test_phi', action='store_true', help='whether to save phi for finetune test')

  args = parser.parse_args()

  # Get cpu or gpu device for training.
  args.device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {args.device} device")
  print(args.reg_lambda, args.reg_lambda_2, args.n_outer_loop)

  # create dictionary if not exist
  if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)

  # dataset related flags
  args.torch_loader = False
  if args.run_fine_tune_test_standalone:
    args.torch_loader = True
  args.num_workers = 0

  # create datasets
  if args.dataset == "syn":
    if args.causal_dir_syn == "causal":
      print("Sampling from causal synthetic datasets")
      # env = CausalAdditiveNoSpurious()
      env = CausalControlDataset()
    if args.causal_dir_syn == "anti":
      print("Sampling from causal anti datasets")
      # env = AntiCausal()
      env = AntiCausalControlDataset()

    if args.causal_dir_syn == "anti-multi":
      print("Sampling from causal anti multi-class datasets")
      env = AntiCausalControlDatasetMultiClass()  

    args.n_envs = env.num_train_evns

    # create training data
    train_dataset = []
    for i in range(env.num_train_evns):
      x, y = env.sample_envs(i, n = args.syn_dataset_train_size)
      train_dataset.append((x,y))

    # create val dataset
    x, y = env.sample_envs(env.num_train_evns, n = args.syn_dataset_train_size)
    val_dataset = (x, y)

    # create test dataset
    x, y = env.sample_envs(env.num_train_evns + 1, n = 10)
    test_finetune_dataset = (x, y)

    x, _ = env.sample_envs(env.num_train_evns + 1, n = args.syn_dataset_train_size)
    test_unlabelled_dataset = (x,)

    x, y = env.sample_envs(env.num_train_evns + 1, n = args.syn_dataset_train_size)
    test_dataset = (x, y)
  
  elif args.dataset == "vlcs":
    args.torch_loader = True
    env = VLCS(args)
    train_dataset = env.train_data_list
    val_dataset = env.val_data_list
    test_finetune_dataset, test_unlabelled_dataset, test_dataset= env.sample_envs(train_val_test=2)

  elif args.dataset == "camelyon17":
      args.torch_loader = True
      print("camelyon17 dataset")
      env = Camelyon17(args)
      train_dataset = env.train_data_list
      val_dataset = env.val_data_list
      test_finetune_dataset, test_unlabelled_dataset, test_dataset= env.sample_envs(train_val_test=2)

  else:
    if args.dataset == "bike":
      print("bikesharing dataset")
      env = BikeSharingDataset(test_season=args.bike_test_season, year=args.bike_year)
    if args.dataset == "color_mnist":
      print("color mnist dataset")
      env = ColorMnist()
    

    args.n_envs = env.num_train_evns

    # create training data
    train_dataset = []
    for i in range(env.num_train_evns):
      x, y = env.sample_envs(env_ind=i, train_val_test=0)
      train_dataset.append((x,y))

    # create val dataset
    x, y = env.sample_envs(train_val_test=1)
    val_dataset = (x, y)

    # create test dataset
    test_finetune_dataset, test_unlabelled_dataset, test_dataset= env.sample_envs(train_val_test=2)

  if args.hyper_param_tuning:
    test_dataset = val_dataset

  # loss fn
  if args.classification:
    criterion = torch.nn.CrossEntropyLoss()
  else:
    criterion = torch.nn.MSELoss(reduction='mean')

  # define models
  input_dim = env.input_dim
  phi_odim = args.phi_odim
  if args.classification:
    out_dim = env.num_class
    args.num_class = env.num_class
    if out_dim > phi_odim:
      args.phi_odim = out_dim
      phi_odim = args.phi_odim
  else:
    out_dim = 1
    args.num_class = 1
    
  if args.dataset == "syn":
    Phi = nn.Sequential(
              nn.Linear(input_dim, 4),
              nn.ReLU(),
              nn.Linear(4, phi_odim)
              # nn.Linear(input_dim, phi_odim)
          )

  if args.dataset == "bike":
    Phi = nn.Sequential(
              nn.Linear(input_dim, 8),
              nn.ReLU(),
              nn.Linear(8, 16),
              nn.ReLU(),
              nn.Linear(16, phi_odim)
          )

  if args.dataset == "color_mnist":
    hidden_dims = 512
    lin1 = nn.Linear(input_dim, hidden_dims)
    lin2 = nn.Linear(hidden_dims, hidden_dims)
    lin3 = nn.Linear(hidden_dims, phi_odim)
    for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
    Phi = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

  if args.dataset == "vlcs":
    """use resnet18"""
    args.model_kwargs = {
            'pretrained': True,
        }
    Phi = initialize_torchvision_model(
                name='resnet18',
                d_out=None,
                **args.model_kwargs)
    args.phi_odim = Phi.d_out
    Phi = ResNet(Phi)

  if args.dataset == "camelyon17":
    """use resnet18"""
    args.model_kwargs = {
            'pretrained': True,
        }

# <<<<<<< HEAD
#     # feature = initialize_torchvision_model(
#     #             name='resnet18',
#     #             d_out=None,
#     #             **args.model_kwargs)
#     # # args.phi_odim = Phi.d_out
#     # print(feature.d_out)
#     # args.phi_odim = 16
# =======
    Phi = initialize_torchvision_model(
                name='resnet18',
                d_out=128,
                **args.model_kwargs)
    args.phi_odim = Phi.d_out

    # args.phi_odim = 32
    # lin = nn.Linear(feature.d_out, args.phi_odim)
    # Phi = nn.Sequential(feature, lin)

  
    # reshape = torch.nn.Flatten(start_dim=-3, end_dim=- 1)
    # hidden_dims = 256
    # lin1 = nn.Linear(input_dim, hidden_dims)
    # lin2 = nn.Linear(hidden_dims, hidden_dims)
    # lin3 = nn.Linear(hidden_dims, phi_odim)
    # for lin in [lin1, lin2, lin3]:
    #     nn.init.xavier_uniform_(lin.weight)
    #     nn.init.zeros_(lin.bias)
    # Phi = nn.Sequential(reshape, lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    
  """ HSIC """
  if args.model_name == "hsic" or args.compare_all_invariant_models:
    if not args.run_fine_tune_test_standalone:
      model = BaseClass(input_dim, Phi, phi_dim = args.phi_odim).to(args.device)
      trainer = HSIC(model, criterion, args)

      print("hsic training...")
      trainer.train(train_dataset, args.batch_size)

      print("hsic test...")
      hsic_loss = trainer.test(test_dataset)
      # hsic_loss = 0

  """ IRM """
  if args.model_name == "irm" or args.compare_all_invariant_models:
    if not args.run_fine_tune_test_standalone:
      model = BaseClass(input_dim, Phi, out_dim = out_dim, phi_dim = args.phi_odim).to(args.device)
      trainer = IRM(model, criterion, args)
      
      print("irm training...")
      trainer.train(train_dataset, args.batch_size)

      print("irm test...")
      irm_loss = trainer.test(test_dataset)

      if args.hyper_param_tuning:
        with open(args.cvs_dir, 'a', newline='') as file: 
          writer = csv.writer(file)
          row = [args.irm_reg_lambda, irm_loss]
          writer.writerow(row)


  """ ERM """
  if args.model_name == "erm" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi, out_dim = out_dim, phi_dim = args.phi_odim)
    trainer = ERM(model, criterion, args)

    if not args.run_fine_tune_test_standalone:
      model.to(args.device)
      trainer = ERM(model, criterion, args)
      print("erm training...")
      trainer.train(train_dataset, args.batch_size)
      trainer.save_model()

      print("erm test...")
      erm_loss = trainer.test(test_dataset)

      # print("erm val...")
      # erm_loss_val = trainer.test(val_dataset)

      if args.run_fine_tune_test:
          for n_finetune_loop in [20]:
              print(n_finetune_loop)
              trainer.config.n_finetune_loop = n_finetune_loop
              for learning_rate in [1e-2]:
                  print("learning rate:" + str(learning_rate))
                  trainer.fine_inner_lr = learning_rate
                  # trainer.test_inner_optimizer = torch.optim.Adam(trainer.model.etas.parameters(), lr=learning_rate)
              # anti_causal_finetune_loss = []
                  erm_finetune_loss = []
                  for n_tune_points in  args.n_fine_tune_points:
                      erm_finetune_loss.append(fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points))
    else:
      model.load_state_dict(torch.load(trainer.model_path, map_location=torch.device('cpu'))['model_state_dict'])
      model.to(args.device)
      trainer = ERM(model, criterion, args)
      embedding_dataset = FolderDataset(trainer.emb_path)

      erm_acc_lists = []
      for n_tune_points in  args.n_fine_tune_points:
        erm_acc_lists.append(standalone_tunning_test(trainer, args, embedding_dataset, n_fine_tune_points = n_tune_points))
      

  """ MAML """
  if args.model_name == "maml" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi, out_dim = out_dim, phi_dim = args.phi_odim)
    trainer = LinearMAML(model, criterion, args)

    if not args.run_fine_tune_test_standalone:
      model.to(args.device)
      trainer = LinearMAML(model, criterion, args)
      print("maml training...")
      maml_train_loss = trainer.train(train_dataset, args.batch_size)
      trainer.save_model()

      torch.save(trainer.model, './maml.pt')
      print("maml test...")
      maml_loss = trainer.test(test_dataset)

      if args.run_fine_tune_test:
          for n_finetune_loop in [20]:
              print(n_finetune_loop)
              trainer.config.n_finetune_loop = n_finetune_loop
              for learning_rate in [1e-2]:
                  print("learning rate:" + str(learning_rate))
                  # trainer.test_inner_optimizer = torch.optim.Adam(trainer.model.etas.parameters(), lr=learning_rate)
                  trainer.fine_inner_lr = learning_rate
              # anti_causal_finetune_loss = []
                  maml_finetune_loss = []
                  for n_tune_points in  args.n_fine_tune_points:
                      maml_finetune_loss.append(fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points))
        # maml_finetune_loss = []
        # for n_tune_points in  args.n_fine_tune_points:
        #  maml_finetune_loss.append(fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points))
    else:
      pass
      # model.load_state_dict(torch.load(trainer.model_path)['model_state_dict'])
      # model.to(args.device)
      # trainer = LinearMAML(model, criterion, args)
      # embedding_dataset = FolderDataset(trainer.emb_path)

      # maml_acc_lists = []
      # for n_tune_points in  args.n_fine_tune_points:
      #   maml_acc_lists.append(standalone_tunning_test(trainer, args, embedding_dataset, n_fine_tune_points = n_tune_points))
      
  """ Adaptive Invariant Anti Causal """
  if args.model_name == "adp_invar_anti_causal" or args.compare_all_invariant_models:
    model = AdaptiveInvariantNN(args.n_envs, input_dim, Phi, args, out_dim = out_dim, phi_dim = args.phi_odim)
    trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args, causal_dir = False)

    if not args.run_fine_tune_test_standalone:
      model.to(args.device)
      trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args, causal_dir = False)
      print("adp_invar anti-causal training...")
      trainer.train(train_dataset, args.batch_size)
      trainer.save_model()

      print("adp_invar anti-causal test...")
      adp_invar_anti_causal_base_loss, _ = trainer.test(test_dataset)

      # print("adp_invar anti-causal test val ...")
      # adp_invar_anti_causal_base_loss_val, _ = trainer.test(val_dataset)
      adp_invar_anti_causal_base_loss_val = 0

      if args.hyper_param_tuning:
        with open(args.cvs_dir, 'a', newline='') as file: 
          writer = csv.writer(file)
          row = [args.reg_lambda, args.reg_lambda_2, args.gamma, args.n_outer_loop, adp_invar_anti_causal_base_loss, adp_invar_anti_causal_base_loss_val]
          writer.writerow(row)

      if args.run_fine_tune_test:
        if True:
          for n_finetune_loop in [20]:
            print(n_finetune_loop)
            trainer.config.n_finetune_loop = n_finetune_loop
            for learning_rate in [1e-2]:
              print("learning rate:" + str(learning_rate))
              trainer.test_inner_optimizer = torch.optim.Adam(trainer.model.etas.parameters(), lr=learning_rate)
              anti_causal_finetune_loss = []
              for n_tune_points in  args.n_fine_tune_points:
                anti_causal_finetune_loss.append(fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points, test_unlabelled_dataset))
        else:
          anti_causal_finetune_loss = []
          for n_tune_points in  args.n_fine_tune_points:
            anti_causal_finetune_loss.append(fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points, test_unlabelled_dataset))
    else:
      model.load_state_dict(torch.load(trainer.model_path, map_location=torch.device('cpu'))['model_state_dict'])
      model.to(args.device)
      trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args, causal_dir = False)
      embedding_dataset = FolderDataset(trainer.emb_path)

      adp_invar_anti_acc_lists = []
      for n_tune_points in  args.n_fine_tune_points:
        adp_invar_anti_acc_lists.append(standalone_tunning_test(trainer, args, embedding_dataset, adaptive=True, n_fine_tune_points = n_tune_points))


  """ Adaptive Invariant Causal """
  if args.model_name == "adp_invar" or args.compare_all_invariant_models:
    if not args.run_fine_tune_test_standalone:
      model = AdaptiveInvariantNN(args.n_envs, input_dim, Phi, args, out_dim = out_dim, phi_dim = args.phi_odim).to(args.device)
      trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args)

      print("adp_invar training...")
      trainer.train(train_dataset, args.batch_size)
      trainer.test_inner_optimizer = torch.optim.Adam(trainer.model.etas.parameters(), lr=1e-3)
      torch.save(trainer.model, './anti.pt')
      print("adp_invar test...")
      adp_invar_base_loss, adp_invar_loss = trainer.test(test_dataset)

      trainer.config.n_finetune_loop = 2
      
      if args.hyper_param_tuning:
        with open(args.cvs_dir, 'a', newline='') as file: 
          writer = csv.writer(file)
          row = [args.reg_lambda, args.reg_lambda_2, args.gamma, adp_invar_base_loss]
          writer.writerow(row)

      if args.run_fine_tune_test:
        causal_proj_gd_losses = []
        causal_gd_losses = []
        for n_tune_points in  args.n_fine_tune_points:
          causal_gd_loss, causal_proj_gd_loss = fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points, test_unlabelled_dataset, True)
          causal_proj_gd_losses.append(causal_proj_gd_loss)
          causal_gd_losses.append(causal_gd_loss)

  if args.compare_all_invariant_models:
    if not args.run_fine_tune_test_standalone:
      with open(args.cvs_dir, 'a', newline='') as file: 
        writer = csv.writer(file)
        row = [hsic_loss, irm_loss, erm_loss, maml_train_loss, maml_loss, adp_invar_anti_causal_base_loss, adp_invar_base_loss, adp_invar_loss]
        if args.run_fine_tune_test:
          for i, n_tune_points in enumerate(args.n_fine_tune_points):
            row.append(erm_finetune_loss[i])
            row.append(maml_finetune_loss[i])
            row.append(anti_causal_finetune_loss[i])
            row.append(causal_proj_gd_losses[i])
            row.append(causal_gd_losses[i])
        writer.writerow(row)
      print(hsic_loss, irm_loss, erm_loss, maml_train_loss, maml_loss, adp_invar_anti_causal_base_loss, adp_invar_base_loss, adp_invar_loss)
    else:
      fig = plt.figure()
      plt.clf()

      df = create_DF(np.array(erm_acc_lists).T, np.array(args.n_fine_tune_points))
      sns.lineplot(x='num of finetuning points', y='finetuned accuary', err_style=err_sty, data = df, ci='sd', label = 'erm')

      df = create_DF(np.array(adp_invar_anti_acc_lists).T, np.array(args.n_fine_tune_points))
      sns.lineplot(x='num of finetuning points', y='finetuned accuary', err_style=err_sty, data = df, ci='sd', label = 'adaptive causal')
    
      # df = create_DF(np.array(maml_acc_lists).T, np.array(args.n_fine_tune_points))
      # sns.lineplot(x='num of finetuning points', y='finetuned accuary', err_style=err_sty, data = df, ci='sd', label = 'maml')

      # other plot stuff
      ax = plt.gca()
      plt.xlabel('# Finetunning Points', fontsize=20)
      plt.ylabel('Finetuned Accuary', fontsize=20)
      plt.setp(ax.get_xticklabels(), fontsize=10)
      plt.setp(ax.get_yticklabels(), fontsize=10)
      plt.tight_layout()
      plt.xticks(np.array(args.n_fine_tune_points))
      plt.legend(loc=1, fontsize=15, title='Algo')
      # plt.ylim([0, 1])

      plt.savefig("all_fine_tune.png")
