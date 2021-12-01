# -*- coding: utf-8 -*-
"""Adaptive-Invariant.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c5CDAFfDlkuhe9HMwS5Al9RR6B_b3FQ-
"""

# Commented out IPython magic to ensure Python compatibility.
import torch 
from torch import nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import argparse
# %matplotlib inline

import matplotlib.pyplot as plt

from dataset.syn_env import CausalAdditiveNoSpurious, CausalHiddenAdditiveNoSpurious, AntiCausal
from models.adap_invar import AdaptiveInvariantNN
from models.base_classifer import BaseClass
from trainers.adap_invar_trainer import AdaptiveInvariantNNTrainer
from trainers.erm import ERM
from trainers.irm import IRM
from trainers.hsic import HSIC
from trainers.maml import LinearMAML


if __name__ == '__main__':
  torch.manual_seed(0)
  random.seed(0)
  np.random.seed(0)

  parser = argparse.ArgumentParser()

  parser.add_argument('--n_envs', type=int, default= 5, help='number of enviroments per training epoch')
  parser.add_argument('--batch_size', type=int, default= 32, help='batch size')
  parser.add_argument('--reg_lambda', type=float, default= 0.5, help='regularization coeff for adaptive invariant learning')
  parser.add_argument('--phi_odim',  type=int, default= 8, help='Phi output size')

  # different models
  parser.add_argument('--model_name', type=str, default= "adp_invar", help='type of modesl. current support: adp_invar, erm')
  parser.add_argument('--compare_all_invariant_models', type=bool, default=False, help='compare all invariant models')

  # dataset
  parser.add_argument('--dataset', type=str, default= "syn", help='type of experiment')
  parser.add_argument('--causal_dir_syn', type=str, default= "anti", help='anti or causal')
  # synthetic dataset specifics
  parser.add_argument('--syn_dataset_train_size', type=int, default= 256, help='size of synthetic dataset per env')

  # misc
  parser.add_argument('-print_base_graph', type=bool, default=False, help='whether to print base classifer comparision graph, can only be used in 1 dimension')
  parser.add_argument('-verbose', type=bool, default=False, help='verbose or not')
  args = parser.parse_args()

  # Get cpu or gpu device for training.
  args.device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {args.device} device")

  # create datasets
  if args.dataset == "syn":
    if args.causal_dir_syn == "causal":
      print("Sampling from causal synthetic datasets")
      env = CausalAdditiveNoSpurious()
    if args.causal_dir_syn == "anti":
      print("Sampling from causal anti datasets")
      env = AntiCausal()

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

  # model of phi (used by all models)
  input_dim = env.input_dim
  phi_odim = args.phi_odim

  Phi = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, phi_odim)
        )

  # loss fn
  criterion = torch.nn.MSELoss(reduction='mean')

  if args.model_name == "hsic" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi).to(args.device)
    trainer = HSIC(model, criterion, args)
    
    print("hsic training...")
    trainer.train(train_dataset, args.batch_size)

    print("hsic test...")
    trainer.test(test_dataset)

  if args.model_name == "irm" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi).to(args.device)
    trainer = IRM(model, criterion, args)
    
    print("irm training...")
    trainer.train(train_dataset, args.batch_size)

    print("irm test...")
    trainer.test(test_dataset)

  if args.model_name == "erm" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi).to(args.device)
    trainer = ERM(model, criterion, args)
    
    print("erm training...")
    trainer.train(train_dataset, args.batch_size)

    print("erm test...")
    trainer.test(test_dataset)

  if args.model_name == "maml" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi).to(args.device)
    trainer = LinearMAML(model, criterion, args)

    print("maml training...")
    trainer.train(train_dataset, args.batch_size)

    print("maml test...")
    trainer.test(test_dataset)

    if True:
      # Finetuning tests
      finetuned_loss = 0.0
      for i in range(8):
        x, y = test_finetune_dataset

        partical_test_finetune_dataset = (x[i:i+1,:], y[i:i+1])

        model = trainer.finetune_test(partical_test_finetune_dataset)
        finetuned_loss+=trainer.test(test_dataset, input_model = model, print_flag=False)

      print(finetuned_loss/8)

  if args.model_name == "adp_invar_anti_causal" or args.compare_all_invariant_models:
    model = AdaptiveInvariantNN(args.n_envs, input_dim, Phi).to(args.device)
    trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args, causal_dir = False)

    print("adp_invar anti-causal training...")
    trainer.train(train_dataset, args.batch_size)

    print("adp_invar anti-causal test...")
    trainer.test(test_dataset)

    if True:
      # Finetuning tests
      finetuned_loss = 0.0
      for i in range(8):
        x, y = test_finetune_dataset

        partical_test_finetune_dataset = (x[i:i+1,:], y[i:i+1])

        model = trainer.finetune_test(partical_test_finetune_dataset, test_unlabelled_dataset)
        finetuned_loss+=trainer.test(test_dataset, print_flag=False)

      print(finetuned_loss/8)

  if args.model_name == "adp_invar" or args.compare_all_invariant_models:
    model = AdaptiveInvariantNN(args.n_envs, input_dim, Phi).to(args.device)
    trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args)
    
    if args.print_base_graph:
      # check if the base classifer match before training
      sampe_n = 100
      x_base_test,y_base_test = env.sample_envs(env.num_train_evns + 1, n = sampe_n)
      x_base_test_sorted = np.sort(x_base_test, axis=0)
      y_base = env.sample_base_classifer(x_base_test_sorted)
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.savefig("comparision_before.png")

    print("adp_invar training...")
    trainer.train(train_dataset, args.batch_size)

    print("adp_invar test...")
    trainer.test(test_dataset)

    if True:
      # Finetuning tests
      proj_gd_loss = 0.0
      gd_loss = 0.0
      for i in range(8):
        x, y = test_finetune_dataset

        partical_test_finetune_dataset = (x[i:i+1,:], y[i:i+1])

        # print("prjected gradient descent")
        trainer.finetune_test(partical_test_finetune_dataset, test_unlabelled_dataset)
        proj_gd_loss+=trainer.test(test_dataset, print_flag=False)

        # print("regular gradient descent")
        trainer.finetune_test(partical_test_finetune_dataset, test_unlabelled_dataset, projected_gd=False)
        gd_loss+=trainer.test(test_dataset, print_flag=False)

      print(proj_gd_loss/8, gd_loss/8)


    if args.print_base_graph: 
      # check if the base classifer match after training
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.savefig("comparision_after.png")
