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
import csv

import matplotlib.pyplot as plt

from dataset.syn_env import CausalAdditiveNoSpurious, AntiCausal, CausalControlDataset, AntiCausalControlDataset
from dataset.bike_env import BikeSharingDataset
from models.adap_invar import AdaptiveInvariantNN
from models.base_classifer import BaseClass
from trainers.adap_invar_trainer import AdaptiveInvariantNNTrainer
from trainers.adap_invar_trainer_meta import AdaptiveInvariantNNTrainerMeta
from trainers.erm import ERM
from trainers.irm import IRM
from trainers.hsic import HSIC
from trainers.maml import LinearMAML
from misc import fine_tunning_test, BaseLoss

if __name__ == '__main__':
  # torch.manual_seed(0)
  # random.seed(0)
  # np.random.seed(0)

  parser = argparse.ArgumentParser()

  parser.add_argument('--n_envs', type=int, default= 5, help='number of enviroments per training epoch')
  parser.add_argument('--batch_size', type=int, default= 128, help='batch size')
  parser.add_argument('--reg_lambda', type=float, default= 4000, help='regularization coeff for adaptive invariant learning')
  parser.add_argument('--reg_lambda_2', type=float, default= 1.2, help='second regularization coeff for adaptive invariant learning')
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
  parser.add_argument('--dataset', type=str, default= "syn", help='type of experiment: syn, bike')
  parser.add_argument('--causal_dir_syn', type=str, default= "anti", help='anti or causal')
  # synthetic dataset specifics
  parser.add_argument('--syn_dataset_train_size', type=int, default= 1024, help='size of synthetic dataset per env')
  # bike sharing specifics
  parser.add_argument('--bike_test_season', type=int, default= 1, help='what season to test our model')
  parser.add_argument('--bike_year', type=int, default= 0, help='what year to test our model')

  # misc
  parser.add_argument('--print_base_graph', action='store_true', help='whether to print base classifer comparision graph, can only be used in 1 dimension')
  parser.add_argument('--verbose', action='store_true', help='verbose or not')
  parser.add_argument('--cvs_dir', type=str, default= "./test.cvs", help='path to the cvs file')
  parser.add_argument('--hyper_param_tuning', action='store_true', help='whether to do hyper-parameter tuning')
  args = parser.parse_args()

  # Get cpu or gpu device for training.
  args.device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {args.device} device")

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

  if args.dataset == "bike":
    print("bikesharing dataset")
    env = BikeSharingDataset(test_season=args.bike_test_season, year=args.bike_year)

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
  else:
    out_dim = 1
    args.num_class = 1
    
  if args.dataset == "syn":
    Phi = nn.Sequential(
              nn.Linear(input_dim, 4),
              nn.ReLU(),
              nn.Linear(4, phi_odim)
          )

  if args.dataset == "bike":
    Phi = nn.Sequential(
              nn.Linear(input_dim, 8),
              nn.ReLU(),
              nn.Linear(8, 16),
              nn.ReLU(),
              nn.Linear(16, phi_odim)
          )
    
  """ HSIC """
  if args.model_name == "hsic" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi).to(args.device)
    trainer = HSIC(model, criterion, args)
    
    if args.print_base_graph:
      # check if the base classifer match before training
      sampe_n = 100
      x_base_test,y_base_test = env.sample_envs(env.num_train_evns + 1, n = sampe_n)
      ind = np.argsort(x_base_test[:,0], axis=0)
      y_base_test = y_base_test[ind]
      x_base_test_sorted = np.sort(x_base_test, axis=0)
      y_base = env.sample_base_classifer(x_base_test_sorted)
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/hsci_comparision_before.png")

    print("hsic training...")
    trainer.train(train_dataset, args.batch_size)

    print("hsic test...")
    hsic_loss = trainer.test(test_dataset)

    if args.print_base_graph: 
      # check if the base classifer match after training
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/hsic_comparision_after.png")

  """ IRM """
  if args.model_name == "irm" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi, out_dim).to(args.device)
    trainer = IRM(model, criterion, args)

    if args.print_base_graph:
      # check if the base classifer match before training
      sampe_n = 100
      x_base_test,y_base_test = env.sample_envs(env.num_train_evns + 1, n = sampe_n)
      ind = np.argsort(x_base_test[:,0], axis=0)
      y_base_test = y_base_test[ind]
      x_base_test_sorted = np.sort(x_base_test, axis=0)
      y_base = env.sample_base_classifer(x_base_test_sorted)
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/irm_comparision_before.png")
    
    print("irm training...")
    trainer.train(train_dataset, args.batch_size)

    print("irm test...")
    irm_loss = trainer.test(test_dataset)

    if args.print_base_graph: 
      # check if the base classifer match after training
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/irm_comparision_after.png")


  """ ERM """
  if args.model_name == "erm" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi, out_dim).to(args.device)
    trainer = ERM(model, criterion, args)
    
    if args.print_base_graph:
      # check if the base classifer match before training
      sampe_n = 100
      x_base_test,y_base_test = env.sample_envs(env.num_train_evns + 1, n = sampe_n)
      ind = np.argsort(x_base_test[:,0], axis=0)
      y_base_test = y_base_test[ind]
      x_base_test_sorted = np.sort(x_base_test, axis=0)
      y_base = env.sample_base_classifer(x_base_test_sorted)
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/erm_comparision_before.png")

    print("erm training...")
    trainer.train(train_dataset, args.batch_size)

    print("erm test...")
    erm_loss = trainer.test(test_dataset)

    if args.print_base_graph: 
      # check if the base classifer match after training
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/erm_comparision_after.png")


  """ MAML """
  if args.model_name == "maml" or args.compare_all_invariant_models:
    model = BaseClass(input_dim, Phi, out_dim).to(args.device)
    trainer = LinearMAML(model, criterion, args)

    if args.print_base_graph:
      # check if the base classifer match before training
      sampe_n = 100
      x_base_test,y_base_test = env.sample_envs(env.num_train_evns + 1, n = sampe_n)
      ind = np.argsort(x_base_test[:,0], axis=0)
      y_base_test = y_base_test[ind]
      x_base_test_sorted = np.sort(x_base_test, axis=0)
      y_base = env.sample_base_classifer(x_base_test_sorted)
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/maml_comparision_before.png")

    print("maml training...")
    maml_train_loss = trainer.train(train_dataset, args.batch_size)

    print("maml test...")
    maml_loss = trainer.test(test_dataset)

    if args.print_base_graph: 
      # check if the base classifer match after training
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/maml_comparision_after.png")

    if args.run_fine_tune_test:
      maml_finetune_loss = []
      for n_tune_points in  args.n_fine_tune_points:
        maml_finetune_loss.append(fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points))

  """ Adaptive Invariant Anti Causal """
  if args.model_name == "adp_invar_anti_causal" or args.compare_all_invariant_models:
    model = AdaptiveInvariantNN(args.n_envs, input_dim, Phi, args, out_dim).to(args.device)
    trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args, causal_dir = False)

    if args.print_base_graph:
      # check if the base classifer match before training
      sampe_n = 100
      x_base_test,y_base_test = env.sample_envs(env.num_train_evns + 1, n = sampe_n)
      ind = np.argsort(x_base_test[:,0], axis=0)
      y_base_test = y_base_test[ind]
      x_base_test_sorted = np.sort(x_base_test, axis=0)
      y_base = env.sample_base_classifer(x_base_test_sorted)
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/adp_invar_anti_causal_comparision_before.png")

    print("adp_invar anti-causal training...")
    trainer.train(train_dataset, args.batch_size)

    print("adp_invar anti-causal test...")
    adp_invar_anti_causal_base_loss, _ = trainer.test(test_dataset)

    if args.hyper_param_tuning:
      with open(args.cvs_dir, 'a', newline='') as file: 
        writer = csv.writer(file)
        row = [args.reg_lambda, args.reg_lambda_2, args.gamma, adp_invar_anti_causal_base_loss]
        writer.writerow(row)

    if args.print_base_graph: 
      # check if the base classifer match after training
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/adp_invar_anti_causal_comparision_after.png")

    if args.run_fine_tune_test:
      anti_causal_finetune_loss = []
      for n_tune_points in  args.n_fine_tune_points:
        anti_causal_finetune_loss.append(fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points, test_unlabelled_dataset))

  """ Adaptive Invariant Causal """
  if args.model_name == "adp_invar" or args.compare_all_invariant_models:
    model = AdaptiveInvariantNN(args.n_envs, input_dim, Phi, args, out_dim).to(args.device)
    trainer = AdaptiveInvariantNNTrainer(model, criterion, args.reg_lambda, args)
    
    if args.print_base_graph:
      # check if the base classifer match before training
      sampe_n = 100
      x_base_test,y_base_test = env.sample_envs(env.num_train_evns + 1, n = sampe_n)
      ind = np.argsort(x_base_test[:,0], axis=0)
      y_base_test = y_base_test[ind]
      x_base_test_sorted = np.sort(x_base_test, axis=0)
      y_base = env.sample_base_classifer(x_base_test_sorted)
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/adp_invar_comparision_before.png")

    print("adp_invar training...")
    trainer.train(train_dataset, args.batch_size)

    print("adp_invar test...")
    adp_invar_base_loss, adp_invar_loss = trainer.test(test_dataset)

    if args.hyper_param_tuning:
      with open(args.cvs_dir, 'a', newline='') as file: 
        writer = csv.writer(file)
        row = [args.reg_lambda, args.reg_lambda_2, args.gamma, adp_invar_base_loss]
        writer.writerow(row)

    if args.print_base_graph: 
      # check if the base classifer match after training
      with torch.no_grad(): 
        y_base_predicted = trainer.model.sample_base_classifer(x_base_test_sorted)
      plt.figure()
      plt.plot(x_base_test_sorted[:,0], y_base, label="true base classifer")
      plt.plot(x_base_test_sorted[:,0], y_base_test, label="true y")
      plt.plot(x_base_test_sorted[:,0], y_base_predicted.numpy(), label="estimated base classifer")
      plt.legend()
      plt.savefig("png_folder/adp_invar_comparision_after.png")

    if args.run_fine_tune_test:
      causal_proj_gd_losses = []
      causal_gd_losses = []
      for n_tune_points in  args.n_fine_tune_points:
        causal_gd_loss, causal_proj_gd_loss = fine_tunning_test(trainer, args, test_finetune_dataset, test_dataset, n_tune_points, test_unlabelled_dataset, True)
        causal_proj_gd_losses.append(causal_proj_gd_loss)
        causal_gd_losses.append(causal_gd_loss)

  if args.compare_all_invariant_models:
    with open(args.cvs_dir, 'a', newline='') as file: 
      writer = csv.writer(file)
      row = [hsic_loss, irm_loss, erm_loss, maml_train_loss, maml_loss, adp_invar_anti_causal_base_loss, adp_invar_base_loss, adp_invar_loss]
      if args.run_fine_tune_test:
        for i, n_tune_points in enumerate(args.n_fine_tune_points):
          row.append(maml_finetune_loss[i])
          row.append(anti_causal_finetune_loss[i])
          row.append(causal_proj_gd_losses[i])
          row.append(causal_gd_losses[i])
      writer.writerow(row)
    print(hsic_loss, irm_loss, erm_loss, maml_train_loss, maml_loss, adp_invar_anti_causal_base_loss, adp_invar_base_loss, adp_invar_loss)