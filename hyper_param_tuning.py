from pyexpat import model
import numpy as np
import subprocess
import logging
import itertools
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
from tqdm import tqdm
import csv
import os
import argparse

def run_cmd(cmd):
    logging.info("Running command: {:}".format(cmd))
    subprocess.check_call(cmd,shell=True)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  # different models
  parser.add_argument('--model_name', type=str, default= "adp_invar_anti_causal", help='type of modesl. current support: adp_invar_anti_causal, irm')
  # dataset
  parser.add_argument('--dataset', type=str, default= "vlcs", help='type of experiment: syn, bike, color_mnist, vlcs, camelyon17')
  
  # domainbed specifics 
  parser.add_argument('--test_index', type=int, default= 3, help='which dataset to test')
  parser.add_argument('--val_index', type=int, default= 1, help='which dataset to val, it has to be strictly positive')
  parser.add_argument('--reg_lambda_list', nargs='+', type=float, help='reg_lambda to test')
  parser.add_argument('--reg_lambda_2_list', nargs='+', type=float, help='reg_lambda_2 to tests')
  parser.add_argument('--irm_reg_lambda_list', nargs='+', type=float, help='irm_reg_lambda to tests')

  args = parser.parse_args()
  log_directory = "./log_hyper_parameter"
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)

  log_directory += "/log_hyper_parameter_" + str(args.dataset) + "_" + args.model_name + '_' +str(args.test_index)
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)

  filename = log_directory + "/" + "test_"+ str(args.val_index)  +".csv"
  if os.path.exists(filename):
      os.remove(filename)

  if not args.reg_lambda_list:
      args.reg_lambda_list = [10, 1, 0.1]
    # args.reg_lambda_list = [1000, 500, 100, 20, 10, 1]

  if not args.reg_lambda_2_list:
      args.reg_lambda_2_list = [0.1, 1, 5, 10, 0.01]
    # args.reg_lambda_2_list = [1, 10, 100, 1000]

  if args.model_name == "adp_invar_anti_causal":
    # for reg_lambda in args.reg_lambda_list:
    for reg_lambda in np.logspace(-1, 3, num=20):
    # for reg_lambda in [1, 2, 3, 4, 5, 6, 7, 8]:
        # for reg_lambda_2 in args.reg_lambda_2_list:
        for reg_lambda_2 in np.logspace(-1, 3, num=20):
        # for reg_lambda_2 in [1.5]:
            for gamma in [0.9]:
                for n_loop in [100]:
                    for lr in [1e-2]:
                        # synthetical anti-causal
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)

                        # synthetical anti-causal classification
                        if args.dataset == "anti":
                          cmd = 'python main.py --lr={:} --n_outer_loop={:} --model_name=adp_invar_anti_causal --causal_dir_syn=anti --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(lr, n_loop, reg_lambda, reg_lambda_2, filename, gamma)
                          run_cmd(cmd)
                        
                        # synthetical anti-causal multi-class classification
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti-multi --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop)
                        # run_cmd(cmd)
                        
                        if args.dataset == "vlcs":
                            n_loop = 50
                            cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=vlcs --data_dir="./vlcs/VLCS" --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning --test_index {:} --val_index {:}'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop, args.test_index, args.val_index)
                            run_cmd(cmd)

                        if args.dataset == "pacs":
                            n_loop = 25
                            cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=pacs --data_dir="./pacs/PACS" --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning --test_index {:} --val_index {:}'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop, args.test_index, args.val_index)
                            run_cmd(cmd)
                        
                        if args.dataset == "color_mnist":
                            cmd = 'python main.py --classification --lr={:} --n_outer_loop={:} --model_name=adp_invar_anti_causal --dataset=color_mnist --phi_odim=8 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(lr, n_loop, reg_lambda, reg_lambda_2, filename, gamma)
                            run_cmd(cmd)
                        
                        if args.dataset == "camelyon17":
                            cmd = 'python -u main.py --data_dir="./Camelyon17" --model_name=adp_invar_anti_causal  --dataset=camelyon17 --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --lr={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma, lr, n_loop)
                            run_cmd(cmd)

  if not args.irm_reg_lambda_list:
    args.irm_reg_lambda_list = [0.1, 1, 5, 10, 0.01, 20, 100, 150, 200]
  if args.model_name == "irm":
    n_loop = 50
    # for reg_lambda in args.irm_reg_lambda_list:
    for reg_lambda in np.logspace(-2, 4, num=30):
        if args.dataset == "anti":
          cmd = 'python main.py --lr=1e-2 --n_outer_loop=100 --model_name=irm --causal_dir_syn=anti --classification --irm_reg_lambda={:} --cvs_dir={:} --hyper_param_tuning'.format(reg_lambda, filename)
          run_cmd(cmd)              
        if args.dataset == "vlcs":
            cmd = 'python main.py --model_name=irm --dataset=vlcs --data_dir="./vlcs/VLCS" --classification --irm_reg_lambda={:} --cvs_dir={:} --n_outer_loop={:} --test_index {:} --val_index {:} --hyper_param_tuning'.format(reg_lambda, filename, n_loop, args.test_index, args.val_index)
            run_cmd(cmd)
        if args.dataset == "pacs":
            cmd = 'python main.py --model_name=irm --dataset=pacs --data_dir="./pacs/PACS" --classification --irm_reg_lambda={:} --cvs_dir={:} --n_outer_loop={:} --test_index {:} --val_index {:} --hyper_param_tuning'.format(reg_lambda, filename, n_loop, args.test_index, args.val_index)
            run_cmd(cmd)
        if args.dataset == "color_mnist":
            cmd = 'python main.py --classification --lr 1e-2 --n_outer_loop={:} --model_name=irm --dataset=color_mnist --phi_odim=8 --irm_reg_lambda={:} --cvs_dir={:} --hyper_param_tuning'.format(n_loop, reg_lambda, filename)
            run_cmd(cmd)
        if args.dataset == "camelyon17":
            n_loop = 20
            cmd = 'python -u main.py --data_dir="./Camelyon17" --model_name=irm  --dataset=camelyon17 --classification  --irm_reg_lambda={:} --cvs_dir={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, filename, n_loop)
            run_cmd(cmd)
                        
