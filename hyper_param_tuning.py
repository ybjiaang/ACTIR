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

  args = parser.parse_args()
  log_directory = "./log_hyper_parameter"
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)

  log_directory += "/log_hyper_parameter_" + str(args.dataset) + "_" + args.model_name + '_' +str(args.test_index)
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)

  filename = log_directory + "/" + "test_"+ str(args.val_index)  +".csv"
  if os.path.exists(filename):
      pass
      #os.remove(filename)
  
  if args.model_name == "adp_invar_anti_causal":
    for reg_lambda in [10, 1, 0.1, 5, 100, 20]:
        for reg_lambda_2 in [0.1, 1, 5, 10, 0.01, 20, 100]:
            for gamma in [0.1, 0.5, 0.9]:
                for n_loop in [20, 30, 50]:
                    for lr in [1e-2, 1e-3, 1e-4]:
                        # synthetical anti-causal
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)

                        # synthetical anti-causal classification
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)
                        
                        # synthetical anti-causal multi-class classification
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti-multi --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop)
                        # run_cmd(cmd)
                        
                        if args.dataset == "vlcs":
                            cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=vlcs --data_dir="./vlcs/VLCS" --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning --test_index {:} --val_index {:}'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop, args.test_index, args.val_index)
                            run_cmd(cmd)

                        if args.dataset == "pacs":
                            n_loop = 25
                            cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=pacs --data_dir="./pacs/PACS" --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning --test_index {:} --val_index {:}'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop, args.test_index, args.val_index)
                            run_cmd(cmd)
                        
                        #cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=vlcs --data_dir="./pacs/PACS" --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop)
                        #run_cmd(cmd)
                        
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti-multi --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)
                        
                        # synthetic causal
                        # cmd = 'python main.py --model_name=adp_invar --causal_dir_syn=causal --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)
                        
                        # cmd = 'python main.py --model_name=adp_invar --dataset=bike --bike_test_season=1 --bike_year=0 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)
                        
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=color_mnist --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)
                        
                        if args.dataset == "color_mnist":
                            cmd = 'python main.py --classification --lr={:} --n_outer_loop={:} --model_name=adp_invar_anti_causal --dataset=color_mnist --phi_odim=8 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(lr, n_loop, reg_lambda, reg_lambda_2, filename, gamma)
                            run_cmd(cmd)
                        
                        # cmd = 'python main.py --model_name=adp_invar_anti_causal  --dataset=camelyon17 --classification --n_outer_loop 25 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)
                        
                        # cmd = 'python -u main.py --data_dir="/scratch/midway2/yiboj/data" --model_name=adp_invar_anti_causal  --dataset=camelyon17 --classification --n_outer_loop 25 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --run_fine_tune_test --n_fine_tune_tests 3 --n_fine_tune_points 1 10 50 100 1000 5000'.format(reg_lambda, reg_lambda_2, filename, gamma)
                        # run_cmd(cmd)
                        # pass

  if args.model_name == "irm":
    n_loop = 50
    # for reg_lambda in [0.1, 1, 5, 10, 0.01, 20, 100, 150, 200]:
    for reg_lambda in np.logspace(-2, 4, num=30):
        if args.dataset == "vlcs":
            cmd = 'python main.py --model_name=irm --dataset=vlcs --data_dir="./vlcs/VLCS" --classification --irm_reg_lambda={:} --cvs_dir={:} --n_outer_loop={:} --test_index {:} --val_index {:} --hyper_param_tuning'.format(reg_lambda, filename, n_loop, args.test_index, args.val_index)
            run_cmd(cmd)
        if args.dataset == "pacs":
            cmd = 'python main.py --model_name=irm --dataset=pacs --data_dir="./pacs/PACS" --classification --irm_reg_lambda={:} --cvs_dir={:} --n_outer_loop={:} --test_index {:} --val_index {:} --hyper_param_tuning'.format(reg_lambda, filename, n_loop, args.test_index, args.val_index)
            run_cmd(cmd)
        if args.dataset == "color_mnist":
            cmd = 'python main.py --classification --lr 1e-3 --n_outer_loop={:} --model_name=irm --dataset=color_mnist --phi_odim=8 --irm_reg_lambda={:} --cvs_dir={:} --hyper_param_tuning'.format(n_loop, reg_lambda, filename)
            run_cmd(cmd)
                        
        #             cmd = 'python main.py --model_name=irm  --dataset=camelyon17 --classification --n_outer_loop 20 --irm_reg_lambda={:} --cvs_dir={:} --hyper_param_tuning'.format(reg_lambda, filename)
        #             run_cmd(cmd)
        # cmd = 'python main.py --model_name=irm --causal_dir_syn=anti-multi --classification --irm_reg_lambda={:} --cvs_dir={:} --hyper_param_tuning'.format(reg_lambda, filename)
        # run_cmd(cmd)
