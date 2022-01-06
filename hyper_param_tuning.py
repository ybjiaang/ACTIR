import numpy as np
import subprocess
import logging
import itertools
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
from tqdm import tqdm
import csv
import os

log_directory = "./log_hyper_parameter"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


filename = log_directory + "/" + "test"+ ".csv"
if os.path.exists(filename):
    os.remove(filename)

def run_cmd(cmd):
	logging.info("Running command: {:}".format(cmd))
	subprocess.check_call(cmd,shell=True)

# 
    # for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
reg_lambda = 0.1
gamma = 0.9
for reg_lambda in np.logspace(-2, 4, num=30):
    for reg_lambda_2 in np.logspace(-2, 4, num=30):
        for gamma in [0.7, 0.9]:
            # synthetical anti-causal
            # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
            # run_cmd(cmd)

            # synthetical anti-causal classification
            # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
            # run_cmd(cmd)

            # # synthetic causal
            # cmd = 'python main.py --model_name=adp_invar --causal_dir_syn=causal --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
            # run_cmd(cmd)

            cmd = 'python main.py --model_name=adp_invar  --dataset=bike --bike_test_season=1 --bike_year=0 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
            run_cmd(cmd)