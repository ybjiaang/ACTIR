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



# reg_lambda = 0.1
gamma = 0.9
#for reg_lambda in np.logspace(-1, 4, num=10):
#    for reg_lambda_2 in np.logspace(-1, 4, num=10):
for reg_lambda in [10]:
    for reg_lambda_2 in [5, 10, 20, 0.1, 1, 100, 30, 40, 50]:
        # for gamma in [0.7, 0.9]:
        for gamma in [0.9]:
            for n_loop in [50]:
                # synthetical anti-causal
                # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)

                # synthetical anti-causal classification
                # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)

                # synthetical anti-causal multi-class classification
                # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti-multi --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop)
                # run_cmd(cmd)
                cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=vlcs --data_dir="./vlcs/VLCS" --classification --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma, n_loop)
                run_cmd(cmd)

                # cmd = 'python main.py --model_name=adp_invar_anti_causal --causal_dir_syn=anti-multi --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)

                # # synthetic causal
                # cmd = 'python main.py --model_name=adp_invar --causal_dir_syn=causal --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)

                # cmd = 'python main.py --model_name=adp_invar --dataset=bike --bike_test_season=1 --bike_year=0 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)

                # cmd = 'python main.py --model_name=adp_invar_anti_causal --dataset=color_mnist --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)

                # cmd = 'python main.py --classification --n_outer_loop=50 --model_name=adp_invar_anti_causal --dataset=color_mnist --phi_odim=8 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)

                # cmd = 'python main.py --model_name=adp_invar_anti_causal  --dataset=camelyon17 --classification --n_outer_loop 25 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --hyper_param_tuning'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)
                # cmd = 'python -u main.py --data_dir="/scratch/midway2/yiboj/data" --model_name=adp_invar_anti_causal  --dataset=camelyon17 --classification --n_outer_loop 25 --reg_lambda={:} --reg_lambda_2={:} --cvs_dir={:} --gamma={:} --run_fine_tune_test --n_fine_tune_tests 3 --n_fine_tune_points 1 10 50 100 1000 5000'.format(reg_lambda, reg_lambda_2, filename, gamma)
                # run_cmd(cmd)
                # pass

#for reg_lambda in np.logspace(-1, 5, num=100): 
#n_loop = 50
#for reg_lambda in [0.1, 1, 10, 50, 100]:
#    cmd = 'python main.py --model_name=irm --dataset=vlcs --data_dir="/scratch/midway2/yiboj/data/VLCS" --classification --irm_reg_lambda={:} --cvs_dir={:} --n_outer_loop={:} --hyper_param_tuning'.format(reg_lambda, filename, n_loop)
#    run_cmd(cmd)
        #cmd = 'python main.py --model_name=irm  --dataset=camelyon17 --classification --n_outer_loop 20 --irm_reg_lambda={:} --cvs_dir={:} --hyper_param_tuning'.format(reg_lambda, filename)
        # run_cmd(cmd)
#     cmd = 'python main.py --model_name=irm --causal_dir_syn=anti-multi --classification --irm_reg_lambda={:} --cvs_dir={:} --hyper_param_tuning'.format(reg_lambda, filename)
#     run_cmd(cmd)
