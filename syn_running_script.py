import numpy as np
import subprocess
import logging
import itertools
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
from tqdm import tqdm
import csv
import os

def run_cmd(cmd):
	logging.info("Running command: {:}".format(cmd))
	subprocess.check_call(cmd,shell=True)

log_directory = "./log_syn_classification"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


filename = log_directory + "/" + "test_causal"+ ".csv"
if os.path.exists(filename):
    os.remove(filename)

n_fine_tune_points = [1, 5, 10]
with open(filename, 'a', newline='') as file: 
    writer = csv.writer(file)
    colname = ["IRM", "ERM", "MAML Train", "MAML", "Anti-Causal"]
    for point in n_fine_tune_points:
        colname.append("IRM " + str(point))
        colname.append("ERM " + str(point))
        colname.append("MAML " + str(point))
        colname.append("Anti Causal " + str(point))
    writer.writerow(colname)

for _ in range(100):
    # cmd = 'python main.py --compare_all_invariant_models --lr 1e-2 --fine_tune_lr 1e-2 --classification --causal_dir_syn=anti --cvs_dir={:} --run_fine_tune_test --n_fine_tune_tests 100 --random_seed -1 --n_fine_tune_points'.format(filename)
    cmd = 'python main.py --compare_all_invariant_models --lr 1e-2 --fine_tune_lr 1e-2 --classification --causal_dir_syn=causal_descent --reg_lambda=4.520353656360243 --reg_lambda_2=9.23670857187386 --irm_reg_lambda=137.3823795883264 --run_fine_tune_test --n_fine_tune_tests 100 --random_seed -1 --cvs_dir={:} --n_fine_tune_points'.format(filename)
    for point in n_fine_tune_points:
        cmd += " " + str(point)
    run_cmd(cmd)
