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

log_directory = "./log_erm_color_mnist"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


filename = log_directory + "/" + "test_causal"+ ".csv"
if os.path.exists(filename):
    os.remove(filename)

n_fine_tune_points = [1, 5, 10, 20, 30, 40, 50, 100]
with open(filename, 'a', newline='') as file: 
    writer = csv.writer(file)
    colname = ["HSIC", "IRM", "ERM", "MAML Train", "MAML", "Anti-Causal"]
    for point in n_fine_tune_points:
        colname.append("ERM " + str(point))
        colname.append("MAML " + str(point))
        colname.append("Anti Causal " + str(point))
    writer.writerow(colname)

# 774.2636826811278,4.641588833612779
for _ in range(20):
    # cmd = ' python main.py --compare_all_invariant_models --classification --dataset=color_mnist --lr 1e-2 --fine_tune_lr 1e-4 --reg_lambda 20 --reg_lambda_2 100 --phi_odim 8 --irm_reg_lambda 1487.3521072935118 --run_fine_tune_test --n_fine_tune_tests 100 --n_fine_tune_points'.format(filename)
    cmd = ' python main.py --compare_all_invariant_models --classification --dataset=color_mnist --lr 1e-2 --fine_tune_lr 1e-4 --reg_lambda 10 --reg_lambda_2 0.1 --phi_odim 8 --irm_reg_lambda 1487.3521072935118 --run_fine_tune_test --n_fine_tune_tests 10 --n_fine_tune_points'.format(filename)
    for point in n_fine_tune_points:
        cmd += " " + str(point)
    run_cmd(cmd)
