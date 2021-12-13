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

log_directory = "./log_syn"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


filename = log_directory + "/" + "test_causal"+ ".csv"
if os.path.exists(filename):
    os.remove(filename)

n_fine_tune_points = [3]
with open(filename, 'a', newline='') as file: 
    writer = csv.writer(file)
    colname = ["HSIC", "IRM", "ERM", "MAML Train", "MAML", "Anti-Causal", "Causal Base", "Causal"]
    for point in n_fine_tune_points:
        colname.append("MAML " + str(point))
        colname.append("Anti Causal " + str(point))
        colname.append("Causal (projected) " + str(point))
        colname.append("Causal " + str(point))
    writer.writerow(colname)

for _ in range(1):
    cmd = 'python main.py --compare_all_invariant_models --causal_dir_syn=causal --cvs_dir={:} --run_fine_tune_test --n_fine_tune_tests 100 --n_fine_tune_points'.format(filename)
    for point in n_fine_tune_points:
        cmd += " " + str(point)
    run_cmd(cmd)
