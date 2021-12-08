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

log_directory = "./log_syn_causal"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


filename = log_directory + "/" + "test_causal"+ ".csv"
if os.path.exists(filename):
    os.remove(filename)

with open(filename, 'a', newline='') as file: 
    writer = csv.writer(file)
    writer.writerow(["HSIC", "IRM", "ERM", "MAML", "Anti-Causal", "Causal Base", "Causal"])

for _ in range(10):
    cmd = 'python main.py --compare_all_invariant_models --causal_dir_syn=causal --cvs_dir={:}'.format(filename)
    run_cmd(cmd)
