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

log_directory = "./log_base_year_1"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

bike_year = 0
for bike_season in range(4):
    filename = log_directory + "/" + "test_season_" + str(bike_season) + "_year_" + str(bike_year) + ".csv"
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'a', newline='') as file: 
      writer = csv.writer(file)
      writer.writerow(["HSIC", "IRM", "ERM", "MAML", "Anti-Causal", "Causal"])
    
    for _ in range(50):
        cmd = 'python main.py --compare_all_invariant_models --dataset=bike --bike_test_season={:} --bike_year=0 --cvs_dir={:}'.format(bike_season, filename)
        run_cmd(cmd)
