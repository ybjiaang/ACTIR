import pandas as pd
import glob, os

dir = "./log_base_year_1"
os.chdir(dir)
for file in glob.glob("*.csv"):
    print(file)
    data= pd.read_csv(file)

    print(data.mean(axis = 0))
    print(data.std(axis = 0))
