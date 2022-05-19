# Code for Invariant and Transportable Representations for Anti-Causal Domain Shifts

## Synthetic Dataset

`python syn_running_script.py`

Results will be saved as csv files in  folder `log_syn_classification`. You can use `csv_reading_script.py` to get summary statistics. 


## Color MNIST

`python color_mnist_running_script.py`

Results will be saved as csv files in  folder `log_color_mnist`. You can use `csv_reading_script.py` to get summary statistics. 

## Camlyon17 

`bash run_cam.sh`

Models and tensors will be saved in `./saved_model` and one can run `run_adaptive.sh` to run adaptive tests. For Camlyon17, please run it with mutiple random seeds as suggested by WILDS. 

## PACS
Use `dataset/DomianBed/download.py` to download dataset and change the data directory in `run_pacs.sh`.

`bash run_pacs.sh`

Models and tensors will be saved in `./saved_model` and one can run `run_adaptive.sh` to run adaptive tests. You can run it with different random seed

## VLCS

Use `dataset/DomianBed/download.py` to download dataset and change the data directory in `run_vlcs.sh`.

`bash run_vlcs.sh`

Models and tensors will be saved in `./saved_model` and one can run `run_adaptive.sh` to run adaptive tests. You can run it with different random seed