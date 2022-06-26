# Code for "Invariant and Transportable Representations for Anti-Causal Domain Shifts"

## Synthetic Dataset

`python syn_running_script.py`

Results will be saved as csv files in  folder `log_syn_classification`. You can use `csv_reading_script.py` to get summary statistics by changing `dir`.

To get disentanglement plot `disentangle_syn.png`, run

`python main.py --model_name=adp_invar_anti_causal --lr 1e-2 --classification --causal_dir_syn=anti --disentagnle_plot`

## Synthetic Dataset -- Counterexample

`python syn_running_script.py --dataset=counter-example`

Results will be saved as csv files in  folder `log_syn_classification`. You can use `csv_reading_script.py` to get summary statistics by changing `dir`.

## Color MNIST

`python color_mnist_running_script.py`

Results will be saved as csv files in  folder `log_color_mnist`. You can use `csv_reading_script.py` to get summary statistics by changing `dir`.

To get disentanglement plot `disentangle_color_mnist.png`, run

`python main.py --model_name=adp_invar_anti_causal --classification --dataset=color_mnist --disentagnle_plot --lr=0.001 --n_outer_loop=15 --fine_tune_lr 1e-2 --reg_lambda=2.1544346900318834 --reg_lambda_2=46.41588833612777 --phi_odim 8`

## Camlyon17 

`bash run_cam.sh`

Models and tensors will be saved in `./saved_model` and one can run `run_adaptive.sh` to run adaptive tests. For Camlyon17, please run it with mutiple random seeds as suggested by WILDS. We can change random seed by change `RANDOMSEED` in the `run_cam.sh` file.

## PACS
Use `dataset/DomianBed/download.py --dataset=pacs` to download dataset and change the data directory in `run_pacs.sh`.

`bash run_pacs.sh`

Models and tensors will be saved in `./saved_model` and one can run `run_adaptive.sh` to run adaptive tests. You can run it with different random seed

## VLCS

Use `dataset/DomianBed/download.py --dataset=vlcs` to download dataset and change the data directory in `run_vlcs.sh`.

`bash run_vlcs.sh`

Models and tensors will be saved in `./saved_model` and one can run `run_adaptive.sh` to run adaptive tests. You can run it with different random seed

## Hyper-Parameter
See `hyper_param_tuning.py` for details on running hyper parameter search. Feel free to run your own hyperparameter search.

## Others
Set `--nb_workers=0` if you see this warning `Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend`