#!/bin/bash

RANDOM_SEED=0
N_LOOPS=50
RESNETDIM=8

TEST_IND=0
CAUSAL_REG=10
CAUSAL_REG_2=1
IRM_REG=0.1
python3 -u main.py --model_name=erm --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=adp_invar_anti_causal --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=irm --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=maml --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}


TEST_IND=1
CAUSAL_REG=0.1
CAUSAL_REG_2=0.1
IRM_REG=0.01
python3 -u main.py --model_name=erm --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=adp_invar_anti_causal --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=irm --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=maml --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}

TEST_IND=2
CAUSAL_REG=10
CAUSAL_REG_2=1
IRM_REG=0.1
python3 -u main.py --model_name=erm --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=adp_invar_anti_causal --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=irm --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=maml --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}


TEST_IND=3
CAUSAL_REG=10
CAUSAL_REG_2=1
IRM_REG=5
python3 -u main.py --model_name=erm --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=adp_invar_anti_causal --random_seed ${RANDOM_SEED} --dataset=vlcs --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=irm --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM}
python3 -u main.py --model_name=maml --dataset=vlcs --random_seed ${RANDOM_SEED} --classification --data_dir="./dataset/VLCS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM}