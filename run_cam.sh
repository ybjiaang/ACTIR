#!/bin/bash
#

CAUSAL_REG=10
CAUSAL_REG_2=0.01
IRM_REG=10
RANDOMSEED=0
RESNETDIM=128
python3 -u main.py --model_name=erm --dataset=camelyon17 --classification --data_dir="./Camelyon17" --n_outer_loop 25  --random_seed ${RANDOMSEED} --resnet_dim ${RESNETDIM} --save_test_phi
python3 -u main.py --model_name=adp_invar_anti_causal --dataset=camelyon17 --classification --data_dir="./Camelyon17" --n_outer_loop 25 --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --random_seed ${RANDOMSEED} --resnet_dim ${RESNETDIM} --save_test_phi
python3 -u main.py --model_name=irm --dataset=camelyon17 --classification --data_dir="./Camelyon17" --n_outer_loop 25 --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --random_seed ${RANDOMSEED} --resnet_dim ${RESNETDIM} --save_test_phi
python3 -u main.py --model_name=maml --dataset=camelyon17 --classification --data_dir="./Camelyon17" --n_outer_loop 25  --test_index ${TEST_IND} --random_seed ${RANDOMSEED} --resnet_dim ${RESNETDIM} --save_test_phi

