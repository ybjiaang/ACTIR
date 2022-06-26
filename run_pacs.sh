#!/bin/bash
N_LOOPS=25
RESNETDIM=8
RANDOMSEED=0

TEST_IND=0
CAUSAL_REG=0.1
CAUSAL_REG_2=5
IRM_REG=10
python3 -u main.py --model_name=erm --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --verbose --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=adp_invar_anti_causal --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=irm --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=maml --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}

TEST_IND=1
CAUSAL_REG=10
CAUSAL_REG_2=1
IRM_REG=0.1
python3 -u main.py --model_name=erm --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --verbose --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=adp_invar_anti_causal --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=irm --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=maml --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}

TEST_IND=2
CAUSAL_REG=0.1
CAUSAL_REG_2=10
IRM_REG=0.1
python3 -u main.py --model_name=erm --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --verbose --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=adp_invar_anti_causal --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=irm --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=maml --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}

TEST_IND=3
CAUSAL_REG=1
CAUSAL_REG_2=10
IRM_REG=10
python3 -u main.py --model_name=erm --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --verbose --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=adp_invar_anti_causal --balanced_dataset --dataset=pacs --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --reg_lambda ${CAUSAL_REG} --reg_lambda_2 ${CAUSAL_REG_2} --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=irm --dataset=pacs --balanced_dataset --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS} --save_test_phi --test_index ${TEST_IND} --irm_reg_lambda ${IRM_REG} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}
python3 -u main.py --model_name=maml --dataset=pacs --balanced_dataset --classification --data_dir="./dataset/PACS" --n_outer_loop ${N_LOOPS}  --save_test_phi --test_index ${TEST_IND} --resnet_dim ${RESNETDIM} --random_seed ${RANDOMSEED}