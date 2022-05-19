#!/bin/bash

TEST_INDS=(1)
FINE_TUNE_LRS=(1e-2)
N_FINETUNE_LOOPS=(20)
SEED=(0 1 2 3 4)

for N_FINETUNE_LOOP in "${N_FINETUNE_LOOPS[@]}"
do

for FINE_TUNE_LR in "${FINE_TUNE_LRS[@]}"
do

for TEST_IND in "${TEST_INDS[@]}"
do

for RANDOM_SEED in "${SEED[@]}"
do

echo $TEST_IND, $FINE_TUNE_LR, $N_FINETUNE_LOOP, $RANDOM_SEED
export TEST_IND FINE_TUNE_LR N_FINETUNE_LOOP RANDOM_SEED

python3 -u main.py --compare_all_invariant_models --dataset=pacs --classification --data_dir="./dataset/PACS" \
--run_fine_tune_test_standalone --n_fine_tune_points 0 10 20 40 60 80 100  --n_fine_tune_tests 100 --n_finetune_loop ${N_FINETUNE_LOOP} \
--fine_tune_lr ${FINE_TUNE_LR} --test_index ${TEST_IND} \
--model_save_dir="./saved_model/" --nb_workers 0 --resnet_dim 8 --random_seed ${RANDOM_SEED}

python3 -u main.py --compare_all_invariant_models --dataset=vlcs --classification --data_dir="./dataset/VLCS" \
--run_fine_tune_test_standalone --n_fine_tune_points 0 10 20 40 60 80 100  --n_fine_tune_tests 100 --n_finetune_loop ${N_FINETUNE_LOOP} \
--fine_tune_lr ${FINE_TUNE_LR} --test_index ${TEST_IND} \
--model_save_dir="./saved_model/" --nb_workers 0 --resnet_dim 8 --random_seed ${RANDOM_SEED}

python3 -u main.py --compare_all_invariant_models --dataset=camelyon17 --classification --data_dir="./data" \
--run_fine_tune_test_standalone --n_fine_tune_points 0 50 100 150 200 250 300 --n_fine_tune_tests 100 --n_finetune_loop ${N_FINETUNE_LOOP} \
--fine_tune_lr ${FINE_TUNE_LR} --test_index ${TEST_IND} \
--model_save_dir="./saved_model/" --nb_workers 0 --resnet_dim 128 --random_seed ${RANDOM_SEED}

done
done
done
done