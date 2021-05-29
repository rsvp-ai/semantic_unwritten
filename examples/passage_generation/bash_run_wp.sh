#!/usr/bin/env bash
PREFIX=""
DATA=dataset/writingprompt
PARA=$1
SEP=$2
MODEL="gpt2"
WEOS=False
TRAIN=False
EVAL=False
TEST=True
GENERATE=False
OUTPUT=${PREFIX}/output/wp_filter/$1_$2
if [ ${SEP} == "True" ]; then
OUTPUT=${OUTPUT}_sep
fi
#OUTPUT=${PREFIX}/checkpoints/gpt2

export CUDA_VISIBLE_DEVICES=$3
export PYTHONPATH="${PREFIX}/project/transformers"
if [ ${TRAIN} == "True" ]; then
python -u run_wp_finetuning.py \
--paragraph_eos ${PARA} \
--sep ${SEP} \
--output_dir ${OUTPUT} \
--train_tgt_data_file ${PREFIX}/${DATA}/train.wp_target_filtered \
--train_src_data_file ${PREFIX}/${DATA}/train.wp_source_filtered \
--eval_tgt_data_file ${PREFIX}/${DATA}/eval.wp_target_filtered \
--eval_src_data_file ${PREFIX}/${DATA}/eval.wp_source_filtered \
--model_name_or_path ${PREFIX}/checkpoints/gpt2 \
--overwrite_output_dir \
--overwrite_cache \
--model_type ${MODEL} \
--block_size 1024 \
--without_eos ${WEOS} \
--per_gpu_train_batch_size 4 \
--num_train_epochs 8 \
--logging_steps 500 \
--warmup_steps 800 \
--save_steps 16000 \
--do_train \
--fp16
#--model_name_or_path ${PREFIX}/checkpoints/${MODEL} \
echo "===========Train Finished============"
fi &&

if [ ${EVAL} == "True" ]; then
python -u run_wp_finetuning.py \
--paragraph_eos ${PARA} \
--output_dir ${OUTPUT} \
--sep ${SEP} \
--train_tgt_data_file ${PREFIX}/${DATA}/train.wp_target_filtered \
--eval_tgt_data_file ${PREFIX}/${DATA}/eval.wp_target_filtered \
--train_src_data_file ${PREFIX}/${DATA}/train.wp_source_filtered \
--eval_src_data_file ${PREFIX}/${DATA}/eval.wp_source_filtered \
--model_name_or_path ${PREFIX}/checkpoints/gpt2 \
--model_type ${MODEL} \
--block_size 1024 \
--without_eos ${WEOS} \
--per_gpu_eval_batch_size 6 \
--do_eval \
--fp16 \
--eval_all_checkpoints &&

echo "===========EVAL Finished============"
fi &&

#DECODE_CKPT_PATH=`cat ${OUTPUT}/best_step.txt`
DECODE_CKPT_PATH=${OUTPUT}
DECODE_CKPT_PATH_0=${OUTPUT}/checkpoint-0

if [ ${TEST} == "True" ]; then
python -u run_wp_finetuning.py \
--paragraph_eos ${PARA} \
--sep ${SEP} \
--output_dir ${DECODE_CKPT_PATH} \
--train_tgt_data_file ${PREFIX}/${DATA}/train.wp_target_filtered \
--train_src_data_file ${PREFIX}/${DATA}/train.wp_source_filtered \
--eval_tgt_data_file ${PREFIX}/${DATA}/test.wp_target_filtered \
--eval_src_data_file ${PREFIX}/${DATA}/test.wp_source_filtered \
--model_name_or_path ${PREFIX}/checkpoints/gpt2 \
--model_type ${MODEL} \
--block_size 1024 \
--without_eos ${WEOS} \
--per_gpu_eval_batch_size 6 \
--do_eval \
--fp16 &&

#python -u run_wp_finetuning.py \
#--paragraph_eos ${PARA} \
#--sep ${SEP} \
#--training_percent ${PERCENT} \
#--output_dir ${DECODE_CKPT_PATH_0} \
#--train_tgt_data_file ${PREFIX}/${DATA}/train.wp_target_filtered \
#--train_src_data_file ${PREFIX}/${DATA}/train.wp_source_filtered \
#--eval_tgt_data_file ${PREFIX}/${DATA}/test.wp_target_filtered \
#--eval_src_data_file ${PREFIX}/${DATA}/test.wp_source_filtered \
#--model_name_or_path ${PREFIX}/checkpoints/gpt2 \
#--model_type ${MODEL} \
#--without_eos ${WEOS} \
#--block_size 1024 \
#--per_gpu_eval_batch_size 6 \
#--do_eval \
#--fp16 &&

echo "===========Test Finished============"
fi &&

if [ ${GENERATE} == "True" ]; then
export CUDA_VISIBLE_DEVICES=$5
python -u run_generation.py \
--input_format wp \
--language en \
--paragraph_eos ${PARA} \
--sep ${SEP} \
--input_path ${PREFIX}/${DATA}/test_partA.txt \
--groundT_path ${PREFIX}/${DATA}/test_partB.txt \
--model_name_or_path ${DECODE_CKPT_PATH} \
--model_type ${MODEL} \
--length 1024 \
--stop_token "<|endoftext|>" \
--p 0.95 &&

python -u run_evaluation.py \
--language en \
--model_type ${MODEL} \
--input_path ${PREFIX}/${DATA}/test_partA.txt \
--groundT_path ${PREFIX}/${DATA}/test_partB.txt \
--generated_path  ${DECODE_CKPT_PATH}/test_generated.txt \
--model_name_or_path ${DECODE_CKPT_PATH}
fi