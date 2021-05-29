#!/usr/bin/env bash
PREFIX=""
TRAIN=False
EVAL=False
TEST=True
GENERATE=False
INPUT=$1
OUTPUT=${PREFIX}/output/zh/$1
SEP=$2
if [ ${SEP} == "True" ]; then
OUTPUT=${OUTPUT}_sep
fi

export CUDA_VISIBLE_DEVICES=$3
export PYTHONPATH="${PREFIX}/project/transformers"
if [ ${TRAIN} == "True" ]; then
python -u run_lm_finetuning.py \
--input_format ${INPUT} \
--sep ${SEP} \
--output_dir ${OUTPUT} \
--train_data_file ${PREFIX}/dataset/Chinese_Eassy/processed_data/annotation/train.txt \
--eval_data_file ${PREFIX}/dataset/Chinese_Eassy/processed_data/annotation/valid.txt \
--model_name_or_path ${PREFIX}/output/gpt2_ml \
--model_type grover \
--block_size 1024 \
--per_gpu_train_batch_size 4 \
--num_train_epochs 3 \
--logging_steps 100 \
--warmup_steps 200 \
--save_steps 100 \
--overwrite_cache \
--overwrite_output_dir \
--do_train \
--checkpoint
fi &&

if [ ${EVAL} == "True" ]; then
python -u run_lm_finetuning.py \
--input_format ${INPUT} \
--output_dir ${OUTPUT} \
--sep ${SEP} \
--train_data_file ${PREFIX}/dataset/Chinese_Eassy/processed_data/annotation/train.txt \
--eval_data_file ${PREFIX}/dataset/Chinese_Eassy/processed_data/annotation/valid.txt \
--model_name_or_path ${PREFIX}/output/gpt2_ml \
--model_type grover \
--block_size 1024 \
--per_gpu_eval_batch_size 8 \
--num_train_epochs 3 \
--logging_steps 100 \
--warmup_steps 200 \
--save_steps 100 \
--overwrite_cache \
--overwrite_output_dir \
--do_eval \
--eval_all_checkpoints
fi &&

DECODE_CKPT=`cat ${OUTPUT}/best_step.txt` &&

if [ ${TEST} == "True" ]; then
python -u run_lm_finetuning.py \
--input_format ${INPUT} \
--output_dir ${OUTPUT}/checkpoint-${DECODE_CKPT} \
--sep ${SEP} \
--train_data_file ${PREFIX}/dataset/Chinese_Eassy/processed_data/annotation/train.txt \
--eval_data_file ${PREFIX}/dataset/Chinese_Eassy/processed_data/annotation/test.txt \
--model_name_or_path ${PREFIX}/output/gpt2_ml \
--model_type grover \
--block_size 1024 \
--per_gpu_eval_batch_size 8 \
--overwrite_cache \
--do_eval
fi &&


if [ ${GENERATE} == "True" ]; then
python -u run_generation.py \
--language zh \
--input_format ${INPUT} \
--sep ${SEP} \
--add_annotation_tokens ${ANNOT} \
--input_path ${PREFIX}/dataset/Chinese_Eassy/processed_data/test_partA.txt \
--groundT_path ${PREFIX}/dataset/Chinese_Eassy/processed_data/test_partB.txt \
--model_name_or_path ${OUTPUT}/checkpoint-${DECODE_CKPT} \
--model_type grover \
--length 1024 \
--stop_token [PASSAGE_EOS] \
--p 0.95 &&

python -u run_evaluation.py \
--language zh \
--model_type grover \
--input_path ${PREFIX}/dataset/Chinese_Eassy/processed_data/test_partA.txt \
--groundT_path ${PREFIX}/dataset/Chinese_Eassy/processed_data/test_partB.txt \
--generated_path  ${OUTPUT}/checkpoint-${DECODE_CKPT}/test_generated.txt \
--model_name_or_path ${PREFIX}/output/gpt2_ml
fi