# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from examples.passage_generation.utils import EnglishDataset, get_matching_embs_wp, get_ppl_per_paragraph
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          GPT2ParaPosConfig,GPT2LMHeadParaPosModel)


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,GPT2LMHeadModel),
    'gpt2_ppe': (GPT2ParaPosConfig,GPT2LMHeadParaPosModel , GPT2Tokenizer, GPT2LMHeadParaPosModel)
}


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = EnglishDataset(args, tokenizer,evaluate)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.warmup_steps = args.warmup_steps //max(1, args.n_gpu)
    args.save_steps = int(args.save_steps) // max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    if args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint'
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

        _rotate_checkpoints(args, checkpoint_prefix)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # text,annotation,paragraph_pos,sentence_pos = batch
            inputs, labels, tgt_masks = batch['text'], batch['text'], batch['tgt_mask']
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            # tgt_masks = tgt_masks.to(args.device)
            model.train()
            if 'ppe' in args.model_type:
                para_position_ids = batch['para_position_ids'].to(args.device)
                outputs = model(inputs,labels=labels,para_position_ids=para_position_ids)
            else:
                outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            # loss = torch.sum(loss*tgt_masks[..., 1:].contiguous().view(-1).float())/torch.sum(tgt_masks[...,1:])
            pad_mask = (inputs!=tokenizer.pad_token_id)[...,1:].contiguous().float()
            eos_mask = (inputs==tokenizer.eos_token_id)[...,1:].contiguous().float()
            if tokenizer.pad_token_id==tokenizer.eos_token_id:
                pad_mask_right = torch.cat([torch.zeros(pad_mask.shape[0], 1).float().to(args.device),
                                             pad_mask[:, :-1]], -1)
                pad_mask = ((pad_mask + pad_mask_right) != 0).float()
                eos_mask = (eos_mask == pad_mask).float()
            loss = torch.sum(loss*(pad_mask.view(-1))) / torch.sum(pad_mask)
            eos_loss = torch.sum(loss*(eos_mask.view(-1)))/torch.sum(eos_mask)
            # TODO: mask prompt for this task
            if args.match_annotation_task:
                last_layer_states = outputs[2][-1]
                annotations = batch['annotation']
                para_avg_emb, annotation_embedding =get_matching_embs_wp(model, tokenizer, inputs, annotations,
                                                                      last_layer_states, args.device)
                loss_annotation = model.maching_forward(para_avg_emb, annotation_embedding,
                                                        loss_type=args.annotation_loss_type)
                lambda_annotation_weight = args.annotation_loss_weight
                loss=(1-lambda_annotation_weight)*loss + lambda_annotation_weight*loss_annotation
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                eos_loss = eos_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                eos_loss = eos_loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar('eos_loss', eos_loss.item() / args.logging_steps, global_step)
                    logger.info("Training Loss %f", (tr_loss - logging_loss) / args.logging_steps)
                    logger.info("EOS Loss %f", eos_loss.item() / args.logging_steps)
                    logging_loss = tr_loss


                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_lm_loss = 0.0
    eval_loss_pass_eos = 0.0
    eval_loss_para_eos = 0.0
    eval_lm_loss_with_sp = 0.0
    eval_loss_matching = 0.0
    # ppl_word_macro = []
    # ppl_token_macro_with_source = []

    ppl_per_para = defaultdict(list)
    with open(args.eval_src_data_file, encoding="utf-8") as f:
        src_text = f.read()
    with open(args.eval_tgt_data_file, encoding="utf-8") as f:
        tgt_text = f.read()
    src_passages = src_text.strip().split('\n')
    tgt_passages = tgt_text.strip().split('\n')
    current_index = -1

    nb_eval_steps = 0
    model.eval()

    count_pass = 0
    count_para = 0
    count_word = 0
    count_token = 0
    count_sent = 0
    para_next_logits = []
    paragraph_counter = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels, tgt_mask = batch['text'], batch['text'], batch['tgt_mask']
        word_count = batch['word_count']
        para_mask = ((inputs==tokenizer.paragraph_eos_id) & tgt_mask.bool()).int()
        if tokenizer.paragraph_eos_token == '[IOP]':
            para_mask = ((inputs <= tokenizer.max_paragraph_eos_id)&(inputs >= tokenizer.min_paragraph_eos_id) & tgt_mask.bool()).int()
        para_right = torch.cat([torch.zeros(para_mask.shape[0], 1).int(),
                                    para_mask[:, :-1].int()], -1)
        pass_mask = (inputs==tokenizer.eos_token_id).int()
        if tokenizer.eos_token_id == tokenizer.pad_token_id:
            pass_mask_right = torch.cat([torch.zeros(pass_mask.shape[0], 1).int(),
                                    pass_mask[:, :-1].int()], -1)
            pass_mask = ((pass_mask+pass_mask_right)==1).int()
        pass_mask = (pass_mask.bool() & tgt_mask.bool()).int()
        if args.sep:
            para_right = para_right+pass_mask
        if tokenizer.paragraph_eos_id==-1:
            para_right = pass_mask
        sent_mask = ((inputs==tokenizer.sentence_eos_id) & tgt_mask.bool()).int()

        token_count = torch.sum(tgt_mask.int()-para_mask-sent_mask-pass_mask,-1)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            if 'ppe' in args.model_type:
                para_position_ids = batch['para_position_ids'].to(args.device)
                loss, logits, outputs = model(inputs, labels=labels,para_position_ids=para_position_ids)
            else:
                loss, logits, outputs = model(inputs,labels=labels)

            lm_loss_with_sp = torch.sum(loss*tgt_mask[..., 1:].contiguous().view(-1).float().to(args.device))

            # # marco
            # lm_loss_marco = torch.sum(loss.cpu().view(tgt_mask.shape[0],-1) * tgt_mask[..., 1:].contiguous().float(),-1)
            # ppl_word_macro.extend(torch.exp(lm_loss_marco/word_count.float()).tolist())
            # lm_loss_marco_source = torch.sum(loss.cpu().view(tgt_mask.shape[0],-1) * (tgt_mask | (inputs.cpu()!=tokenizer.pad_token_id).long())[..., 1:].contiguous().float(),-1)
            # ppl_token_macro_with_source.extend(torch.exp(lm_loss_marco_source/token_count.float()).tolist())

            if not args.sep:
                pass_mask_left = torch.cat([pass_mask.int(),
                                             torch.zeros(pass_mask.shape[0], 1).int()], -1)
                lm_loss_with_sp = lm_loss_with_sp-torch.sum(loss*pass_mask_left[..., 2:].contiguous().view(-1).float().to(args.device))
            tgt_mask[sent_mask.bool()] = 0
            tgt_mask[para_mask.bool()] = 0
            tgt_mask[pass_mask.bool()] = 0

            lm_loss = torch.sum(loss * tgt_mask[..., 1:].contiguous().view(-1).float().to(args.device))
            para_loss = torch.sum(loss*para_mask[..., 1:].contiguous().view(-1).float().to(args.device))
            pass_loss = torch.sum(loss * pass_mask[..., 1:].contiguous().view(-1).float().to(args.device))
            batch_size = inputs.shape[0]

            for i in range(batch_size):
                loss_i = loss.reshape(batch_size,-1)[i]
                start_index_dict = {}
                end_index_dict = {}

                example = inputs[i].tolist()
                eos_pos = (np.array(example) == tokenizer.eos_token_id).nonzero()[0][0]
                def get_para_index(text, tokenizer, para_index, text_index):
                    passage = text.replace("[SENTENCE_EOS]", args.sentence_eos).strip(" ")
                    passage = passage.replace("[PARAGRAPH_EOS]", '\n').strip(" ")
                    while "  " in passage:
                        passage = passage.replace("  ", " ")
                    passage = passage.replace(" \n ", "\n ")

                    ids = tokenizer.encode(passage.strip())
                    ids = [str(id) for id in ids]
                    ids_str = ' '.join(ids)
                    paragraphs = ids_str.split(' 198 ')
                    for paragraph in paragraphs:
                        if paragraph == '':
                            continue
                        para_length = len(paragraph.strip().split())
                        start_index_dict[para_index] = text_index
                        end_index_dict[para_index] = text_index + para_length
                        para_index += 1
                        text_index += para_length
                        if text_index > eos_pos:
                            break

                    return para_index, text_index


                if tokenizer.paragraph_eos_id == -1:
                    current_index += 1
                    tgt_passage = tgt_passages[current_index]
                    src_passage = src_passages[current_index]
                    while src_passage.strip()=='':
                        current_index+=1
                        tgt_passage = tgt_passages[current_index]
                        src_passage = src_passages[current_index]
                    para_index, text_index = get_para_index(src_passage, tokenizer, 0, 0)
                    para_index, text_index = get_para_index(tgt_passage, tokenizer, para_index, text_index+1)

                else:
                    all_para_pos = (np.array(example) == tokenizer.paragraph_eos_id).nonzero()[0]
                    text_index = 0
                    for para_index, index in enumerate(all_para_pos):
                        start_index_dict[para_index] = text_index
                        end_index_dict[para_index] = index
                        text_index = index + 1
                        if text_index >= eos_pos:
                            break
                    if args.sep:
                        start_index_dict[len(all_para_pos)] = text_index
                        end_index_dict[len(all_para_pos)] = eos_pos
                ppl_per_para = get_ppl_per_paragraph((start_index_dict, end_index_dict),loss_i,ppl_per_para)
            if not args.eval_all_checkpoints:
                paragraph_counter.append(torch.sum(para_right[...,1:],axis=-1))
                para_next_logits.append(logits[:, :-1, :].reshape(-1,logits.shape[-1])[para_right[..., 1:].reshape(-1)==1].cpu())
            # for logits_per_example, para_pos_per_example in zip(
            #         logits[:, :-1, :].cpu().numpy(),para_right[..., 1:].numpy()):
            #     para_next_logits.append(logits_per_example[para_pos_per_example==1])
            if args.match_annotation_task:
                last_layer_states = outputs[2][-1]
                annotations = batch['annotation']
                para_avg_emb, annotation_embedding =get_matching_embs_wp(model, tokenizer, inputs, annotations,
                                                                      last_layer_states, args.device)
                loss_annotation = model.maching_forward(para_avg_emb, annotation_embedding,
                                                        loss_type=args.annotation_loss_type)
                eval_loss_matching += loss_annotation


            count_pass += torch.sum(pass_mask[..., 1:])
            count_para += torch.sum(para_mask[..., 1:])
            if not args.sep:
                count_para -= torch.sum(pass_mask[..., 1:])
            count_sent += torch.sum(sent_mask[..., 1:])
            count_word += torch.sum(word_count).item()
            count_token += torch.sum(token_count).item()
            eval_lm_loss += lm_loss.item()
            eval_lm_loss_with_sp += lm_loss_with_sp.item()

            eval_loss_pass_eos += torch.sum(pass_loss)
            eval_loss_para_eos += torch.sum(para_loss)
        nb_eval_steps += 1
    eval_lm_loss_word = eval_lm_loss / count_word
    eval_lm_loss_token = eval_lm_loss / count_token
    eval_loss_pass_eos = eval_loss_pass_eos/count_pass
    eval_loss_para_eos = eval_loss_para_eos / count_para if count_para.item()!=0 else 0
    eval_loss_matching = eval_loss_matching/nb_eval_steps
    eval_lm_loss_word_with_sp = eval_lm_loss_with_sp/((count_word+count_para+count_sent+count_pass).float())
    eval_lm_loss_token_with_sp = eval_lm_loss_with_sp/((count_token+count_para+count_sent+count_pass).float())
    perplexity_word = torch.exp(torch.tensor(eval_lm_loss_word).float())
    perplexity_token = torch.exp(torch.tensor(eval_lm_loss_token).float())
    perplexity_word_with_sp = torch.exp(torch.tensor(eval_lm_loss_word_with_sp).float())
    perplexity_token_with_sp = torch.exp(torch.tensor(eval_lm_loss_token_with_sp).float())
    perplexity_pass_eos = torch.exp(torch.tensor(eval_loss_pass_eos).float())
    perplexity_para_eos = torch.exp(torch.tensor(eval_loss_para_eos).float())
    eval_loss_matching = torch.tensor(eval_loss_matching)

    with open(os.path.join(args.output_dir, 'loss_per_para.npz'), 'wb') as handle:
        pickle.dump(ppl_per_para,handle,protocol=pickle.HIGHEST_PROTOCOL)

    result = {
        "perplexity_word": perplexity_word.item(),
        "perplexity_token": perplexity_token.item(),
        "perplexity_word_with_sp": perplexity_word_with_sp.item(),
        "perplexity_token_with_sp": perplexity_token_with_sp.item(),
        "perplexity_pass_eos": perplexity_pass_eos.item(),
        "perplexity_para_eos": perplexity_para_eos.item(),
        # "ppl_word_macro": np.average(ppl_word_macro),
        # "ppl_token_macro_with_source": np.average(ppl_token_macro_with_source),
    }
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))
    if not args.eval_all_checkpoints:
        para_next_logits = torch.cat(para_next_logits).numpy()
        paragraph_counter = torch.cat(paragraph_counter).numpy()
        with open(os.path.join(args.output_dir,'para_next_logits.npz'), 'wb') as handle:
            #pickle.dump(para_next_logits,handle,protocol=pickle.HIGHEST_PROTOCOL)
            np.save(handle,para_next_logits)
        with open(os.path.join(args.output_dir,'para_next_counter.npz'), 'wb') as handle:
            #pickle.dump(para_next_logits,handle,protocol=pickle.HIGHEST_PROTOCOL)
            np.save(handle,paragraph_counter)

    return result



def main():
    parser = argparse.ArgumentParser()

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    ## Required parameters
    # parser.add_argument("--input_format", default="passage", type=str, required=True,
    #                     help="The input format of data file, including passage, paragraph, and sentence.")
    parser.add_argument("--paragraph_eos", default="none", type=str,
                        help="The input format of data file, including passage, paragraph, and sentence.")
    parser.add_argument("--sentence_eos", default="none", type=str,
                        help="The input format of data file, including passage, paragraph, and sentence.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--sep", default="False", type=boolean_string,
                        help="EOP of SEP")
    parser.add_argument("--without_eos", default="False", type=boolean_string)
    parser.add_argument("--training_percent", default=1.0, type=float,
                        help="usage percent of training data")

    ## Other parameters
    parser.add_argument("--train_src_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--train_tgt_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_src_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--eval_tgt_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--match_annotation_task", type=boolean_string, default="False",
                        help="Whether to run multi-task training.")
    parser.add_argument("--lt1024", type=boolean_string, default="False",
                        help="Whether to train with seq_len < 1024.")

    parser.add_argument("--annotation_loss_type", default='kl', type=str,
                        help="annotation_loss_type")
    parser.add_argument("--annotation_loss_weight", default=0.2, type=float,
                        help="annotation_loss_weight")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--checkpoint', action='store_true',
                        help="Whether to use checkpoint technique to save GPU memory")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_tgt_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.sentence_eos == 'none':
        args.sentence_eos = ' '
    else:
        args.sentence_eos = "[SENTENCE_EOS]"
    if args.paragraph_eos == 'none':
        args.paragraph_eos = ' '
    elif args.paragraph_eos == 'newline':
        args.paragraph_eos = "\n"
    elif args.paragraph_eos == 'eos':
        args.paragraph_eos = "[PARAGRAPH_EOS]"
    else:
        args.paragraph_eos = "[IOP]"
    config_class, model_class, tokenizer_class, matching_model_class = MODEL_CLASSES[args.model_type]
    if args.match_annotation_task:
        model_class = matching_model_class
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.checkpoint=True if args.checkpoint else False
    config.output_past = False
    config.output_hidden_states = True
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,
                                                sep_token="---",eos_token="<|endoftext|>",pad_token="<|endoftext|>",
                                                paragraph_eos_token=args.paragraph_eos,
                                                sentence_eos_token=args.sentence_eos,)
    config.eop_id = tokenizer.paragraph_eos_id
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer,evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        # # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        #
        # model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        best_checkpoint = ''
        lowest_ppl_with_sp = 100
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if (checkpoint.find('checkpoint') != -1 and (len(checkpoints) > 1))else ""
            config = config_class.from_pretrained(checkpoint)
            config.checkpoint = False
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
            if 'gpt2' in args.output_dir:
                tokenizer = tokenizer_class.from_pretrained(
                    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                    do_lower_case=args.do_lower_case,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                    sep_token="---", eos_token="<|endoftext|>",pad_token="<|endoftext|>",
                    paragraph_eos_token=args.paragraph_eos,
                    sentence_eos_token=args.sentence_eos, )
            config.eop_id = tokenizer.paragraph_eos_id
            model = model_class.from_pretrained(checkpoint,config=config)
            # model_to_resize = model.module if hasattr(model, 'module') else model
            # model_to_resize.resize_token_embeddings(len(tokenizer))
            model.to(args.device)

            result = evaluate(args, model, tokenizer, prefix=prefix)
            if result['perplexity_word']<lowest_ppl_with_sp:
                lowest_ppl_with_sp = result['perplexity_word']
                best_checkpoint = checkpoint
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        import json
        json.dump(results,open(os.path.join(args.output_dir,'result.txt'),'w',encoding='utf-8'))
        with open(os.path.join(args.output_dir,'best_step.txt'),'w',encoding='utf-8') as f:
            f.write(best_checkpoint)
    return results


if __name__ == "__main__":
    main()
