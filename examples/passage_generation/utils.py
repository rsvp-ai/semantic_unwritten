#!/usr/bin/env python3

import torch.nn as nn
import torch
from torch.utils.data import Dataset
from typing import Optional
import os,pickle
from tqdm import tqdm
import numpy as np
from transformers.tokenization_gpt2 import whitespace_tokenize
import re
import random


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class PassageDataset(Dataset):
    '''
            Data format:
            paragraph_first [sep] paragraph_second [sep] ... paragraph_last [sep] [cls] [pad] ... [pad]
    '''

    def __init__(self, args, tokenizer,evaluate=False, annotation_size=7):
        file_path=args.eval_data_file if evaluate else args.train_data_file
        overwrite_cache = args.overwrite_cache
        sep = args.sep
        lt1024 = args.lt1024
        block_size = args.block_size
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        if lt1024:
            filename = filename+'_1024'
        cached_features_file = os.path.join(directory, 'passage_cached_lm_' + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            self.examples = []
            self.text_examples = []
            self.annotation_examples = []
            self.weights = []
            self.word_counts = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            passages = text.strip().split('\n\n\n')
            for pid, passage in enumerate(passages):
                paragraphs = passage.split('\n\n')
                passage = ''
                for paragraph in paragraphs:
                    sentences = paragraph.split("\n")
                    for sentence in sentences:
                        s, annotation = sentence.split('|||')
                        if len(self.annotation_examples) <= pid:
                            tokenized_annotation = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(annotation))
                            if len(tokenized_annotation) > annotation_size:
                                tokenized_annotation = tokenized_annotation[0:annotation_size]
                            while len(tokenized_annotation) < annotation_size:
                                tokenized_annotation.append(tokenizer.pad_token_id)
                            self.annotation_examples.append(tokenized_annotation)
                        passage = passage + ' ' + s
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(passage))
                example = tokenizer.build_inputs_with_special_tokens_passage(tokenized_text)
                if len(example) > block_size:
                    if not evaluate and lt1024:
                        self.annotation_examples.pop(-1)
                        continue
                    example = example[0:block_size]
                while len(example) < block_size:
                    example.append(tokenizer.pad_token_id)
                self.text_examples.append(example)
            self.examples = {'text': self.text_examples,
                                             'annotation': self.annotation_examples
                                             }
            # for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
            #         self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
            # # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # # If your dataset is small, first you should loook for a bigger one :-) and second you
            # # can change this behavior by adding (model specific) padding.

            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples['text'])

    def __getitem__(self, item):
        data = {}
        data["text"] = torch.tensor(self.examples['text'][item])
        # data['weight'] = torch.tensor(self.examples['weight'][item])
        # data['word_count'] = torch.tensor(self.examples['word_count'][item])
        data["annotation"] = torch.tensor(self.examples['annotation'][item])
        return data


class ParagraphDataset(Dataset):
    '''
            Data format:
            paragraph_first [mask] paragraph_second [mask] ... paragraph_last [mask] [cls] [pad] ... [pad]
    '''

    def __init__(self, args, tokenizer,evaluate=False, annotation_size=7):
        file_path=args.eval_data_file if evaluate else args.train_data_file
        overwrite_cache = args.overwrite_cache
        sep = args.sep
        lt1024 = args.lt1024
        block_size = args.block_size
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        if sep:
            filename = filename+'_sep'
        if lt1024:
            filename = filename+'_1024'
        if args.add_annotation_tokens:
            filename = filename+'_topic'
        cached_features_file = os.path.join(directory, 'paragraph_cached_lm_' + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:

            self.examples = []
            self.text_examples = []
            self.annotation_examples = []
            passage_annotation = ''
            with open(file_path, 'r',encoding="utf-8") as f:
                text = f.read()
            passages = text.strip().split('\n\n\n')
            for pid, passage in enumerate(passages):
                example = []
                paragraphs = passage.split("\n\n")
                for paragraph in paragraphs:
                    sentences = paragraph.split("\n")
                    paragraph = ''
                    annotation = ''
                    for sentence in sentences:
                        s, annotation = sentence.split('|||')
                        paragraph = paragraph + ' ' + s
                    if len(self.annotation_examples) <= pid:
                        tokenized_annotation = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(annotation))
                        if len(tokenized_annotation) > annotation_size:
                            tokenized_annotation = tokenized_annotation[0:annotation_size]
                        passage_annotation = tokenized_annotation
                        while len(tokenized_annotation) < annotation_size:
                            tokenized_annotation.append(tokenizer.pad_token_id)
                        self.annotation_examples.append(tokenized_annotation)
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(paragraph))
                    example.append(tokenized_text)
                example = tokenizer.build_inputs_with_special_tokens_paragraph(example)
                if sep:
                    example.pop(-1)
                example = tokenizer.build_inputs_with_special_tokens_passage(example)
                if args.add_annotation_tokens:
                    example = passage_annotation+tokenizer.convert_tokens_to_ids(tokenizer.tokenize('*'))+example
                if len(example) > block_size:
                    if not evaluate and lt1024:
                        self.annotation_examples.pop(-1)
                        continue
                    example = example[0:block_size]
                while len(example) < block_size:
                    example.append(tokenizer.pad_token_id)
                self.text_examples.append(example)

            # for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
            #         self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
            # # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # # If your dataset is small, first you should loook for a bigger one :-) and second you
            # # can change this behavior by adding (model specific) padding.
            self.examples = {'text': self.text_examples,
                                             'annotation': self.annotation_examples
                                             }

            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples['text'])

    def __getitem__(self, item):
        data = {}
        data["text"] = torch.tensor(self.examples['text'][item])
        data["annotation"] = torch.tensor(self.examples['annotation'][item])
        return data


class SentenceDataset(Dataset):
    '''
            Data format:
            sentence_first [sep] sentence_second [sep] ... paragraph_last sentence last [sep] [mask] [cls] [pad] ... [pad]
    '''

    def __init__(self, args, tokenizer,evaluate=False, annotation_size=7):
        file_path=args.eval_data_file if evaluate else args.train_data_file
        overwrite_cache = args.overwrite_cache
        sep = args.sep
        lt1024 = args.lt1024
        block_size = args.block_size
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        if lt1024:
            filename = filename+'_1024'
        cached_features_file = os.path.join(directory, 'sentence_cached_lm_' + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            self.text_examples = []
            self.annotation_examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            passages = text.strip().split('\n\n\n')
            for pid, passage in enumerate(passages):
                example = []
                paragraphs = passage.split("\n\n")
                for paragraph in paragraphs:
                    sentence_example = []
                    sentences = paragraph.split("\n")
                    for sentence in sentences:
                        if len(self.annotation_examples) <= pid:
                            annotation = sentence.split('|||')[1]
                            tokenized_annotation = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(annotation))
                            if len(tokenized_annotation) > annotation_size:
                                tokenized_annotation = tokenized_annotation[0:annotation_size]
                            while len(tokenized_annotation) < annotation_size:
                                tokenized_annotation.append(tokenizer.pad_token_id)
                            self.annotation_examples.append(tokenized_annotation)
                        sentence = sentence.split('|||')[0]
                        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
                        sentence_example.append(tokenized_text)
                    example.append(tokenizer.build_inputs_with_special_tokens_sentence(sentence_example))

                example = tokenizer.build_inputs_with_special_tokens_paragraph(example)
                example = tokenizer.build_inputs_with_special_tokens_passage(example)

                if len(example) > block_size:
                    if not evaluate and lt1024:
                        self.annotation_examples.pop(-1)
                        continue
                    example = example[0:block_size]
                while len(example) < block_size:
                    example.append(tokenizer.pad_token_id)
                self.text_examples.append(example)
            # for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
            #         self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
            # # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # # If your dataset is small, first you should loook for a bigger one :-) and second you
            # # can change this behavior by adding (model specific) padding.
            self.examples = {'text': self.text_examples,
                                             'annotation': self.annotation_examples
                                             }
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples['text'])

    def __getitem__(self, item):
        data = {}
        data["text"] = torch.tensor(self.examples['text'][item])
        data["annotation"] = torch.tensor(self.examples['annotation'][item])
        return data


class EnglishDataset(Dataset):
    def __init__(self,args,tokenizer,evaluate=False):
        set_seed(args)
        source_file_path=args.eval_src_data_file if evaluate else args.train_src_data_file
        target_file_path = args.eval_tgt_data_file if evaluate else args.train_tgt_data_file

        overwrite_cache = args.overwrite_cache
        block_size = args.block_size
        model_type = args.model_type
        sep = args.sep
        training_percent = args.training_percent

        assert os.path.isfile(source_file_path)
        directory, src_filename = os.path.split(source_file_path)
        filename = src_filename.split('.')[0]
        paragraph_sep = tokenizer.paragraph_eos_token
        sentence_sep = tokenizer.sentence_eos_token
        sep_name = (paragraph_sep+'_'+sentence_sep).replace(" ","none")
        sep_name = sep_name.replace('\n','newline')
        if model_type!='gpt2':
            sep_name = model_type+'_'+sep_name
        if sep:
            sep_name = 'sep' + '_' + sep_name
        without_eos = args.without_eos
        if without_eos:
            sep_name = 'withouteos' + '_' + sep_name
        if training_percent!=1.0:
            sep_name = str(training_percent)+'_'+sep_name
        cached_features_file = os.path.join(directory, sep_name+'_cached_lm_'
                                            + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            self.examples = []
            self.text_examples = []
            self.tgt_masks = []
            self.word_counts = []
            self.para_position_ids = []

            with open(source_file_path, encoding="utf-8") as f:
                src_text = f.read()
            with open(target_file_path, encoding="utf-8") as f:
                tgt_text = f.read()
            src_passages = src_text.strip().split('\n')
            tgt_passages = tgt_text.strip().split('\n')

            total_num = len(src_passages)
            sample_num = int(total_num * args.training_percent)
            training_index = list(np.random.randint(total_num, size=sample_num))

            for index,(src_passage,tgt_passage) in tqdm(enumerate(zip(src_passages,tgt_passages)),total=len(src_passages)):
                if src_passage.strip()=='':
                    continue
                if args.training_percent != 1.0 and not evaluate and index not in training_index:
                    continue
                para_count = tgt_passage.count("[PARAGRAPH_EOS]")
                sent_count = tgt_passage.count("[SENTENCE_EOS]")
                if sep:
                    para_count -= 1
                    sent_count -= 1
                def encode(text):
                    if sep:
                        if text.endswith('[PARAGRAPH_EOS]'):
                            text = text[:-len('[PARAGRAPH_EOS]')].strip()

                        if text.endswith('[SENTENCE_EOS]'):
                            text = text[:-len('[SENTENCE_EOS]')].strip()
                    passage = text.replace("[SENTENCE_EOS]", sentence_sep).strip(" ")
                    # if paragraph_sep == "[PARAGRAPH_EOS]":
                    #     passage = passage.replace("[PARAGRAPH_EOS]", '\n').strip(" ")
                    # else:
                    #     passage = passage.replace("[PARAGRAPH_EOS]", paragraph_sep).strip(" ")
                    passage = passage.replace("[PARAGRAPH_EOS]", paragraph_sep).strip(" ")
                    while "  " in passage:
                        passage = passage.replace("  ", " ")
                    passage = passage.replace(" \n", "\n")
                    passage = passage.replace("\n ", "\n")

                    ids = tokenizer.encode(passage)
                    # if not sep and tokenizer.paragraph_eos_id!=-1:
                    #     ids.append(198)
                    # if paragraph_sep == "[PARAGRAPH_EOS]":
                    #     ids = [tokenizer.paragraph_eos_id if x == 198 else x for x in ids]
                    if not sep and tokenizer.paragraph_eos_token == '\n':
                        ids.append(tokenizer.paragraph_eos_id)

                    return ids


                    # if sep:
                    #     if text.endswith('[PARAGRAPH_EOS]'):
                    #         text = text[:-len('[PARAGRAPH_EOS]')].strip()
                    #
                    #     if text.endswith('[SENTENCE_EOS]'):
                    #         text = text[:-len('[SENTENCE_EOS]')].strip()

                    # passage = text.replace("[SENTENCE_EOS]", sentence_sep).strip(" ")
                    # if paragraph_sep=='[IOP]':
                    #     count = 0
                    #     while "[PARAGRAPH_EOS]" in passage:
                    #         passage = passage.replace("[PARAGRAPH_EOS]", paragraph_sep+str(count), 1).strip(" ")
                    #         if count <30:
                    #             count+=1
                    # else:
                    #     passage = passage.replace("[PARAGRAPH_EOS]", paragraph_sep).strip(" ")
                    # while "  " in passage:
                    #     passage = passage.replace("  ", " ")
                    # while " \n" in passage:
                    #     passage = passage.replace(" \n", "\n")
                    # ids = tokenizer.encode(passage)
                    # if not sep and paragraph_sep=='\n':
                    #     ids.append(tokenizer.paragraph_eos_id)
                    # return ids

                src_ids = encode(src_passage.strip())
                tgt_ids = encode(tgt_passage.strip())
                src_length = len(src_ids)+1
                if without_eos:
                    example = src_ids + [tokenizer.sep_token_id] + tgt_ids
                else:
                    example = src_ids+[tokenizer.sep_token_id]+tgt_ids+[tokenizer.eos_token_id]

                def count_words(text):
                    text = text.replace("\n", " \n ")
                    text = text.replace("<|endoftext|>", " <|endoftext|>")
                    text = re.sub('"', ' " ', text)
                    text = re.sub('(\'|\.|\,|\:|\?|\!|;)', ' \g<1>', text)
                    # Fix contraction
                    text = text.replace("n 't", " n't")
                    text = re.sub(' +', ' ', text)
                    text = text.replace('. . .', '...')
                    # Edge cases
                    text = text.replace("ca n't-", "can't-")
                    text = text.replace("St .", "St.")
                    text = re.sub(r"//www \.(.*) \.(.*)/", r"//www\.\g<1>\.\g<1>\/", text)
                    while "  " in text:
                        text = text.replace("  ", " ")
                    tokens = text.strip().split(' ')
                    return len(tokens)

                if len(example) > block_size:
                    if not evaluate:
                        continue
                    rm_text = tokenizer.decode(example[block_size:])
                    sent_count = sent_count-(rm_text.count("[SENTENCE_EOS]"))
                    para_count = para_count-(rm_text.count("[PARAGRAPH_EOS]"))
                    example = example[0:block_size]

                if tokenizer.sentence_eos_id==-1:
                    sent_count=0
                if tokenizer.paragraph_eos_id==-1:
                    para_count=0

                tokens = example[src_length:]
                text = tokenizer.decode(tokens)
                all_word_count = count_words(text)
                #minus eos
                word_count = all_word_count-para_count-sent_count-1
                if without_eos:
                    word_count+=1

                tgt_mask = []
                while len(example) < block_size:
                    example.append(tokenizer.pad_token_id)
                    tgt_mask.append(0)

                tgt_mask = [0] * src_length + [1] * (len(example) - src_length - len(tgt_mask)) + tgt_mask
                self.text_examples.append(example)
                self.tgt_masks.append(tgt_mask)
                self.word_counts.append(word_count)
                if 'ppe' in args.model_type:
                    para_pos = (torch.tensor(example) == tokenizer.paragraph_eos_id).nonzero().squeeze(1)
                    para_position_id = (torch.arange(len(example)).unsqueeze(0).expand(
                        para_pos.shape[0], len(example)) > para_pos.unsqueeze(1)).int().sum(0)
                    para_position_id[para_position_id >= 30] = 30 - 1
                    self.para_position_ids.append(para_position_id)

            self.examples = {'text': self.text_examples,
                             'tgt_mask':self.tgt_masks,
                             'word_count':self.word_counts,
                             'para_position_ids': self.para_position_ids
                             }
            # for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
            #         self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
            # # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # # If your dataset is small, first you should loook for a bigger one :-) and second you
            # # can change this behavior by adding (model specific) padding.

            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item):
        data = {}
        data["text"] = torch.tensor(self.examples['text'][item])
        data['tgt_mask'] = torch.tensor(self.examples['tgt_mask'][item])
        data['word_count'] = torch.tensor(self.examples['word_count'][item])
        if len(self.examples['para_position_ids'])>0:
            data['para_position_ids'] = torch.tensor(self.examples['para_position_ids'][item])
        return data

    def __len__(self):
        return len(self.examples['text'])

def average_emb(model, input_ids, tokenizer,device):
    # batch*seq_len
    pad_mask = (input_ids != tokenizer.pad_token_id).unsqueeze(-1).float().to(device)
    length = torch.sum(pad_mask, 1)
    # batch*seq_len*emb_size
    inputs_embeds = model(input_ids, output_past=False, output_hidden_states=True)[1][-1] * pad_mask
    avg_embeds = (torch.sum(inputs_embeds, dim=1) / length).cpu().detach()
    return avg_embeds


def get_matching_embs(model, tokenizer, inputs,annotations,last_layer_states, device):

    annotations = annotations.to(device)
    annotation_embedding = average_emb(model, annotations, tokenizer, device)


    sequence_tensor = last_layer_states.view(-1, last_layer_states.shape[-1])
    # sent_pos = all_inputs == tokenizer.sentence_eos_id
    # seq_avg_emb
    para_pos_inverse = ((inputs != tokenizer.passage_eos_id) &
                        (inputs != tokenizer.pad_token_id) &
                        (inputs != tokenizer.paragraph_eos_id)).int()
    para_pos_right = torch.cat([torch.zeros(para_pos_inverse.shape[0], 1).int().to(device),
                                para_pos_inverse[:, :-1].int()], -1)
    para_pos_add = para_pos_right + para_pos_inverse

    mask_except_last_col = torch.zeros_like(para_pos_inverse)
    mask_except_last_col[...,-1] = mask_except_last_col[...,-1]+1

    para_pos = inputs == tokenizer.paragraph_eos_id
    non_eos_last_col = (para_pos_inverse == 1) & (mask_except_last_col== 1)

    end_pos_last_col = torch.cat([torch.tensor([False]).to(device),
                                  non_eos_last_col.view(-1)]).to(device)


    end_pos_wo_last_col = torch.cat([(para_pos_inverse.view(-1) == 0) & (para_pos_add.view(-1) == 1),
                                    torch.tensor([False]).to(device),]).to(device)

    end_pos = (end_pos_wo_last_col|end_pos_last_col).nonzero()

    start_pos = ((para_pos_inverse.view(-1) == 1) & (para_pos_add.view(-1) == 1)).nonzero()
    span_index = torch.cat([start_pos, end_pos], -1).unsqueeze(0)
    ase = AverageSpanExtractor()
    para_avg_emb = ase(sequence_tensor.unsqueeze(0), span_index).squeeze(0).to(device)
    # sep_position_emb
    # para_eos_emb = sequence_tensor[para_pos.view(-1)]
    annotation_embedding = torch.repeat_interleave(annotation_embedding,
                                                   torch.sum((para_pos|non_eos_last_col).int(), -1).to('cpu'),
                                                   dim=0).to(device)
    return para_avg_emb, annotation_embedding

def get_matching_embs_wp(model, tokenizer, inputs,annotations,last_layer_states, device):

    annotations = annotations.to(device)
    annotation_embedding = average_emb(model, annotations, tokenizer, device)


    sequence_tensor = last_layer_states.view(-1, last_layer_states.shape[-1])
    # sent_pos = all_inputs == tokenizer.sentence_eos_id
    # seq_avg_emb
    para_pos_inverse = ((inputs != tokenizer.added_tokens_encoder['[PASSAGE_EOS]']) &
                        (inputs != tokenizer.pad_token_id) &
                        (inputs != tokenizer.added_tokens_encoder['[PARAGRAPH_EOS]'])).int()
    para_pos_right = torch.cat([torch.zeros(para_pos_inverse.shape[0], 1).int().to(device),
                                para_pos_inverse[:, :-1].int()], -1)
    para_pos_add = para_pos_right + para_pos_inverse

    mask_except_last_col = torch.zeros_like(para_pos_inverse)
    mask_except_last_col[...,-1] = mask_except_last_col[...,-1]+1

    para_pos = inputs == tokenizer.added_tokens_encoder['[PARAGRAPH_EOS]']
    non_eos_last_col = (para_pos_inverse == 1) & (mask_except_last_col== 1)

    end_pos_last_col = torch.cat([torch.tensor([False]).to(device),
                                  non_eos_last_col.view(-1)]).to(device)


    end_pos_wo_last_col = torch.cat([(para_pos_inverse.view(-1) == 0) & (para_pos_add.view(-1) == 1),
                                    torch.tensor([False]).to(device),]).to(device)

    end_pos = (end_pos_wo_last_col|end_pos_last_col).nonzero()

    start_pos = ((para_pos_inverse.view(-1) == 1) & (para_pos_add.view(-1) == 1)).nonzero()
    span_index = torch.cat([start_pos, end_pos], -1).unsqueeze(0)
    ase = AverageSpanExtractor()
    para_avg_emb = ase(sequence_tensor.unsqueeze(0), span_index).squeeze(0).to(device)
    # sep_position_emb
    # para_eos_emb = sequence_tensor[para_pos.view(-1)]
    annotation_embedding = torch.repeat_interleave(annotation_embedding,
                                                   torch.sum((para_pos|non_eos_last_col).int(), -1).to('cpu'),
                                                   dim=0).to(device)
    return para_avg_emb, annotation_embedding


def get_ppl_per_paragraph(index_tuple, loss,ppl_per_para):
    start_index = index_tuple[0]
    end_index = index_tuple[1]
    for para_index, s_index in start_index.items():
        if para_index==0:
            continue
        s_index = max(0,s_index-1)
        e_index = end_index[para_index]-1
        if e_index>loss.shape[0]:
            print(1)
            break
        avg_loss = (torch.sum(loss[s_index:e_index])/(e_index-s_index)).cpu().detach()
        if e_index-s_index<=5:
            continue
        ppl = torch.exp(torch.tensor(avg_loss).float()).item()
        ppl_per_para[para_index/len(start_index)*10//1*10].append(ppl)
    return ppl_per_para


def weighted_sum(matrix: torch.Tensor,attention: torch.Tensor) -> torch.Tensor:
    """
    Args:
        matrix ():
        attention ():
    """
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)

def get_range_vector(size: int, device) -> torch.Tensor:
    """
    """
    return torch.arange(0, size, dtype=torch.long).to(device)

def flatten_and_batch_shift_indices(indices: torch.LongTensor,
                                    sequence_length: int) -> torch.Tensor:
    """``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor,
    which has size ``(batch_size, sequence_length, embedding_size)``. This function returns a vector
    that correctly indexes into the flattened target. The sequence length of the target must be provided
    to compute the appropriate offset.
    Args:
        indices (torch.LongTensor):
    """
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ValueError("All the elements should be in range (0, {}), but found ({}, {})".format(
            sequence_length - 1, torch.min(indices).item(), torch.max(indices).item()))
    offsets = get_range_vector(indices.size(0), indices.device) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # (batch_size, d_1, ..., d_n) + (batch_size, 1, ..., 1)
    offset_indices = indices + offsets

    # (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices

def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
    """Select ``target`` of size ``(batch_size, sequence_length, embedding_size)`` with ``indices`` of
    size ``(batch_size, d_1, ***, d_n)``.
    Args:
        target (torch.Tensor): A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
    """
    if flattened_indices is None:
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]

    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets

def masked_softmax(vector: torch.Tensor,
                                     mask: torch.Tensor,
                                     dim: int = -1,
                                     mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked. This performs a softmax on just the non-masked positions of ``vector``. Passing ``None``
    in for the mask is also acceptable, which is just the regular softmax.
    """
    if mask is None:
        result = torch.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
        result = torch.softmax(masked_vector, dim=dim)
    return result


class AverageSpanExtractor(nn.Module):
    def __init__(self):
        super(AverageSpanExtractor, self).__init__()

    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
        # Shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        span_ends = span_ends - 1

        span_widths = span_ends - span_starts

        max_batch_span_width = span_widths.max().item() + 1

        # sequence_tensor (batch, length, dim)
        # global_attention_logits = self._global_attention(sequence_tensor)
        global_average_logits = torch.ones(sequence_tensor.size()[:2] + (1,)).float().to(sequence_tensor.device)

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = get_range_vector(max_batch_span_width,sequence_tensor.device).view(1, 1, -1)
        span_mask = (max_span_range_indices <= span_widths).float()

        # (batch_size, num_spans, 1) - (1, 1, max_batch_span_width)
        raw_span_indices = span_ends - max_span_range_indices
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.relu(raw_span_indices.float()).long()

        flat_span_indices = flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        span_embeddings = batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        span_attention_logits = batched_index_select(global_average_logits,span_indices,flat_span_indices).squeeze(-1)

        span_attention_weights = masked_softmax(span_attention_logits, span_mask)

        attended_text_embeddings = weighted_sum(span_embeddings, span_attention_weights)

        if span_indices_mask is not None:
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).float()

        return attended_text_embeddings