#!/usr/bin/env python3
import torch
import argparse
from transformers import BertConfig

def convert_megatron_to_huggingface(megatron_path, bert_config_file, huggingface_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    vocab_size = config.vocab_size
    print("Building PyTorch model from configuration: {}".format(str(config)))
    # Load weights from mega
    mega_model = torch.load(megatron_path)['model']
    target_model = {}
    for k, v in mega_model.items():
        if k =="bert.embeddings.word_embeddings.weight":
            target_model[k] = v[:vocab_size]
        else:
            target_model[k]=v
    # Save pytorch-model
    print("Save PyTorch model to {}".format(huggingface_path))
    torch.save(target_model, huggingface_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--megatron_path",
                        default = "/home/baihe/output/segabert_base_wiki_pt/iter_0500000/mp_rank_00/model_optim_rng.pt",
                        type = str,
                        help = "Path to the Megatron model path.")
    parser.add_argument("--bert_config_file",
                        default = "/home/baihe/project/transformers/examples/eval_finetune/config/base/config.json",
                        type = str,
                        help = "The config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--huggingface_path",
                        default="/home/baihe/output/wiki/bert_base_para_token/iter_0500000/pytorch_model.bin",
                        type = str,
                        help = "Path to the output model.")
    args = parser.parse_args()
    convert_megatron_to_huggingface(args.megatron_path,
                                     args.bert_config_file,
                                     args.huggingface_path)