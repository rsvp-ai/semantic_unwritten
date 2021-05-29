#!/usr/bin/env python3
"""Convert Grover checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse,os
import torch

from transformers import GroverConfig, GroverLMHeadModel

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_tf_weights_in_grover(model, config, tf_checkpoint_path):
  """ Load tf checkpoints in a pytorch model.
  """
  try:
    import re
    import numpy as np
    import tensorflow as tf
  except ImportError:
    logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                 "https://www.tensorflow.org/install/ for installation instructions.")
    raise

  tf_path = os.path.abspath(tf_checkpoint_path)
  logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
  # Load weights from TF model
  init_vars = tf.train.list_variables(tf_path)
  names = []
  arrays = []
  for name, shape in init_vars:
    logger.info("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    names.append(name)
    arrays.append(array)
  for name, array in zip(names, arrays):
    origin_name = name.split('/')
    name = []
    for n in origin_name:
      if n == 'embeddings':
        continue
      elif n == "LayerNorm_mlp_ln0":
        name.extend(["ln_0"])
      elif n == "LayerNorm_mlp_ln1":
        name.extend(["ln_1"])
      elif n == "context_projection_layer":
        name.extend(["attn", "c_proj"])
      elif n == "intermediate":
        name.extend(["mlp", "c_fc"])
      elif n == "output":
        name.extend(["mlp", "c_proj"])
      elif n == "query_layer":
        name.extend(["attn", "c_attn_q"])
      elif n == "key_layer":
        name.extend(["attn", "c_attn_k"])
      elif n == "value_layer":
        name.extend(["attn", "c_attn_v"])
      else:
        name.append(n)

    # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
    # which are not required for using pretrained model
    if any(n in ["adam_v", "adam_m", "global_step",
                 "adafactor_v", "adafactor_vc",
                 "adafactor_vr"] for n in name):
      logger.info("Skipping {}".format("/".join(name)))
      continue
    pointer = model
    for m_name in name:
      if m_name == "newslm":
        m_name = "transformer"
      elif m_name == "pos_embed":
        m_name = "wpe"
      elif m_name == "LayerNorm_embed_norm":
        m_name = "ln_f"
      elif m_name == "word_embed":
        m_name = "wte"
      if re.fullmatch(r'layer\d\d', m_name):
        pointer = getattr(pointer, 'h')
        l = ["h", re.split(r'layer', m_name)[1]]
        num = int(l[1])
        pointer = pointer[num]
        continue
      # if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
      #   l = re.split(r'_(\d+)', m_name)
      else:
        l = [m_name]
      if l[0] == 'kernel' or l[0] == 'gamma':
        pointer = getattr(pointer, 'weight')
      elif l[0] == 'output_bias' or l[0] == 'beta':
        pointer = getattr(pointer, 'bias')
      elif l[0] == 'output_weights':
        pointer = getattr(pointer, 'weight')
      elif l[0] == 'squad':
        pointer = getattr(pointer, 'classifier')
      else:
        try:
          pointer = getattr(pointer, l[0])
        except AttributeError:
          logger.info("Skipping {}".format("/".join(name)))
          continue
    if m_name == 'wpe' or m_name == 'wte':
      pointer = getattr(pointer, 'weight')
    elif m_name == 'kernel':
      array = np.transpose(array)
    try:
      assert pointer.shape == array.shape
    except AssertionError as e:
      e.args += (pointer.shape, array.shape)
      raise
    logger.info("Initialize PyTorch weight {}".format(name))
    pointer.data = torch.from_numpy(array)
  return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = GroverConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    model = GroverLMHeadModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_grover(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = "/data/baihe/dataset/gpt2-ml/model.ckpt-100000",
                        type = str,
                        help = "Path to the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default = "/data/baihe/gpt2-ml/configs/mega.json",
                        type = str,
                        help = "The config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default="/data/baihe/dataset/gpt2-ml/model.100000.pt",
                        type = str,
                        help = "Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.bert_config_file,
                                     args.pytorch_dump_path)