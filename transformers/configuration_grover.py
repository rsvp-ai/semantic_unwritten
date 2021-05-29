#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

Grover_PRETRAINED_CONFIG_ARCHIVE_MAP = {"grover-metga": "https://s3.amazonaws.com/models.huggingface.co/???",
                                      "gpt2-base": "https://s3.amazonaws.com/models.huggingface.co/???",
                                      "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/???"}

class GroverConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `GPT2Model`.

    Args:
        vocab_size: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
        n_positions: Number of positional embeddings.
        n_ctx: Size of the causal mask (usually same as n_positions).
        n_embd: Dimensionality of the embeddings and hidden states.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        layer_norm_epsilon: epsilon to use in the layer norm layers
        resid_pdrop: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attn_pdrop: The dropout ratio for the attention
            probabilities.
        embd_pdrop: The dropout ratio for the embeddings.
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
    """
    pretrained_config_archive_map = Grover_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        vocab_size,
        n_ctx=1024,
        resid_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        hidden_act="gelu",
        summary_type='cls_index',
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        **kwargs
    ):
        """Constructs GPT2Config.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        super(GroverConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.hidden_act = hidden_act
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
