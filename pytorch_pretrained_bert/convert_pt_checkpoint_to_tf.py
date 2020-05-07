# coding=utf-8
"""Convert BERT checkpoint.
ref: https://github.com/nikitakit/self-attentive-parser/blob/8238e79e2089300db059eddff78229a09e254f70/export/export_bert.py#L94-L141
"""

import sys
import bert
import bert.modeling
import bert.tokenization

from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.config import BertConfig
import numpy as np
import torch
import tensorflow as tf
import argparse
import re
import os
from shutil import copyfile


parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--pt_checkpoint_path", default=None, type=str, required=True,
                    help="Path the PyTorch checkpoint path.")
parser.add_argument("--tf_dump_path", default=None, type=str, required=True,
                    help="Path to the output TensorFlow model.")
parser.add_argument("--bert_model", default='bert-large-cased', type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--config_copy_path", default='/mnt/data/bert_convert/cased_L-24_H-1024_A-16/', type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
args = parser.parse_args()


def convert_pt_checkpoint_to_tf():
    sess = tf.InteractiveSession()
    # load pytorch model
    model_recover = torch.load(args.pt_checkpoint_path)
    pt_model = BertForPreTraining.from_pretrained(
        args.bert_model, state_dict=model_recover)
    model_recover = None

    input_ids = tf.placeholder(
        shape=(None, None), dtype=tf.int32, name='input_ids')
    word_end_mask = tf.placeholder(
        shape=(None, None), dtype=tf.int32, name='word_end_mask')
    # We can derive input_mask from either input_ids or word_end_mask
    input_mask = (1 - tf.cumprod(1 - word_end_mask, axis=-1, reverse=True))
    token_type_ids = tf.zeros_like(input_ids)

    # Transfer BERT config into tensorflow implementation
    config = bert.modeling.BertConfig.from_dict(pt_model.config.to_dict())
    model = bert.modeling.BertModel(config=config, is_training=False,
                                    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    # Next, transfer learned weights (after fine-tuning)
    bert_variables = [v for v in tf.get_collection(
        'variables') if 'bert' in v.name]
    tf.variables_initializer(bert_variables).run()

    for variable in bert_variables:
        name = variable.name.split(':')[0]
        name = name.split('/')
        array = variable.eval()
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pytorch_var = pt_model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pytorch_var = getattr(pytorch_var, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pytorch_var = getattr(pytorch_var, 'bias')
            elif l[0] == 'output_weights':
                pytorch_var = getattr(pytorch_var, 'weight')
            else:
                pytorch_var = getattr(pytorch_var, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pytorch_var = pytorch_var[num]
        if m_name[-11:] == '_embeddings':
            pytorch_var = getattr(pytorch_var, 'weight')
        elif m_name == 'kernel':
            pytorch_var = pytorch_var.t()
        try:
            assert pytorch_var.shape == array.shape
        except AssertionError as e:
            e.args += (pytorch_var.shape, array.shape)
            raise

        variable.load(pytorch_var.detach().cpu().numpy())

    saver = tf.train.Saver()
    os.makedirs(args.tf_dump_path, exist_ok=True)
    saver.save(sess, os.path.join(args.tf_dump_path, 'bert_model.ckpt'))

    # copy config file and vocab file
    for fn in ('bert_config.json', 'vocab.txt'):
        src = os.path.join(args.config_copy_path, fn)
        tgt = os.path.join(args.tf_dump_path, fn)
        copyfile(src, tgt)


if __name__ == "__main__":
    convert_pt_checkpoint_to_tf()
