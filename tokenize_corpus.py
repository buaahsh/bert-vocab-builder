import os
import argparse
import pickle
import json
import glob
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from toolz.itertoolz import partition_all
from collections import namedtuple

from pytorch_pretrained_bert.tokenization import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='read data from',
                    type=str, default='/mnt/data/bert_wiki/wikidoc.all')
parser.add_argument('--output', help='save data to',
                    type=str, default='/mnt/data/pretrain_data/wikidoc.tk')
parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument('--cache_dir', help='cache dir',
                    type=str, default='/mnt/data/bert-uncased-pretrained-cache/')
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
default_process_count = max(1, cpu_count() - 1)
args = parser.parse_args()


def is_invalid(tk_list):
    if len(tk_list) >= 8:
        n_subword = 0
        for tk in tk_list:
            if tk.startswith('##'):
                n_subword += 1
        if float(n_subword)/float(len(tk_list)) > 0.7:
            return True
    return False


def get_token_id(tokenizer, tok_list):
    ids = []
    for tok in tok_list:
        ids.append(str(tokenizer.vocab[tok]))
    return ids

def process_doc(_file, output_file):
    # avoid access url too frequently
    if Path(os.path.join(args.cache_dir, 'vocab.txt')).exists():
        tokenizer_load_path = args.cache_dir
    else:
        tokenizer_load_path = args.bert_model
    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_load_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)

    writer = open(output_file, 'w', encoding='utf-8')
    i = 0
    with open(_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            i += 1
        
            if i > 500000:
                break
            line = line.replace('#', '')
            tk_list = tokenizer.tokenize(line)
            ids = get_token_id(tokenizer, tk_list)
            writer.write(' '.join(tk_list))
            writer.write('\t')
            writer.write('\t'.join(ids))
            writer.write('\n')
    writer.close()
 
def main():
    process_doc(args.input, args.output)

if __name__ == '__main__':
    main()
