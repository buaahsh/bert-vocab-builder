# merge two vocabulary files

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--bert_vocab", type=str, default="", help="bert vocab")
parser.add_argument("--ads_vocab", type=str, default="", help="ads vocab")
parser.add_argument("--output", type=str, default="", help="output")

args = parser.parse_args()

writer = open(args.output, 'w', encoding='utf-8')
_set = set()
with open(args.bert_vocab, 'r', encoding='utf-8') as reader:
    for line in reader:
        writer.write(line)
        _set.add(line.strip())

print(len(_set))
with open(args.ads_vocab, 'r', encoding='utf-8') as reader:
    for line in reader:
        if line.strip() not in _set:
            writer.write(line)

writer.close()
