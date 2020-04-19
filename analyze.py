# diff bert vocab and incremental vocab file

def print_incremental_words(raw_vocab, new_vocab):
  raw_lines = open(raw_vocab, 'r', encoding='utf-8').readlines()
  new_lines = open(new_vocab, 'r', encoding='utf-8').readlines()
  raw_lines_set = set(raw_lines)
  new_lines = set(new_lines)
  print(len(new_lines))
  print(len(raw_lines))
  writer = open(new_vocab + '.clean', 'w', encoding='utf-8')
  for item in raw_lines:
    writer.write(item)
  for item in new_lines:
    if item not in raw_lines_set:
      writer.write(item)
  writer.close()
  # for s in (new_lines - raw_lines):
  #   print(s.encode("utf-8"))

def count_new_word(_file):
  all_word = 0
  new_word = 0
  all_inst = 0
  has_new_word_inst = 0
  with open(_file, 'r', encoding='utf-8') as reader:
    for line in reader:
      tok = line.strip().split('\t')[1:]
      ids = [int(i) for i in tok if i != 100]
      all_word += len(ids)
      all_inst += 1
      new_w = 0
      for _id in ids:
        if _id > 30522:
          new_w += 1
      if new_w > 0:
        has_new_word_inst += 1
      new_word += new_w
  print(all_word, new_word)
  print(all_inst, has_new_word_inst)

if __name__ == '__main__':
  # raw_vocab = r'vocab.txt'
  # new_vocab = r'ads.vocab'
  # print_incremental_words(raw_vocab, new_vocab)
  count_new_word('t.tsv.tok')