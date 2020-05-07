import torch

def expand(model_file, tok_file):
    lines = open(tok_file, 'r', encoding='utf-8').readlines()
    model_dict = torch.load(model_file)
    # key: bert.embeddings.word_embeddings.weight
    old_embedding = model_dict['bert.embeddings.word_embeddings.weight']
    new_embedding = torch.rand([len(lines), old_embedding.size(1)])
    new_embedding[:old_embedding.size(0), :] = old_embedding

    for idx, item in enumerate(lines):
        if idx < old_embedding.size(0):
            continue
        tokens = [int(i) for i in item.strip().split('\t')[1:]]
        new_embedding_items = old_embedding[tokens, :]
        new_embedding_item = new_embedding_items.sum(0) / new_embedding_items.size(0)
        new_embedding[idx] = new_embedding_item
    model_dict['bert.embeddings.word_embeddings.weight'] = new_embedding
    torch.save(model_dict, 'electra_large.expand.bin')

def expand_electra(model_file, tok_file):
    lines = open(tok_file, 'r', encoding='utf-8').readlines()
    model_dict = torch.load(model_file)
    # key: bert.embeddings.word_embeddings.weight
    old_embedding = model_dict['electra.embeddings.word_embeddings.weight']
    new_embedding = torch.rand([len(lines), old_embedding.size(1)])
    new_embedding[:old_embedding.size(0), :] = old_embedding

    for idx, item in enumerate(lines):
        if idx < old_embedding.size(0):
            continue
        tokens = [int(i) for i in item.strip().split('\t')[1:]]
        new_embedding_items = old_embedding[tokens, :]
        new_embedding_item = new_embedding_items.sum(0) / new_embedding_items.size(0)
        new_embedding[idx] = new_embedding_item
    model_dict['electra.embeddings.word_embeddings.weight'] = new_embedding
    torch.save(model_dict, 'electra_large.expand.bin')

if __name__ == "__main__":
    model_file = r"electra_large.bin"
    # model_file = r"C:\Users\shaohanh\Downloads\pytorch_model.bin"
    # model_file = r'unilm_base_499992.bin'
    tok_file = 'ads.vocab.clean.tok'
    # expand(model_file, tok_file)
    expand_electra(model_file, tok_file)