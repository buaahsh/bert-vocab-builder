import copy
import json


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 reset_dropout_ratio=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 initializer_range=0.02,
                 task_idx=None,
                 emb_scale=False,
                 fast_qkv=False,
                 ffn_type=0,
                 prenorm=False,
                 rel_pos_type=0,
                 max_rel_pos=128,
                 rel_pos_bins=32,
                 seperate_cls_rel_pos=False,
                 untie_rel_pos=False,
                 att_dnl=0,
                 key_unary=False,
                 query_rel_pos=False,
                 ln_after_ffn1=0,
                 offline_ws=False,
                 label_smoothing=None,
                 next_sentence_prediction=True):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.reset_dropout_ratio = reset_dropout_ratio
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.emb_scale = emb_scale
            self.ffn_type = ffn_type
            self.fast_qkv = fast_qkv
            self.prenorm = prenorm
            self.rel_pos_type = rel_pos_type
            self.max_rel_pos = max_rel_pos
            self.rel_pos_bins = rel_pos_bins
            self.seperate_cls_rel_pos = seperate_cls_rel_pos
            self.untie_rel_pos = untie_rel_pos
            self.att_dnl = att_dnl
            self.key_unary = key_unary
            self.query_rel_pos = query_rel_pos
            self.ln_after_ffn1 = ln_after_ffn1
            self.offline_ws = offline_ws
            self.label_smoothing = label_smoothing
            self.next_sentence_prediction = next_sentence_prediction
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


PREDEFINED_MODEL_CONFIGS = {
    'bert-3x320a8-uncased': BertConfig(
        vocab_size_or_config_json_file=30522, num_hidden_layers=3,
        intermediate_size=1280, num_attention_heads=8, hidden_size=320,
    ),
    'bert-3x768a12-uncased': BertConfig(
        vocab_size_or_config_json_file=30522, num_hidden_layers=3,
        intermediate_size=3072, num_attention_heads=12, hidden_size=768,
    ),
    'bert-6x512a8': BertConfig(
        vocab_size_or_config_json_file=28996, num_hidden_layers=6,
        intermediate_size=2048, num_attention_heads=8, hidden_size=512,
    ),
    'unilm-6x512a8': BertConfig(
        vocab_size_or_config_json_file=28996, num_hidden_layers=6,
        intermediate_size=2048, num_attention_heads=8, hidden_size=512,
        type_vocab_size=6,
    ),
    'unilmbipos-6x512a8': BertConfig(
        vocab_size_or_config_json_file=28996, num_hidden_layers=6,
        intermediate_size=2048, num_attention_heads=8, hidden_size=512,
        type_vocab_size=6, max_position_embeddings=1024,
    ),
    'bert-6x512a8_no_segment': BertConfig(
        vocab_size_or_config_json_file=28996, num_hidden_layers=6,
        intermediate_size=2048, num_attention_heads=8, hidden_size=512,
        type_vocab_size=0,
    ),
    'bert-base_no_segment-cased': BertConfig(
        vocab_size_or_config_json_file=28996, num_hidden_layers=12,
        intermediate_size=3072, num_attention_heads=12, hidden_size=768,
        type_vocab_size=0,
    ),
}

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME_ARCHIVE_MAP = {
    'bert-base-uncased': "config/bert-base-uncased.json",
    'bert-large-uncased': "config/bert-large-uncased.json",
    'bert-base-cased': "config/bert-base-cased.json",
    'bert-large-cased': "config/bert-large-cased.json",
}
