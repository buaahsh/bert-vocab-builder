__version__ = "0.4.0"
from .tokenization import BertTokenizer, BasicTokenizer, WordpieceTokenizer
from .modeling import (BertModel, BertForPreTraining, BertForPreTrainingLossMask)
from .config import BertConfig
from .optimization import BertAdam, BertAdamFineTune
try:
    from .optimization_fp16 import FP16_Optimizer_State
except:
    print("No support for fp16")
from .file_utils import PYTORCH_PRETRAINED_BERT_CACHE
