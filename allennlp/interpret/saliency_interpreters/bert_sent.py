from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
import torch

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
config = Config(
    testing=True,
    seed=1,
    batch_size=64,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100, # necessary to limit memory usage
    max_vocab_size=100000,
)
token_indexer = PretrainedBertIndexer(
    pretrained_model="bert-base-uncased",
    max_pieces=config.max_seq_len,
    do_lowercase=True,
 )
 
def tokenizer(s: str):
    return token_indexer.wordpiece_tokenizer(s)[:config.max_seq_len - 2]
vocab = Vocabulary()

bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased",top_layer_only=True)
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                             # we'll be ignoring masks so we'll need to set this to True
                                                            allow_unmatched_keys = True)
BERT_DIM = word_embeddings.get_output_dim()
 
class BertSentencePooler(Seq2VecEncoder):
    def forward(self, embs: torch.tensor, 
                mask: torch.tensor=None) -> torch.tensor:
        # extract first token tensor
        return embs[:, 0]
     
    @overrides
    def get_output_dim(self) -> int:
        return BERT_DIM
encoder = BertSentencePooler(vocab)