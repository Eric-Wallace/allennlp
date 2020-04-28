from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model,remove_pretrained_embedding_params
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import move_to_device
from allennlp.nn import util
from allennlp.common.params import Params
import sys
sys.path.append("/home/junliw1/gradient-regularization/utils")
from combine_models import merge_models
@Model.register("basic_classifier_combined")
class BasicClassifierCombined(Model):
    def __init__(self, model_1: Model, model_2: Model):
        super(BasicClassifierCombined, self).__init__(model_1.vocab)
        self.combined_model = merge_models(model_1, model_2)

    def forward(self, *args, **kwargs):
        return self.combined_model(*args, **kwargs)

    @classmethod
    def _load(
        cls, config: Params, serialization_dir: str, weights_file: str = None, cuda_device: int = -1
    ) -> "Model":
        model_params = config.get("model")

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        remove_pretrained_embedding_params(model_params)
        model = Model.from_params(vocab=None, params=model_params)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        return model

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        return self.combined_model.make_output_human_readable(output_dict)