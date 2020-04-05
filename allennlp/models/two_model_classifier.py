from typing import Dict

import torch

from allennlp.models.model import Model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models.model import remove_pretrained_embedding_params
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("two_model_classifier")
class TwoModelClassifier(Model):
    def __init__(self, model_1: Model, model_2: Model) -> None:
        vocab = model_1.vocab 
        if model_2.vocab != vocab:
            raise ConfigurationError("Vocabularies in two models differ")

        super().__init__(vocab, None)

        self.submodels = torch.nn.ModuleList([model_1, model_2])

        self.loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

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

    def forward(self, tokens: Dict[str, torch.LongTensor], label: torch.IntTensor = None):
        output_dict = dict()

        model_1_logits = self.submodels[0](tokens)['logits']
        model_2_logits = self.submodels[1](tokens)['logits']

        combined_logits = model_1_logits + model_2_logits
        probs = torch.nn.functional.softmax(combined_logits, dim=-1)
        output_dict['logits'] = combined_logits 
        output_dict['probs'] = probs

        if label is not None:
            loss = self.loss(combined_logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(combined_logits, label)

        return output_dict 

