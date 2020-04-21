import math
from typing import Dict, Any, List

import numpy
import torch

from allennlp.models.model import Model
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.predictors import Predictor
from allennlp.nn import util


@SaliencyInterpreter.register("smooth-gradient")
class SmoothGradient(SaliencyInterpreter):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1706.03825)

    Registered as a `SaliencyInterpreter` with name "smooth-gradient".
    """

    def __init__(self, predictor: Predictor) -> None:
        super().__init__(predictor)
        # Hyperparameters
        self.stdev = 0.01
        self.num_samples = 10

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run smoothgrad
            grads = self._smooth_grads(instance, self.predictor._model)

            # Normalize results
            for key, grad in grads.items():
                # TODO (@Eric-Wallace), SmoothGrad is not using times input normalization.
                # Fine for now, but should fix for consistency.

                # The [0] here is undo-ing the batching that happens in get_gradients.
                embedding_grad = numpy.sum(grad[0].numpy(), axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads

        return sanitize(instances_with_grads)

    def sst_interpret_from_instances(self, labeled_instances, embedding_op, normalization, normalization2, cuda: bool, higher_order_grad: bool): 
        # Get raw gradients and outputs

        norm_grads = []
        raw_grads = []

        for instance in labeled_instances:
            embeddings_list = []
            handle = self._register_forward_hook2(embeddings_list)
            grads = self._smooth_grads(instance)

            for key, grad in grads.items():
                if embedding_op == "dot":
                    input_idx = int(key[-1]) - 1
                    summed_across_embedding_dim = torch.sum(grad[0] * embeddings_list[input_idx], axis=1)
                elif embedding_op == 'l2':
                    summed_across_embedding_dim = torch.norm(grad[0], dim=1)

                # Normalize the gradients 
                normalized_grads = summed_across_embedding_dim
                if normalization == "l2":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                elif normalization == "l1":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

                if normalization2 == "l2":
                    normalized_grads = normalized_grads**2
                elif normalization2 == "l1":
                    normalized_grads = torch.abs(normalized_grads)

                norm_grads.append(normalized_grads)
                raw_grads.append(summed_across_embedding_dim)

            handle.remove()
        return norm_grads, raw_grads

    def _register_forward_hook2(self, embeddings_list: List):
        """
        Finds all of the TextFieldEmbedders, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().detach())

        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)

        return handle

    def _register_forward_hook(self, stdev: float):
        """
        Register a forward hook on the embedding layer which adds random noise to every embedding.
        Used for one term in the SmoothGrad sum.
        """

        def forward_hook(module, inputs, output):
            # Random noise = N(0, stdev * (max-min))
            scale = output.detach().max() - output.detach().min()
            noise = torch.randn(output.shape).to(output.device) * stdev * scale

            # Add the random noise
            output.add_(noise)

        # Register the hook
        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _smooth_grads(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        total_gradients: Dict[str, Any] = {}
        for _ in range(self.num_samples):
            handle = self._register_forward_hook(self.stdev)
            grads = self.predictor.get_gradients([instance], False, False, False, False)[0]
            handle.remove()

            # Sum gradients
            if total_gradients == {}:
                total_gradients = grads
            else:
                for key in grads.keys():
                    total_gradients[key] += grads[key]

        # Average the gradients
        for key in total_gradients.keys():
            total_gradients[key] /= self.num_samples

        return total_gradients
