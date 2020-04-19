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
    """

    def __init__(self, predictor: Predictor) -> None:
        super().__init__(predictor)
        # Hyperparameters
        self.stdev = 0.01
        self.num_samples = 10

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        # Convert inputs to labeled instances

        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        # List of embedding inputs, used for multiplying gradient by the input for normalization
        embeddings_list = []
        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            if hasattr(self.predictor._model, 'submodels'):
                embeddings_list_1 = []
                handle_1 = self._register_forward_hook_regular(embeddings_list_1, self.predictor._model.submodels[0])
                grad_1 = self._smooth_grads(instance, self.predictor._model.submodels[0])
                handle_1.remove()

                embeddings_list_2 = []
                handle_2 = self._register_forward_hook_regular(embeddings_list_2, self.predictor._model.submodels[1])
                grad_2 = self._smooth_grads(instance, self.predictor._model.submodels[1])
                handle_2.remove()

                grads = dict()
                embeddings_list_1.reverse()
                embeddings_list_2.reverse()

                for key, grad in grad_1.items():
                    input_idx = int(key[-1]) - 1
                    emb_grad_1 = numpy.sum(grad_1[key][0].numpy() * embeddings_list_1[input_idx], axis=1)
                    emb_grad_2 = numpy.sum(grad_2[key][0].numpy() * embeddings_list_2[input_idx], axis=1)
                    # emb_grad_1 = numpy.sum(grad_1[key][0].numpy(), axis=1)
                    # emb_grad_2 = numpy.sum(grad_2[key][0].numpy(), axis=1)
                    emb_grad = emb_grad_1 + emb_grad_2
                    norm = numpy.linalg.norm(emb_grad, ord=1)
                    normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                    grads[key] = normalized_grad
            else:
                # Run smoothgrad
                handle = self._register_forward_hook_regular(embeddings_list, self.predictor._model)
                grads = self._smooth_grads(instance, self.predictor._model)
                handle.remove()
                # Gradients come back in the reverse order that they were sent into the network
                embeddings_list.reverse()
                # Normalize results
                for key, grad in grads.items():
                    # TODO (@Eric-Wallace), SmoothGrad is not using times input normalization.
                    # Fine for now, but should fix for consistency.

                    # The [0] here is undo-ing the batching that happens in get_gradients.
                    input_idx = int(key[-1]) - 1
                    embedding_grad = numpy.sum(grad[0].numpy()* embeddings_list[input_idx], axis=1)
                    norm = numpy.linalg.norm(embedding_grad, ord=1)
                    normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                    grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        print("saliency_interpret_from_json(SmoothGrad)",instances_with_grads)
        return sanitize(instances_with_grads)

        labeled_instances = self.predictor.json_to_labeled_instances(inputs)



        return sanitize(instances_with_grads)

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
    def _register_forward_hook_regular(self, embeddings_list: List, model: Model):
        """
        Finds all of the TextFieldEmbedders, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().detach().numpy())

        embedding_layer = util.find_embedding_layer(model)
        handle = embedding_layer.register_forward_hook(forward_hook)

        return handle
    def _smooth_grads(self, instance: Instance, model: Model) -> Dict[str, numpy.ndarray]:
        total_gradients: Dict[str, Any] = {}
        for _ in range(self.num_samples):
            handle = self._register_forward_hook(self.stdev)
            grads = self.predictor.get_gradients([instance], False, False, False, False, model)
            print(grads)
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
