import math
from typing import List
import numpy
import torch
from allennlp.models.model import Model
from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util
from allennlp.nn.util import move_to_device

@SaliencyInterpreter.register("simple-gradient")
class SimpleGradient(SaliencyInterpreter):
    """
    Registered as a `SaliencyInterpreter` with name "simple-gradient".
    """

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Interprets the model's prediction for inputs.  Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        # List of embedding inputs, used for multiplying gradient by the input for normalization
        embeddings_list = []
        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Hook used for saving embeddings
            handle = self._register_forward_hook(embeddings_list)
            grads = self.predictor.get_gradients([instance], False, False, False, False)[0]
            print(grads)
            handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            for key, grad in grads.items():
                # Get number at the end of every gradient key (they look like grad_input_[int],
                # we're getting this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array to match up the
                # gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                # The [0] here is undo-ing the batching that happens in get_gradients.
                emb_grad = numpy.sum(grad[0].numpy() * embeddings_list[input_idx], axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def saliency_interpret_from_instances(self, labeled_instances) -> JsonDict:
        """
        Interprets the model's prediction for inputs.  Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """

        # List of embedding inputs, used for multiplying gradient by the input for normalization
        embeddings_list: List[numpy.ndarray] = []
        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Hook used for saving embeddings
            handle = self._register_forward_hook(embeddings_list)
            grads = self.predictor.get_gradients([instance], False, False, False, False)[0]
            handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            for key, grad in grads.items():
                # Get number at the end of every gradient key (they look like grad_input_[int],
                # we're getting this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array to match up the
                # gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                # The [0] here is undo-ing the batching that happens in get_gradients.
                emb_grad = numpy.sum(grad[0].detach() * embeddings_list[input_idx], axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_forward_hook(self, embeddings_list: List):
        """
        Finds all of the TextFieldEmbedders, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().detach().numpy())

        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)

        return handle

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

    def sst_interpret_from_instances(self, labeled_instances, embedding_op, normalization, normalization2, cuda: bool, higher_order_grad: bool):
        # Get raw gradients and outputs
        embeddings_list = []
        handle = self._register_forward_hook2(embeddings_list)

        grads = self.predictor.get_gradients(labeled_instances, cuda, True, True, higher_order_grad)[0]

        norm_grads = []
        raw_grads = []
        for key, grad in grads.items():
            for idx, gradient in enumerate(grad):
                if embedding_op == "dot":
                    input_idx = int(key[-1]) - 1
                    summed_across_embedding_dim = torch.sum(gradient * embeddings_list[input_idx][idx], axis=1)
                elif embedding_op == 'l2':
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)

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
        