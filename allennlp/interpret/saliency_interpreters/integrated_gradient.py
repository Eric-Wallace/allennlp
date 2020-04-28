import math
from typing import List, Dict, Any

import numpy
import torch

from allennlp.models.model import Model
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util


@SaliencyInterpreter.register("integrated-gradient")
class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)
    """

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            if hasattr(self.predictor._model, 'submodels'):
                # embeddings_list_1 = []
                # handle_1 = self._register_forward_hook_regular(embeddings_list_1, self.predictor._model.submodels[0])
                grad_1 = self._integrate_gradients(instance, self.predictor._model.submodels[0])
                # handle_1.remove()

                # embeddings_list_2 = []
                # handle_2 = self._register_forward_hook_regular(embeddings_list_2, self.predictor._model.submodels[1])
                grad_2 = self._integrate_gradients(instance, self.predictor._model.submodels[1])
                # handle_2.remove()
                # embeddings_list_1.reverse()
                # embeddings_list_2.reverse()
                # Normalize results
                grads = dict()
                for key, grad in grad_1.items():
                    # The [0] here is undo-ing the batching that happens in get_gradients.
                    input_idx = int(key[-1]) - 1
                    # emb_grad_1 = numpy.sum(grad_1[key][0].numpy() * embeddings_list_1[input_idx], axis=1)
                    # emb_grad_2 = numpy.sum(grad_2[key][0].numpy() * embeddings_list_2[input_idx], axis=1)
                    emb_grad_1 = numpy.sum(grad_1[key][0].numpy(), axis=1)
                    emb_grad_2 = numpy.sum(grad_2[key][0].numpy(), axis=1)
                    embedding_grad = emb_grad_1 + emb_grad_2
                    print("integrated gradient",embedding_grad)
                    norm = numpy.linalg.norm(embedding_grad, ord=1)
                    normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                    grads[key] = normalized_grad
            else:
                # Run integrated gradients
                grads = self._integrate_gradients(instance, self.predictor._model)

                # Normalize results
                for key, grad in grads.items():
                    # The [0] here is undo-ing the batching that happens in get_gradients.
                    print(grad[0].numpy())
                    embedding_grad = numpy.sum(grad[0].numpy(), axis=1)
                    norm = numpy.linalg.norm(embedding_grad, ord=1)
                    normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                    grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads

        return sanitize(instances_with_grads)
    def sst_interpret_from_instances(self, labeled_instances, embedding_op, normalization, normalization2, cuda: bool):
        # Get raw gradients and outputs
        embeddings_list = []
        norm_grads = []
        raw_grads = []
        for idx, instance in enumerate(labeled_instances):
            grads = self._integrate_gradients2(instance, self.predictor._model)
            # Normalize results
            for key, grad in grads.items():
                # The [0] here is undo-ing the batching that happens in get_gradients.
                summed_across_embedding_dim = torch.sum(grad[0], dim=1)
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

        return norm_grads, raw_grads
    def _register_forward_hook(self, alpha: int, embeddings_list: List):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used
        for one term in the Integrated Gradients sum.
        We store the embedding output into the embeddings_list when alpha is zero.  This is used
        later to element-wise multiply the input by the averaged gradients.
        """

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach().numpy())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle
    def _register_forward_hook_regular(self, alpha: int, embeddings_list: List):
        """
        Finds all of the TextFieldEmbedders, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).cpu().detach().numpy())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle
    def _integrate_gradients(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Returns integrated gradients for the given [`Instance`](../../data/instance.md)
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[numpy.ndarray] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads = self.predictor.get_gradients([instance], False, False, False, False)[0]
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= torch.Tensor(input_embedding)

        return ig_grads
    def _integrate_gradients2(self, instance: Instance, model: Model) -> Dict[str, numpy.ndarray]:
        """
        Returns integrated gradients for the given [`Instance`](../../data/instance.md)
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[numpy.ndarray] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook_regular(alpha, embeddings_list)

            grads,_ = self.predictor.get_gradients([instance], False, False, False, False, model)
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
                ig_grads = {x:ig_grads[x].cpu() for x in ig_grads}
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key].cpu()

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()
        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= torch.Tensor(input_embedding)

        return ig_grads

    def saliency_interpret_from_instances(self, labeled_instances, embedding_operator, normalization) -> JsonDict:
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self._integrate_gradients(instance)

            final_loss = 0
            rank = None

            # we only handle when we have 1 input at the moment, so this loop does nothing
            for key, grad in grads.items():
                # grads_summed_across_batch = torch.sum(grad, axis=0)
                for idx, gradient in enumerate(grad):
                    # Get rid of embedding dimension
                    summed_across_embedding_dim = None 
                    if embedding_operator == "l1_norm":
                        summed_across_embedding_dim = torch.norm(grads_summed_across_batch, p=1, dim=1)
                    elif embedding_operator == "l2_norm":
                        summed_across_embedding_dim = torch.norm(gradient, dim=1)

                    # Normalize the gradients 
                    normalized_grads = None
                    if normalization == "l2_norm":
                        normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                    elif normalization == "l1_norm":
                        normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

                    # Get the gradient at position of Bob/Joe
                    joe_bob_position = 0 # TODO, hardcoded position

                    # Note we use absolute value of grad here because we only care about magnitude
                    temp = [(idx, numpy.absolute(grad)) for idx, grad in enumerate(normalized_grads.detach().numpy())]
                    temp.sort(key=lambda t: t[1], reverse=True)
                    rank = [i for i, (idx, grad) in enumerate(temp) if idx == joe_bob_position][0]

                    final_loss += normalized_grads[joe_bob_position]
            print("finetuned loss", final_loss)
            if final_loss < 0:
                print("YOOOOOOOOOOOOOOOOOOOOOOO")
            final_loss.requires_grad_()
            return final_loss, rank
