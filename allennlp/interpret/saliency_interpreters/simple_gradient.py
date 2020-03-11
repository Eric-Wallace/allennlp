import math
from typing import List
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util
from allennlp.nn.util import move_to_device

@SaliencyInterpreter.register("simple-gradient")
class SimpleGradient(SaliencyInterpreter):
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
            grads = self.predictor.get_gradients([instance])[0]
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
                emb_grad = numpy.sum(grad[0] * embeddings_list[input_idx], axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def saliency_interpret_from_instance(self, labeled_instances) -> JsonDict:
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
            grads = self.predictor.get_gradients([instance])[0]
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
                emb_grad = numpy.sum(grad[0].detach().numpy() * embeddings_list[input_idx], axis=1)
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

    def sst_interpret_from_instances(self, labeled_instances, embedding_op, normalization, normalization2, softmax, cuda: bool, higher_order_grad: bool):
        # Get raw gradients and outputs
        grads = self.predictor.get_gradients(labeled_instances, cuda, True, True, higher_order_grad)

        norm_grads = []
        raw_grads = []
        for key, grad in grads.items():
            for idx, gradient in enumerate(grad):
                # print("hook gradients", gradient.requires_grad)
                if embedding_op == "dot":
                    summed_across_embedding_dim = self.embedding_dot_product(labeled_instances[idx].fields['tokens'], gradient, cuda)
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

                if softmax:
                    print("Going through softmax ...")
                    softmax = torch.nn.Softmax(dim=0)
                    normalized_grads = softmax(normalized_grads)

                # print('normalized grads', normalized_grads.requires_grad)
                norm_grads.append(normalized_grads)
                raw_grads.append(summed_across_embedding_dim)

        return norm_grads, raw_grads

    def embedding_dot_product(self, tokens, gradient,cuda):
        batch_tokens = tokens.as_tensor(tokens.get_padding_lengths())
        if cuda:
            batch_tokens = move_to_device(batch_tokens,cuda_device=0)
        embeddings = self.predictor._model._text_field_embedder(batch_tokens)
        embeddings = embeddings.squeeze(0).transpose(1,0)
        return torch.diag(torch.mm(gradient, embeddings))

    def saliency_interpret_from_instances(self, labeled_instances, embedding_operator, normalization, normalization2="l1", softmax="False") -> JsonDict:
        # Get raw gradients and outputs
        grads, outputs = self.predictor.get_gradients(labeled_instances)

        final_loss = torch.zeros(grads["grad_input_1"].shape[1])
        rank = None
        joe_bob_position = 0 # TODO, hardcoded position
        softmax = torch.nn.Softmax(dim=0)
        # we only handle when we have 1 input at the moment, so this loop does nothing
        # print(grads.keys())
        for key, grad in grads.items():
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            if key =="grad_input_2":
                continue
            for idx, gradient in enumerate(grad):
                # Get rid of embedding dimension
                summed_across_embedding_dim = None 
                if embedding_operator == "dot_product":
                    batch_tokens = labeled_instances[idx].fields['hypothesis']
                    batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                    print("batch tokens", batch_tokens)
                    embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                    embeddings = embeddings.squeeze(0).transpose(1,0)
                    summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
                elif embedding_operator == "l2_norm":
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)

                # Normalize the gradients 
                normalized_grads = summed_across_embedding_dim
                if normalization == "l2_norm":
                    print("summed_across_embedding_dim",summed_across_embedding_dim.detach().numpy())
                    print("torch.norm(summed_across_embedding_dim)", torch.norm(summed_across_embedding_dim).detach().numpy())
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                    print("normalized_grads",normalized_grads.detach().numpy())
                elif normalization == "l1_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

                if normalization2 == "l2_norm":
                    normalized_grads = normalized_grads**2
                elif normalization2 == "l1_norm":
                    normalized_grads = torch.abs(normalized_grads)

                normalized_grads = torch.nn.utils.rnn.pad_sequence([final_loss, normalized_grads]).transpose(1, 0)[1]
                final_loss += normalized_grads
        final_loss /= grads['grad_input_1'].shape[0]
        
        # L1/L2 norm/sum, -> softmax
        if do_softmax == "True":
            final_loss = softmax(final_loss)
        
         # Note we use absolute value of grad here because we only care about magnitude
        temp = [(idx, numpy.absolute(grad)) for idx, grad in enumerate(final_loss.detach().numpy())]
        temp.sort(key=lambda t: t[1], reverse=True)
        rank = [i for i, (idx, grad) in enumerate(temp) if idx == joe_bob_position][0]
        # print("finetuned loss", final_loss)
        
        final_loss = final_loss
        # print("truncated final_loss", final_loss)
        final_loss.requires_grad_()
        return final_loss, rank

    def match_hooks_autograd(self, labeled_instances, cuda):

        # check that embedding_layer.weight.grad is not None
        self.predictor.get_gradients(labeled_instances, cuda, False, True)

        # Auto grad on, hook grads off
        grads_1, outputs_1, grad_auto_1 = self.predictor.get_gradients(labeled_instances, cuda, True, False)
        
        # Auto grad on, hook grads on
        grads_2, outputs_2, grad_auto_2 = self.predictor.get_gradients(labeled_instances, cuda, True, True)
        
        token_field = labeled_instances[0].fields['tokens']
        tensor_tokens = token_field.as_tensor(token_field.get_padding_lengths())

        grad_auto_1_emb = grad_auto_1[0][tensor_tokens['tokens']]
        grad_auto_2_emb = grad_auto_2[0][tensor_tokens['tokens']]

        return grad_auto_1_emb, grads_2['grad_input_1']
        
    def snli_interpret_from_instances(self, labeled_instances, embedding_operator, loss_function, normalization, normalization2="l1_norm", do_softmax="False", topk=1, cuda=True) -> JsonDict:
        # Get raw gradients and outputs
        grads, outputs, grad_auto = self.predictor.get_gradients(labeled_instances, cuda)

        final_loss = torch.zeros(1)
        if cuda: 
            final_loss = final_loss.cuda()
        softmax = torch.nn.Softmax(dim=0)
        # we only handle when we have 1 input at the moment, so this loop does nothing
        # print(grads.keys())
        embedding_grads = grad_auto[0] # take first element in tuple 
        batch_tokens = labeled_instances[0].fields['hypothesis']
        batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())

        print("embedding grads", embedding_grads[batch_tokens['tokens']])
        return 
        for key, grad in grads.items():
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            if key == "grad_input_2":
                continue
            for idx, gradient in enumerate(grad):
                # Get rid of embedding dimension
                summed_across_embedding_dim = None 
                if embedding_operator == "dot_product":
                    batch_tokens = labeled_instances[idx].fields['hypothesis']
                    batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                    print('batch tokens', batch_tokens["tokens"])
                    print(len(batch_tokens['tokens']))
                    if cuda: 
                        batch_tokens = move_to_device(batch_tokens, cuda_device=0)
                    # Searches gradient embedding matrix for matching gradients 
                    # coming from hooks 
                    # indices = []
                    # for j, g in enumerate(gradient): 
                    #     indices.append([])
                    #     for i, emb_grad in enumerate(embedding_grads):
                    #         if torch.allclose(g, emb_grad, rtol=1e-05, atol=1e-03): 
                    #             indices[j].append(i)

                    print("indices found", indices)
                    print(len(indices))

                    # extracted_embedding_grad = embedding_grads[batch_tokens["tokens"]]
                    # print("embedding grad", extracted_embedding_grad)
                    # print(extracted_embedding_grad.shape)
                    # print("actual grad", gradient)
                    # print(gradient.shape)
                    embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                    embeddings = embeddings.squeeze(0).transpose(1, 0)
                    summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
                elif embedding_operator == "l2_norm":
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)

                # Normalize the gradients 
                normalized_grads = summed_across_embedding_dim
                if normalization == "l2_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                elif normalization == "l1_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

                if normalization2 == "l2_norm":
                    normalized_grads = normalized_grads**2
                elif normalization2 == "l1_norm":
                    normalized_grads = torch.abs(normalized_grads)

                if (torch.sum(torch.isnan(normalized_grads)) != 0):
                    raise Exception('nan values encountered!')

                # print("normalized grads", normalized_grads) 

                max_grad = torch.max(normalized_grads)
                sorted_grads = sorted([(i, grad) for i, grad in enumerate(normalized_grads)], key=lambda t: t[1], reverse=True)
                summed_topk_grads = sum([torch.abs(summed_across_embedding_dim[i]) for i, _ in sorted_grads[:topk]])
                summed_topk_next_grads = sum([torch.abs(summed_across_embedding_dim[i]) for i, _ in sorted_grads[1:topk + 1]])

                with open('snli_max_grads.txt', 'a') as f:
                    f.write("max grad: %f\n" % (max_grad))

                if cuda: 
                    final_loss += loss_function(summed_topk_grads, torch.zeros(1)[0].cuda())
                else: 
                    final_loss += loss_function(summed_topk_grads, torch.zeros(1)[0])
        
        # L1/L2 norm/sum, -> softmax
        if do_softmax == "True":
            final_loss = softmax(final_loss)
        
        print("returning")
        return final_loss 

    def snli_interpret_from_instances2(self, labeled_instances, embedding_operator, loss_function, normalization, normalization2="l1_norm", do_softmax="False", cuda=True) -> JsonDict:
        # Get raw gradients and outputs
        grads, outputs, grad_auto = self.predictor.get_gradients(labeled_instances, cuda)

        final_loss = torch.zeros(grads["grad_input_1"].shape[1])
        if cuda: 
            final_loss = final_loss.cuda()
        softmax = torch.nn.Softmax(dim=0)
        # we only handle when we have 1 input at the moment, so this loop does nothing

        for key, grad in grads.items():
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            if key == "grad_input_2":
                continue
            for idx, gradient in enumerate(grad):
                # Get rid of embedding dimension
                summed_across_embedding_dim = None 
                if embedding_operator == "dot_product":
                    batch_tokens = labeled_instances[idx].fields['hypothesis']
                    batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                    if cuda: 
                        batch_tokens = move_to_device(batch_tokens, cuda_device=0)

                    embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                    embeddings = embeddings.squeeze(0).transpose(1, 0)
                    summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
                elif embedding_operator == "l2_norm":
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)

                # Normalize the gradients 
                normalized_grads = summed_across_embedding_dim
                if normalization == "l2_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                elif normalization == "l1_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

                if normalization2 == "l2_norm":
                    normalized_grads = normalized_grads**2
                elif normalization2 == "l1_norm":
                    normalized_grads = torch.abs(normalized_grads)

                if (torch.sum(torch.isnan(normalized_grads)) != 0):
                    raise Exception('nan values encountered!')

                # print("normalized grads", normalized_grads) 

                normalized_grads = torch.nn.utils.rnn.pad_sequence([final_loss, normalized_grads]).transpose(1, 0)[1]
                final_loss += normalized_grads
                with open("snli_attack_grads.txt", "a") as f:
                    f.write("Grads: %s\n" % (normalized_grads))
        final_loss /= grads['grad_input_1'].shape[0]

        # L1/L2 norm/sum, -> softmax
        if do_softmax == "True":
            final_loss = softmax(final_loss)
        
        # Note we use absolute value of grad here because we only care about magnitude
        temp = [(idx, numpy.absolute(grad)) for idx, grad in enumerate(final_loss.cpu().detach().numpy())]
        temp.sort(key=lambda t: t[1], reverse=True)
        rank = [i for i, (idx, grad) in enumerate(temp) if idx == 0][0]

        with open("snli_attack_rank.txt", "a") as f:
            f.write("Rank: %d\n" % (rank))

        final_loss = final_loss[0]
        targets = torch.ones_like(final_loss)
        print("final_loss", final_loss)
        final_loss = loss_function(final_loss, targets)
        return final_loss