import math
from typing import List
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util
from allennlp.nn.util import move_to_device
import numpy as np
from torch.utils.hooks import RemovableHandle
from torch import Tensor
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
  
    def saliency_interpret_from_instance(self, labeled_instances) -> JsonDict:
        """
        Interprets the model's prediction for inputs.  Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """
        instances_with_grads = dict()
        embeddings_list = []
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
    def saliency_interpret_from_instances(self, labeled_instances, embedding_operator, normalization,normalization2="l1_norm",do_softmax="False") -> JsonDict:
     
        # Get raw gradients and outputs
        grads, outputs,embedding_gradients = self.predictor.get_gradients(labeled_instances,False)

        final_loss = torch.zeros(1)
        ranks = [0] * len(labeled_instances)
        rank = None
        joe_bob_position = 0 # TODO, hardcoded position
        softmax = torch.nn.Softmax(dim=0)
        # we only handle when we have 1 input at the moment, so this loop does nothing
        # print(grads.keys())
        for key, grad in grads.items():
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            for idx, gradient in enumerate(grad):
                # Get rid of embedding dimension
                summed_across_embedding_dim = None 
                if embedding_operator == "dot_product":
                    batch_tokens = labeled_instances[idx].fields['tokens']
                    batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
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

                # normalized_grads = torch.nn.utils.rnn.pad_sequence([final_loss, normalized_grads]).transpose(1, 0)[1]
                temp = [(J, numpy.absolute(grad)) for J, grad in enumerate(normalized_grads.detach().numpy())]
                temp.sort(key=lambda t: t[1], reverse=True)
                rank = [i for i, (J, grad) in enumerate(temp) if J == joe_bob_position][0]
                ranks[idx] = rank
                final_loss += normalized_grads[joe_bob_position]

        final_loss /= grads['grad_input_1'].shape[0]
        print("final Loss", final_loss)
        # L1/L2 norm/sum, -> softmax
        if do_softmax == "True":
            final_loss = softmax(final_loss)
        
        # print("finetuned loss", final_loss)
        
        final_loss.requires_grad_()
        print(ranks)
        return final_loss, np.mean(ranks)
    def saliency_interpret_from_instances_pmi(self, labeled_instances, embedding_operator, normalization,variables,normalization2="l1_norm",do_softmax="False") -> JsonDict:
        # Get raw gradients and outputs
        grads, outputs,embedding_gradients = self.predictor.get_gradients(labeled_instances,True)
        print("embedding_gradients",len(embedding_gradients))
        print("grads",grads)
        final_loss = torch.zeros(1).cuda()
        rank = None
        joe_bob_position = 0 # TODO, hardcoded position
        softmax = torch.nn.Softmax(dim=0)
        # we only handle when we have 1 input at the moment, so this loop does nothing
        # print(grads.keys())
        get_salient_words = variables["fn"]
        get_rank = variables["fn2"]
        lmbda = variables["lmbda"]
        ent100 = variables["ent100"]
        con100 = variables["con100"]
        neu100 = variables["neu100"]
        training_instances = variables["training_instances"]
        for key, grad in grads.items():
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            grad.requires_grad_()
            print(grad.requires_grad)
            if key =="grad_input_2":
                continue
            for idx, gradient in enumerate(grad):
                # 1 Get rid of embedding dimension 
                summed_across_embedding_dim = None 
                if embedding_operator == "dot_product":
                    batch_tokens = labeled_instances[idx].fields['hypothesis']
                    batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                    batch_tokens = move_to_device(batch_tokens, cuda_device=0)
                    print("gradient:",gradient.shape)
                    print("batch_tokens:",batch_tokens)
                    embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                    # print("embeddings",embeddings.cpu().detach().numpy())
                    embeddings = embeddings.squeeze(0).transpose(1,0)
                    summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
                elif embedding_operator == "l2_norm":
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)

                # get grads magnitude
                normalized_grads = torch.abs(summed_across_embedding_dim)
                print("before grads_mag:",summed_across_embedding_dim.cpu().detach().numpy())
                grads_mag = np.abs(summed_across_embedding_dim.cpu().detach().numpy())
                print("grads_mag:",grads_mag)

                
                # 3 Normalize
                # if normalization == "l2_norm":
                #     print("summed_across_embedding_dim",summed_across_embedding_dim.cpu().detach().numpy())
                #     print("torch.norm(summed_across_embedding_dim)", torch.norm(summed_across_embedding_dim).cpu().detach().numpy())
                #     normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                #     print("normalized_grads",normalized_grads.cpu().detach().numpy())
                # elif normalization == "l1_norm":
                #     normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

                # if normalization2 == "l2_norm":
                #     normalized_grads = normalized_grads**2
                # elif normalization2 == "l1_norm":
                #     normalized_grads = torch.abs(normalized_grads)


                # 4 Generate the mask
            
                hyp_grad_rank,hyp_grad_sorted = get_rank(grads_mag)
                print("hyp_grad_rank:",hyp_grad_rank)
                instance = training_instances[idx]
                top100 = get_salient_words(instance,ent100,con100,neu100)
                mask = torch.zeros(grads["grad_input_1"].shape[1]).cuda()
                for i,token in enumerate(instance["hypothesis"]):
                    word = token.text.lower()
                    if word in top100:
                        print(word)
                        if word == "sleeping":
                            batch_tokens = labeled_instances[idx].fields['hypothesis']
                            batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                            batch_tokens = move_to_device(batch_tokens, cuda_device=0)
                            embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                            print("embedding",embeddings.shape)
                            print()
                            with open("sanity_checks/embedding_check.txt", "a") as myfile:
                                line = ""
                                for each in embeddings[i]:
                                    # print(each)
                                    line = line + str(each.cpu().detach().numpy()) +", "
                                myfile.write("%s\n"%(line))
                        top100[word].append((grads_mag[i], hyp_grad_rank[i],grads_mag))
                        if hyp_grad_rank[i] <=4:
                            mask[i] = 1
                # normalized_grads = torch.nn.utils.rnn.pad_sequence([final_loss, normalized_grads]).transpose(1, 0)[1]
                masked_loss = torch.dot(mask,normalized_grads)
                print("mask:",mask)
                print("masked_loss:",masked_loss)
                final_loss += masked_loss
        # final_loss /= grads['grad_input_1'].shape[0]
        
        # L1/L2 norm/sum, -> softmax
        if do_softmax == "True":
            final_loss = softmax(final_loss)
        
         # Note we use absolute value of grad here because we only care about magnitude
        temp = [(idx, numpy.absolute(grad)) for idx, grad in enumerate(final_loss.cpu().detach().numpy())]
        temp.sort(key=lambda t: t[1], reverse=True)
        rank = [i for i, (idx, grad) in enumerate(temp) if idx == joe_bob_position][0]
        # print("finetuned loss", final_loss)
        
        final_loss = final_loss
        # print("truncated final_loss", final_loss)
        final_loss.requires_grad_()
        return final_loss, outputs, grads_mag,gradient,grad,embedding_gradients
    def saliency_interpret_from_instances_autograd(self, labeled_instances, embedding_operator, normalization,variables,normalization2="l1_norm",do_softmax="False",cuda = "True") -> JsonDict:
        # Get raw gradients and outputs
        if cuda == "True":
            grads, outputs,embedding_gradients = self.predictor.get_gradients(labeled_instances,True)
        else:
            grads, outputs,embedding_gradients = self.predictor.get_gradients(labeled_instances,False)
        print("embedding_gradients",len(embedding_gradients[0]))
        # print("grads",grads)
        final_loss = torch.zeros(1)
        if cuda == "True":
            final_loss = final_loss.cuda()
        get_salient_words = variables["fn"]
        get_rank = variables["fn2"]
        lmbda = variables["lmbda"]
        ent100 = variables["ent100"]
        con100 = variables["con100"]
        neu100 = variables["neu100"]
        training_instances = variables["training_instances"]
        embedding_gradients = embedding_gradients[0]
        for idx, _ in enumerate(labeled_instances):
            # 1 Get rid of embedding dimension 
            summed_across_embedding_dim = None 
            if embedding_operator == "dot_product":
                batch_tokens = labeled_instances[idx].fields['hypothesis']
                batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                if cuda == "True":
                    batch_tokens = move_to_device(batch_tokens, cuda_device=0)
                # print("grad shape:",embedding_gradients.shape)
                print("batch_tokens:",batch_tokens)
                # print("my effort",embedding_gradients[batch_tokens["tokens"]].shape)
                # print("my effort",embedding_gradients[batch_tokens["tokens"]])
                extracted_embedding = embedding_gradients[batch_tokens["tokens"]]
                embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                # print("embeddings",embeddings.cpu().detach().numpy())
                embeddings = embeddings.squeeze(0).transpose(1,0)
                # print(embeddings)
                summed_across_embedding_dim = torch.diag(torch.mm(extracted_embedding, embeddings))
            elif embedding_operator == "l2_norm":
                summed_across_embedding_dim = torch.norm(embedding_gradients, dim=1)
            normalized_grads = torch.abs(summed_across_embedding_dim)
            print("before grads_mag:",summed_across_embedding_dim.cpu().detach().numpy())
            grads_mag = normalized_grads.cpu().detach().numpy()
            # 3 Normalize
            # if normalization == "l2_norm":
            #     print("summed_across_embedding_dim",summed_across_embedding_dim.cpu().detach().numpy())
            #     print("torch.norm(summed_across_embedding_dim)", torch.norm(summed_across_embedding_dim).cpu().detach().numpy())
            #     normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
            #     print("normalized_grads",normalized_grads.cpu().detach().numpy())
            # elif normalization == "l1_norm":
            #     normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

            # if normalization2 == "l2_norm":
            #     normalized_grads = normalized_grads**2
            # elif normalization2 == "l1_norm":
            #     normalized_grads = torch.abs(normalized_grads)
            # rank it
            hyp_grad_rank,hyp_grad_sorted = get_rank(grads_mag)
            print("hyp_grad_rank:",hyp_grad_rank)
            instance = training_instances[idx]
            top100 = get_salient_words(instance,ent100,con100,neu100)
            mask = torch.zeros(len(batch_tokens["tokens"]))
            if cuda == "True":
                mask = mask.cuda()
            for i,token in enumerate(instance["hypothesis"]):
                word = token.text.lower()
                if word in top100:
                    print(word)
                    if word == "sleeping":
                        batch_tokens = labeled_instances[idx].fields['hypothesis']
                        batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                        if cuda == "True":
                            batch_tokens = move_to_device(batch_tokens, cuda_device=0)
                        embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                        # print("embedding",embeddings.shape)
                        # print()
                        with open("sanity_checks/embedding_check.txt", "a") as myfile:
                            line = ""
                            for each in embeddings[i]:
                                # print(each)
                                line = line + str(each.cpu().detach().numpy()) +", "
                            myfile.write("%s\n"%(line))
                    top100[word].append((grads_mag[i], hyp_grad_rank[i],grads_mag))
                    if hyp_grad_rank[i] <=4:
                        mask[i] = 1

            masked_loss = torch.dot(mask,normalized_grads)
            print("mask:",mask)
            # print("masked_loss:",masked_loss)
            final_loss += masked_loss
        return final_loss,0,0,0,0,0
