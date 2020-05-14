import math
from typing import List
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn.util import move_to_device
import numpy as np
from torch.utils.hooks import RemovableHandle
from torch import Tensor
from collections import defaultdict
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
            if hasattr(self.predictor._model, 'submodels'):
                grads = dict()
                embeddings_list_1 = []
                handle_1 = self._register_forward_hook(embeddings_list_1, self.predictor._model.submodels[0])
                grad_1 = self.predictor.get_gradients([instance], False, False, False, False, self.predictor._model.submodels[0])
                handle_1.remove()
                print("first_model's grad",grad_1)

                embeddings_list_2 = []
                handle_2 = self._register_forward_hook(embeddings_list_2, self.predictor._model.submodels[1])
                grad_2 = self.predictor.get_gradients([instance], False, False, False, False, self.predictor._model.submodels[1])
                handle_2.remove()
                print("second_model's grad",grad_2)

                embeddings_list_1.reverse()
                embeddings_list_2.reverse()

                for key, grad in grad_1.items():
                    input_idx = int(key[-1]) - 1
                    emb_grad_1 = numpy.sum(grad_1[key][0].numpy() * embeddings_list_1[input_idx], axis=1)
                    print(emb_grad_1)
                    emb_grad_2 = numpy.sum(grad_2[key][0].numpy() * embeddings_list_2[input_idx], axis=1)
                    print(emb_grad_2)
                    emb_grad = emb_grad_1 + emb_grad_2
                    norm = numpy.linalg.norm(emb_grad, ord=1)
                    normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                    grads[key] = normalized_grad
                
            else: 
                # Hook used for saving embeddings
                handle = self._register_forward_hook(embeddings_list, self.predictor._model)
                grads = self.predictor.get_gradients([instance], False, False, False, False, self.predictor._model)
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
        print("saliency_interpret_from_json",instances_with_grads)
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

    def _register_forward_hook(self, embeddings_list: List, model: Model):
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
    def sst_interpret_from_instances(self, labeled_instances, embedding_op, normalization, normalization2, cuda: bool):
        # Get raw gradients and outputs
        embeddings_list = []
        handle = self._register_forward_hook2(embeddings_list)

        grads,_ = self.predictor.get_gradients(labeled_instances, cuda, True, True, False,self.predictor._model)

        norm_grads = []
        raw_grads = []
        for key, grad in grads.items():
            for idx, gradient in enumerate(grad):
                print(gradient.size())
                
                if embedding_op == "dot":
                    input_idx = int(key[-1]) - 1
                    print(embeddings_list[input_idx].size())
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
                print(gradient)
                print(normalized_grads)
                norm_grads.append(normalized_grads)
                raw_grads.append(summed_across_embedding_dim)

        handle.remove()
        return norm_grads, raw_grads
    def saliency_interpret_from_instances_pmi_sst(self, labeled_instances, embedding_operator, normalization,variables,normalization2="l1_norm",do_softmax="False", cuda = "False",autograd="False",all_low='False') -> JsonDict:
        # Get raw gradients and outputs
        use_autograd = True if autograd =="True" else False
        if cuda == "True":
            grads, outputs,embedding_gradients = self.predictor.get_gradients(labeled_instances,True,use_autograd)
        else:
            grads, outputs,embedding_gradients = self.predictor.get_gradients(labeled_instances,False,use_autograd)
        # print("***",embedding_gradients)
        # print("***",grads["grad_input_1"])
        final_loss = torch.zeros(1)
        if cuda == "True":
            final_loss = final_loss.cuda()
        rank = None
        joe_bob_position = 0 # TODO, hardcoded position
        # softmax = torch.nn.Softmax(dim=0)
        # we only handle when we have 1 input at the moment, so this loop does nothing
        get_salient_words = variables["fn"]
        get_rank = variables["fn2"]
        lmbda = variables["lmbda"]
        pos100 = variables["pos100"]
        neg100 = variables["neg100"]
        training_instances = variables["training_instances"]
        highest_grad = 0
        for key, grad in grads.items():
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            grad.requires_grad_()
            if key =="grad_input_2":
                continue
            if use_autograd == True:
                embedding_gradients = embedding_gradients[0]
                grad = list(range(len(labeled_instances)))
            for idx, gradient in enumerate(grad):
                # 1 Get rid of embedding dimension 
                summed_across_embedding_dim = None 
                batch_tokens = labeled_instances[idx].fields['tokens']
                batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                if cuda=="True":
                    batch_tokens = move_to_device(batch_tokens, cuda_device=0)
                print("batch_tokens:",batch_tokens)
                embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                # print("embeddings",embeddings.cpu().detach().numpy())
                embeddings = embeddings.squeeze(0).transpose(1,0)
                if use_autograd == True:
                    gradient = embedding_gradients[batch_tokens["tokens"]]
                    embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                    print("embeddings",embeddings.shape)
                    if embeddings.shape[0]==1:
                        embeddings = embeddings.transpose(1,0)
                    else:
                        embeddings = embeddings.squeeze(0).transpose(1,0)
                if embedding_operator == "dot_product":
                    summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
                elif embedding_operator == "l2_norm":
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)
                # get grads magnitude
                normalized_grads = torch.abs(summed_across_embedding_dim)
                # print("before grads_mag:",summed_across_embedding_dim.cpu().detach().numpy())
                grads_mag = np.abs(summed_across_embedding_dim.cpu().detach().numpy())
                print("grads_mag:",grads_mag)
                highest_grad += np.max(grads_mag)
                # 3 Normalize
                # 4 Generate the mask
            
                hyp_grad_rank,hyp_grad_sorted = get_rank(grads_mag)
                print("hyp_grad_rank:",hyp_grad_rank)
                instance = training_instances[idx]
                top100 = get_salient_words(instance,pos100,neg100)
                mask = torch.zeros(batch_tokens["tokens"].shape[0])
                if cuda=="True":
                    mask = torch.zeros(grads['grad_input_1'].shape[1])
                    mask = mask.cuda()
                else:
                    mask = torch.zeros(batch_tokens["tokens"].shape[0])
                for i,token in enumerate(instance["tokens"]):
                    word = token.text.lower()
                    if word in top100:
                        print(word)
                        if word == "sleeping":
                            batch_tokens = labeled_instances[idx].fields['tokens']
                            batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                            if cuda=="True":
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
                        # if hyp_grad_rank[i] <=4:
                        mask[i] = 1
                # normalized_grads = torch.nn.utils.rnn.pad_sequence([final_loss, normalized_grads]).transpose(1, 0)[1]
                if all_low =="True":
                    # masked_loss = 0.5*torch.sum(summed_across_embedding_dim**2)
                    masked_loss = torch.sum(normalized_grads)
                else:
                    masked_loss = torch.dot(mask,normalized_grads)
                print("mask:",mask)
                print("masked_loss:",masked_loss)
                final_loss += masked_loss
        final_loss /= len(labeled_instances)
        highest_grad /= grads["grad_input_1"].shape[0]
        # L1/L2 norm/sum, -> softmax
        # if do_softmax == "True":
        #     final_loss = softmax(final_loss)
        
        # Note we use absolute value of grad here because we only care about magnitude
        temp = [(idx, numpy.absolute(grad)) for idx, grad in enumerate(final_loss.cpu().detach().numpy())]
        temp.sort(key=lambda t: t[1], reverse=True)
        rank = [i for i, (idx, grad) in enumerate(temp) if idx == joe_bob_position][0]
        # final_loss.requires_grad_()
        print(final_loss)
        return final_loss, outputs, grads_mag,gradient,grad,embedding_gradients, highest_grad
    def saliency_interpret_from_instances_2_model_sst(self, labeled_instances, embedding_operator, normalization,normalization2="l1_norm",do_softmax="False", cuda = "False",autograd="False",all_low='False',bert=False,recording=False) -> JsonDict:
        # Get raw gradients and outputs
        use_autograd = True if autograd =="True" else False
        token_id_name = "bert" if bert else "tokens"
        if cuda == "True":
            embedding_gradients = self.predictor.get_gradients(labeled_instances,True,use_autograd,bert=bert,recording=recording)
        else:
            embedding_gradients = self.predictor.get_gradients(labeled_instances,False,use_autograd,bert=bert,recording=recording)
        # print("***",embedding_gradients)
        # print("***",grads["grad_input_1"].size())
        final_loss = torch.zeros(1)
        if cuda == "True":
            final_loss = final_loss.cuda()
        # softmax = torch.nn.Softmax(dim=0)
        # final_loss = torch.norm(torch.sum(torch.abs(grads["grad_input_1"]), dim=1), dim=0)[0] 
        # return final_loss
        highest_grad = 0
        mean_grad = 0
        grad_mags = []
        instances = list(range(len(labeled_instances)))
        if use_autograd == True:
            embedding_gradients = embedding_gradients[0]
        for idx, grad in enumerate(instances):
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            # grad.requires_grad_()

            if not use_autograd:
                gradient = embedding_gradients["grad_input_1"][idx]
            # print(embedding_gradients.size())
            # 1 Get the tokens for each instance
            summed_across_embedding_dim = None 
            # 3 Get rid of gradients dimensions
            batch_tokens = labeled_instances[idx].fields['tokens']
            length = len(batch_tokens)
            batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
            if cuda=="True":
                batch_tokens = move_to_device(batch_tokens, cuda_device=0)
            # print("batch_tokens:",batch_tokens)
            if not bert:
                embeddings = self.predictor._model._text_field_embedder(batch_tokens)
            else:
                embeddings = self.predictor._model.bert_model.embeddings.word_embeddings(batch_tokens[token_id_name])
            embeddings = embeddings.squeeze(0).transpose(1,0)
            if use_autograd == True:
                gradient = embedding_gradients[batch_tokens[token_id_name]]
                if bert:
                    embeddings = self.predictor._model.bert_model.embeddings.word_embeddings(batch_tokens[token_id_name])
                else:
                    embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                # print("embeddings",embeddings.shape)
                if embeddings.shape[0]==1:
                    embeddings = embeddings.transpose(1,0)
                else:
                    embeddings = embeddings.squeeze(0).transpose(1,0)
            # print(gradient.size(),embeddings.size())
            if embedding_operator == "dot_product":
                summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
            elif embedding_operator == "l2_norm":
                summed_across_embedding_dim = torch.norm(gradient, dim=1)
            # 4 normalization
            if normalization == "l1_norm":
                normalized_grads = torch.abs(summed_across_embedding_dim)
            grads_mag = np.abs(summed_across_embedding_dim.cpu().detach().numpy())
            grad_mags.append(grads_mag[:length])
            cur_max = np.max(grads_mag)
            highest_grad = np.max([cur_max,highest_grad])
            mean_grad += np.mean(grads_mag[:length])
            # if do_softmax == "True":
            #     print(normalized_grads.size())
            #     normalized_grads = softmax(normalized_grads)  
            #     normalized_grads = torch.max(normalized_grads)  
                # print("softmaxed grads",normalized_grads)
                # masked_loss = 0.5*torch.sum(summed_across_embedding_dim**2)
            if all_low == "True":
                masked_loss = torch.sum(normalized_grads)
            else:
                masked_loss = normalized_grads[0]
            # print("masked_loss:",masked_loss)
            final_loss += masked_loss
        final_loss /= len(labeled_instances)
        mean_grad /= len(labeled_instances)
        del embedding_gradients
        torch.cuda.empty_cache()

        print(final_loss)
        return final_loss, grad_mags, highest_grad, mean_grad
    def saliency_interpret_autograd(self, labeled_instances, embedding_operator, normalization,normalization2="l1_norm",do_softmax="False", cuda = "False",autograd="False",all_low='False',importance="first_tok",stopwords=None,stop_ids=None,bert=False,recording=False) -> JsonDict:
        # Get raw gradients and outputs
        use_autograd = True if autograd =="True" else False
        token_id_name = "tokens" if bert else "tokens"
        if cuda == "True":
            embedding_gradients = self.predictor.get_gradients_autograd(labeled_instances,True,bert=bert,recording = recording)
        else:
            embedding_gradients = self.predictor.get_gradients_autograd(labeled_instances,False,bert=bert,recording=recording)
        embedding_gradients = embedding_gradients[0]
        instances = list(range(len(labeled_instances)))
        if all_low == "True":
            final_loss = torch.zeros(embedding_gradients.size(1))
            if cuda == "True":
                final_loss = final_loss.cuda()
        else:
            final_loss = torch.zeros(1)
            # final_loss = torch.zeros(embedding_gradients.size(1))
            if cuda == "True":
                final_loss = final_loss.cuda()
        print(final_loss.size(0))
        highest_grad = 0
        mean_grad = 0
        grad_mags = []
        # word_count = defaultdict(int)
        # for idx, gradient in enumerate(instances):
        #     batch_tokens = labeled_instances[idx].fields['tokens']
        #     print()
        #     print(idx)
        #     print(batch_tokens)
        #     batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
        #     print(batch_tokens)
        #     token_arr = batch_tokens["bert"].detach().numpy()
        #     for z in range(len(token_arr)):
        #         word_count[token_arr[z]] += 1
        # print(word_count)
        appeared_tokens = set()
        for idx, gradient in enumerate(instances):
                # 1 Get the tokens for each instance
                summed_across_embedding_dim = None 
                batch_tokens = labeled_instances[idx].fields['tokens']
                batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                if cuda=="True":
                    batch_tokens = move_to_device(batch_tokens, cuda_device=0)
                # print(labeled_instances[idx].fields['tokens'].get_padding_lengths())
                # print(batch_tokens)
                length = batch_tokens[token_id_name]["token_ids"].size(0)

                # gradient = embedding_gradients[batch_tokens[token_id_name]]
                gradient = embedding_gradients[idx]
                embeddings = self.predictor._model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings.word_embeddings(batch_tokens[token_id_name]["token_ids"])
                if embeddings.shape[0]==1:
                    embeddings = embeddings.transpose(1,0)
                else:
                    embeddings = embeddings.squeeze(0).transpose(1,0)
                if embedding_operator == "dot_product":
                    summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
                elif embedding_operator == "l2_norm":
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)

                # 4 normalization
                summed_across_embedding_dim = torch.abs(summed_across_embedding_dim)
                normalized_grads = summed_across_embedding_dim
                if normalization == "l1_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)
                elif normalization == "l2_norm":
                    # qn = torch.norm(summed_across_embedding_dim, dim=0).detach()
                    # normalized_grads = summed_across_embedding_dim.div(qn.expand_as(summed_across_embedding_dim))
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                grads_mag = np.abs(summed_across_embedding_dim.cpu().detach().numpy())
                grad_mags.append(grads_mag[:length])
                cur_max = np.max(grads_mag)
                highest_grad = np.max([cur_max,highest_grad])
                mean_grad += np.mean(grads_mag[:length])
                # mask = torch.zeros(batch_tokens["bert"].shape[0]).cuda()
                # for z in range(len(token_arr)):
                #     if token_arr[z] in appeared_tokens:
                #         mask[z] = 1
                #     else:
                #         appeared_tokens.add(token_arr[z])
                if all_low == "True":
                    masked_loss = normalized_grads
                    # masked_loss = torch.sum(normalized_grads)
                    # masked_loss = torch.dot(mask,normalized_grads)
                    final_loss[:length] += masked_loss
                else:
                    if importance == "stop_token":
                        # grad_val_1 = torch.sum(torch.abs(grad[stop_ids[idx]])).unsqueeze(0)
                        print("simple_grad",stop_ids[idx])
                        # print(normalized_grads[stop_ids[idx]])
                        print(normalized_grads)
                        masked_loss = torch.sum(normalized_grads[stop_ids[idx]])
                        # print(masked_loss)
                    else:
                        masked_loss = normalized_grads[1]
                        # masked_loss = normalized_grads
                        final_loss += masked_loss
                # final_loss[:length] += masked_loss
                
        final_loss /= len(labeled_instances)
        mean_grad /= len(labeled_instances)
        return final_loss, grad_mags, highest_grad, mean_grad
