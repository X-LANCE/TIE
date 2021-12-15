#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
from copy import deepcopy

import torch
import torch.nn as nn
from transformers.modeling_bert import BertEncoder, BertPreTrainedModel
from transformers import PretrainedConfig
from torch.utils.data import Dataset
import numpy as np

from utils import form_tree_mask


class SubDataset(Dataset):
    def __init__(self, examples, evaluate, total, base_name, num_parts, shape, loss_method, separate):
        self.examples = examples
        self.evaluate = evaluate
        self.total = total
        self.base_name = base_name
        self.num_parts = num_parts
        self.shape = shape
        self.loss_method = loss_method
        self.separate = separate

        self.current_part = 1
        self.passed_index = 0
        self.dataset = None
        self.all_html_trees = [e.html_tree for e in self.examples]
        self._read_data()

    def _read_data(self):
        del self.dataset
        features = torch.load('{}_sub{}'.format(self.base_name, self.current_part))
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_tag_depth = torch.tensor([f.depth for f in features], dtype=torch.long)
        all_app_tags = [f.app_tags for f in features]
        all_example_index = [f.example_index for f in features]
        all_base_index = [f.base_index for f in features]
        all_tag_to_token = [f.tag_to_token_index for f in features]

        if self.evaluate:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                                        all_tag_depth, gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index, tag2tok=all_tag_to_token, shape=self.shape,
                                        training=False, separate=self.separate)
        else:
            all_answer_tid = torch.tensor([f.answer_tid for f in features],
                                          dtype=torch.long if self.loss_method != 'soft' else torch.float)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids,
                                        all_answer_tid, all_start_positions, all_end_positions, all_tag_depth,
                                        gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index, tag2tok=all_tag_to_token, shape=self.shape,
                                        training=True, separate=self.separate)

    def __getitem__(self, index):
        if index - self.passed_index < 0:
            self.passed_index = 0
            self.current_part = 1
            self._read_data()
        elif index - self.passed_index >= len(self.dataset):
            self.passed_index += len(self.dataset)
            self.current_part += 1
            self._read_data()
        return self.dataset[index - self.passed_index]

    def __len__(self):
        return self.total


class StrucDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, gat_mask=None, base_index=None, tag2tok=None,
                 shape=None, training=True, separate=False):
        tensors = tuple(tensor for tensor in tensors if tensor is not None)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.gat_mask = gat_mask
        self.base_index = base_index
        self.tag2tok = tag2tok
        self.shape = shape
        self.training = training
        self.separate = separate

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]

        tag_to_token_index = self.tag2tok[index]
        app_tags = self.gat_mask[0][index]
        example_index = self.gat_mask[1][index]
        html_tree = self.gat_mask[2][example_index]
        base = self.base_index[index]
        output.append(base)

        gat_mask = torch.zeros((self.shape[0], self.shape[0]), dtype=torch.long)
        gat_mask[:, :len(tag_to_token_index)] = 1
        temp = torch.tensor(form_tree_mask(app_tags, html_tree, separate=self.separate))
        if self.separate:  # TODO multiple mask and variable total head number implementation
            gat_mask = gat_mask.unsqueeze(0).repeat(2, 1, 1)
            gat_mask[0, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0])
            gat_mask[1, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[1])
            gat_mask = gat_mask.repeat(6, 1, 1)
            tree_children = torch.tensor(temp[2])
        else:
            gat_mask[base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0])
            tree_children = torch.tensor(temp[1])
        output.append(gat_mask)

        children = torch.zeros((self.shape[0], self.shape[0]), dtype=torch.long)
        tree_children[tree_children.sum(dim=1) == 0, 0] = 1
        children[base:base + len(app_tags), base:base + len(app_tags)] = tree_children
        children[base, 0] = 1
        output.append(children)

        pooling_matrix = np.zeros(self.shape, dtype=np.double)
        for i in range(len(tag_to_token_index)):
            temp = tag_to_token_index[i]
            pooling_matrix[i][temp[0]: temp[1] + 1] = 1 / (temp[1] - temp[0] + 1)
        pooling_matrix = torch.tensor(pooling_matrix, dtype=torch.float)
        output.append(pooling_matrix)

        return tuple(item for item in output)

    def __len__(self):
        return len(self.tensors[0])


class GraphHtmlConfig(PretrainedConfig):
    def __init__(self,
                 args,
                 **kwargs):
        self.hidden_size = kwargs.pop("hidden_size", None)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", None)
        super().__init__(**kwargs)
        self.method = args.method
        self.model_type = args.model_type
        self.loss_method = args.loss_method
        self.num_hidden_layers = args.num_node_block
        self.max_depth_embeddings = args.max_depth_embeddings


class Link(nn.Module):
    def __init__(self, method, config):
        super().__init__()
        self.method = method
        self.loss_method = config.loss_method
        self.add_position_embeddings = True if config.max_depth_embeddings is not None else False
        self.sequential_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("sequential_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if self.add_position_embeddings:
            self.depth_embeddings = nn.Embedding(config.max_depth_embeddings, config.hidden_size, padding_idx=0)

    def forward(self, inputs, tag_to_token, tag_depth):
        assert tag_to_token.dim() == 3
        modified_tag2token = self.deduce_direct_string(tag_to_token)
        outputs = torch.matmul(modified_tag2token, inputs)

        sequential_ids = self.sequential_ids[:, :inputs.size(1)]
        sequential_embeddings = self.sequential_embeddings(sequential_ids)
        outputs = outputs + sequential_embeddings
        if self.add_position_embeddings:
            depth_embeddings = self.depth_embeddings(tag_depth)
            outputs = outputs + depth_embeddings

        return outputs  # , self.deduce_child(gat_mask)

    def deduce_direct_string(self, tag_to_token):
        if self.method != 'init_direct':
            return tag_to_token
        temp = torch.zeros_like(tag_to_token)
        temp[tag_to_token > 0] = 1
        for i in range(tag_to_token.size(1)):
            temp[:, i] -= temp[:, i + 1:].sum(dim=1)
        temp[temp <= 0] = 0.
        for i in range(tag_to_token.size(1)):
            s = temp[:, i].sum(dim=1, keepdim=True)
            s[s == 0] = 1
            temp[:, i] /= s
        return temp

    # def deduce_child(self, gat_mask):
    #     if self.loss_method != 'hierarchy':
    #         return None
    #     assert gat_mask.dim() == 3
    #     child = deepcopy(gat_mask)
    #     l = gat_mask.size(1)
    #     for i in range(l):
    #         child[:, i, i] = 0
    #     for i in range(l):
    #         for j in range(i + 1, l):
    #             temp = child[:, j, j].unsqueeze(dim=1) * child[:, j]
    #             child[:, i] = (child[:, i] - temp) > 0
    #     return child


def convert_mask_to_reality(mask, dtype=torch.float):
    mask = mask.to(dtype=dtype)
    mask = (1.0 - mask) * -10000.0
    return mask


class GraphHtmlBert(BertPreTrainedModel):
    def __init__(self, PTMForQA, config: GraphHtmlConfig):
        super(GraphHtmlBert, self).__init__(config)
        self.method = config.method
        self.base_type = config.model_type
        self.loss_method = config.loss_method
        if config.model_type == 'bert':
            self.ptm = PTMForQA.bert
        elif config.model_type == 'albert':
            self.ptm = PTMForQA.albert
        elif config.model_type == 'electra':
            self.ptm = PTMForQA.electra
        else:
            raise NotImplementedError()
        self.link = Link(self.method, config)
        self.num_gat_layers = config.num_hidden_layers
        self.gat = BertEncoder(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.hidden_size = config.hidden_size
        self.scale = math.sqrt(self.hidden_size // 2)
        if self.loss_method == 'hierarchy':
            self.gat_outputs = nn.Linear(config.hidden_size, config.hidden_size)
            self.stop_margin = torch.nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        else:
            self.gat_outputs = nn.Linear(config.hidden_size, 1)
            self.stop_margin = None

    def forward(
            self,
            input_ids,
            attention_mask=None,
            gat_mask=None,
            children=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_tid=None,
            tag_to_tok=None,
            tag_depth=None,
            base_ind=None,
    ):

        outputs = self.ptm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        outputs = outputs[2:]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,) + outputs

        gat_inputs = self.link(sequence_output, tag_to_tok, tag_depth)
        if head_mask is None:
            head_mask = [None] * self.num_gat_layers
        if gat_mask.dim() == 3:
            extended_gat_mask = gat_mask[:, None, :, :]
        elif gat_mask.dim() == 4:
            extended_gat_mask = gat_mask
        else:
            raise ValueError('Wrong dim num for gat_mask, whose size is {}'.format(gat_mask.size()))
        extended_gat_mask = convert_mask_to_reality(extended_gat_mask)
        gat_outputs = self.gat(gat_inputs, attention_mask=extended_gat_mask, head_mask=head_mask)
        final_outputs = gat_outputs[0]
        # if self.loss_method == 'hierarchy':
        #     for ind in range(final_outputs.size(0)):
        #         question_emb = final_outputs[ind, 1:base_ind[ind] - 1, :].mean(dim=0, keepdim=True)
        #         final_outputs[ind] = final_outputs[ind] + question_emb
        tag_logits = self.gat_outputs(final_outputs)
        tag_logits = tag_logits.squeeze(-1)
        if self.loss_method == 'hierarchy':
            tag_logits = torch.matmul(tag_logits[:, :, :self.hidden_size // 2],
                                      tag_logits[:, :, self.hidden_size // 2:].permute(0, 2, 1))
            tag_logits = tag_logits / self.scale
            children = convert_mask_to_reality(children)
            tag_logits = tag_logits + children
            b, t, _ = tag_logits.size()
            tag_logits = torch.cat([tag_logits,
                                    self.stop_margin.unsqueeze(0).unsqueeze(0).repeat((b, t, 1))], dim=2)
            tag_probs = torch.softmax(tag_logits, dim=2)
            prob, index = tag_probs.max(dim=2)
            outputs = (tag_probs, prob, index,) + outputs
        elif 'multi' in self.loss_method:
            tag_prob = nn.functional.sigmoid(tag_logits)
            outputs = (tag_prob,) + outputs
        else:
            outputs = (tag_logits,) + outputs

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        if answer_tid is not None:
            if len(answer_tid.size()) > 1:
                answer_tid = answer_tid.squeeze(-1)
            if self.loss_method == 'base':
                ignored_index = tag_logits.size(1)
                answer_tid.clamp_(0, ignored_index)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            elif self.loss_method == 'soft':
                answer_tid.clamp_(0, 1)
                tag_logits = torch.nn.functional.log_softmax(tag_logits, dim=1)
                loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
            elif self.loss_method == 'hierarchy':
                answer_tid.clamp_(-1, tag_logits.size(1))
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                b, t = answer_tid.size()
                tag_logits = tag_logits.reshape((b * t, -1))
                answer_tid = answer_tid.reshape((b * t))
            elif 'multi' in self.loss_method:
                answer_tid.clamp_(0, 1)
                tag_logits = nn.functional.sigmoid(tag_logits)
                loss_fct = torch.nn.MSELoss()
            else:
                raise NotImplementedError('Loss method {} is not implemented yet'.format(self.loss_method))
            loss = loss_fct(tag_logits, answer_tid)
            outputs = (loss,) + outputs

        return outputs
        # (loss), (total_loss), tag_logits, (prob, index), start_logits, end_logits, (hidden_states), (attentions)


