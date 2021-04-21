#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json

import torch
import torch.nn as nn
from transformers.modeling_bert import BertEncoder, BertPreTrainedModel
from transformers import PretrainedConfig
from torch.utils.data import Dataset
import numpy as np

from utils import form_tree_mask, form_spatial_mask


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
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_answer_tid, all_tag_depth,
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

    def __init__(self, *tensors, gat_mask=None, base_index=None, tag2tok=None, shape=None, training=True,
                 page_id=None, mask_method=1, mask_dir=None, separate=False):
        tensors = tuple(tensor for tensor in tensors if tensor is not None)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.gat_mask = gat_mask
        self.base_index = base_index
        self.tag2tok = tag2tok
        self.shape = shape
        self.training = training
        self.page_id = page_id
        self.mask_method = mask_method
        self.mask_dir = mask_dir
        self._init_spatial_mask()
        self.separate = separate

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]

        tag_to_token_index = self.tag2tok[index]
        app_tags = self.gat_mask[0][index]
        example_index = self.gat_mask[1][index]
        html_tree = self.gat_mask[2][example_index]
        base = self.base_index[index]

        if self.spatial_mask is not None and self.mask_method < 2:
            spa_mask = torch.zeros((4, self.shape[0], self.shape[0]), dtype=torch.long)
            spa_mask[:, :base, :base + len(app_tags) + 1] = 1
            spa_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
            temp = form_spatial_mask(app_tags, self.spatial_mask[self.page_id[index]])
            spa_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp)
            output.append(spa_mask)

        gat_mask = torch.zeros((1, self.shape[0], self.shape[0]), dtype=torch.long)
        gat_mask[:, :base, :base + len(app_tags) + 1] = 1
        gat_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
        temp = torch.tensor(form_tree_mask(app_tags, html_tree, separate=self.separate,
                                           accelerate=self.mask_method != 3))
        if self.separate:  # TODO multiple mask and variable total head number implementation
            gat_mask = gat_mask.repeat(2, 1, 1)
            gat_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0:2])
            tree_children = torch.tensor(temp[2])
        else:
            gat_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0])
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

    def _init_spatial_mask(self):
        if self.mask_dir is None:
            self.spatial_mask = None
            return
        self.spatial_mask = {}
        for d, _, fs in os.walk(self.mask_dir):
            for f in fs:
                if not f.endswith('.spatial.json'):
                    continue
                domain = d.split('/')[-3][:2]
                page = f.split('.')[0]
                self.spatial_mask[domain + page] = json.load(open(os.path.join(d, f)))
        return


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
        self.mask_method = args.mask_method


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
        self.mask_method = config.mask_method
        if config.model_type == 'bert':
            self.ptm = PTMForQA.bert
        elif config.model_type == 'electra':
            self.ptm = PTMForQA.electra
        else:
            raise NotImplementedError()
        self.link = Link(self.method, config)
        self.num_gat_layers = config.num_hidden_layers
        self.gat = BertEncoder(config)
        self.gat_outputs = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            gat_mask=None,
            spa_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            answer_tid=None,
            tag_to_tok=None,
            tag_depth=None,
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

        gat_inputs = self.link(sequence_output, tag_to_tok, tag_depth)
        if self.mask_method == 0:
            spa_mask = spa_mask.repeat(1, 2, 1, 1)
            if gat_mask.size(1) == 1:
                gat_mask = gat_mask.repeat(1, 4, 1, 1)
            else:
                gat_mask = gat_mask.repeat(1, 2, 1, 1)
            gat_mask = torch.cat([gat_mask, spa_mask], dim=1)
        elif self.mask_method == 1:
            gat_mask = spa_mask.repeat(1, 3, 1, 1)
        elif gat_mask.size(1) != 1:
            gat_mask = gat_mask.repeat(1, 6, 1, 1)
        if head_mask is None:
            head_mask = [None] * self.num_gat_layers
        extended_gat_mask = convert_mask_to_reality(gat_mask)
        gat_outputs = self.gat(gat_inputs, attention_mask=extended_gat_mask, head_mask=head_mask)
        final_outputs = gat_outputs[0]
        tag_logits = self.gat_outputs(final_outputs)
        tag_logits = tag_logits.squeeze(-1)
        if 'multi' in self.loss_method:
            tag_prob = nn.functional.sigmoid(tag_logits)
            outputs = (tag_prob,) + outputs
        else:
            outputs = (tag_logits,) + outputs

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
            elif 'multi' in self.loss_method:
                answer_tid.clamp_(0, 1)
                tag_logits = nn.functional.sigmoid(tag_logits)
                loss_fct = torch.nn.MSELoss()
            else:
                raise NotImplementedError('Loss method {} is not implemented yet'.format(self.loss_method))
            loss = loss_fct(tag_logits, answer_tid)
            outputs = (loss,) + outputs

        return outputs
        # (loss), tag_logits/probs, (hidden_states), (attentions)


