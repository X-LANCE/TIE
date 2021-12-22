#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertLayer, BertAttention
from transformers.modeling_outputs import BaseModelOutput
from transformers import PretrainedConfig
from torch.utils.data import Dataset
import numpy as np

from utils import form_tree_mask, form_spatial_mask


class SubDataset(Dataset):
    def __init__(self, examples, evaluate, total, base_name, num_parts, args):
        self.examples = examples
        self.evaluate = evaluate
        self.total = total
        self.base_name = base_name
        self.num_parts = num_parts
        self.input_file = args.predict_file if evaluate else args.train_file
        self.args = args

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
        all_tag_lists = torch.tensor([f.tag_list for f in features], dtype=torch.long)
        all_example_index = [f.example_index for f in features]
        all_base_index = [f.base_index for f in features]
        all_tag_to_token = [f.tag_to_token_index for f in features]
        all_page_id = [f.page_id for f in features]

        if self.args.model_type == 'markuplm':
            all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in features], dtype=torch.long)
            all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in features], dtype=torch.long)
        else:
            all_xpath_tags_seq, all_xpath_subs_seq = None, None

        if self.evaluate:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                                        all_tag_depth, all_xpath_tags_seq, all_xpath_subs_seq,
                                        tag_list=all_tag_lists,
                                        gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index,
                                        tag2tok=all_tag_to_token,
                                        shape=(self.args.max_tag_length, self.args.max_seq_length),
                                        training=False, page_id=all_page_id, mask_method=self.args.mask_method,
                                        mask_dir=os.path.dirname(self.input_file) if self.args.mask_method < 2 else None,
                                        separate=self.args.separate_mask, cnn_feature_dir=self.args.cnn_feature_dir,
                                        direction=self.args.direction)
        else:
            all_answer_tid = torch.tensor([f.answer_tid for f in features],
                                          dtype=torch.long if self.args.loss_method == 'base' else torch.float)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_answer_tid, all_tag_depth,
                                        all_xpath_tags_seq, all_xpath_subs_seq,
                                        tag_list=all_tag_lists,
                                        gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index,
                                        tag2tok=all_tag_to_token,
                                        shape=(self.args.max_tag_length, self.args.max_seq_length),
                                        training=True, page_id=all_page_id, mask_method=self.args.mask_method,
                                        mask_dir=os.path.dirname(self.input_file) if self.args.mask_method < 2 else None,
                                        separate=self.args.separate_mask, cnn_feature_dir=self.args.cnn_feature_dir,
                                        direction=self.args.direction)

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


class BaseDataset(Dataset):
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

    def _init_cnn_feature(self):
        if self.cnn_feature_dir is None:
            self.cnn_feature = None
            print('None!')
            return
        self.cnn_feature = {}
        cnn_feature_dir = os.walk(self.cnn_feature_dir)
        for d, _, fs in cnn_feature_dir:
            for f in fs:
                if f.split('.')[-1] != 'npy':
                    continue
                domain = d.split('/')[-3][:2]
                page = f.split('.')[0]
                temp = torch.as_tensor(np.load(os.path.join(d, f)), dtype=torch.float)
                self.cnn_feature[domain + page] = torch.cat([temp, torch.zeros_like(temp[0]).unsqueeze(0)], dim=0)
        print(len(self.cnn_feature))
        return


class StrucDataset(BaseDataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, tag_list=None, gat_mask=None, base_index=None, tag2tok=None, shape=None, training=True,
                 page_id=None, mask_method=1, mask_dir=None, direction=None, separate=False, cnn_feature_dir=None):
        tensors = tuple(tensor for tensor in tensors if tensor is not None)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.tag_list = tag_list
        self.gat_mask = gat_mask
        self.base_index = base_index
        self.tag2tok = tag2tok
        self.shape = shape
        self.training = training
        self.page_id = page_id
        self.mask_method = mask_method
        self.mask_dir = mask_dir
        self.direction = direction
        self.separate = separate
        self.cnn_feature_dir = cnn_feature_dir
        self._init_spatial_mask()
        self._init_cnn_feature()

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]

        tag_to_token_index = self.tag2tok[index]
        app_tags = self.gat_mask[0][index]
        example_index = self.gat_mask[1][index]
        html_tree = self.gat_mask[2][example_index]
        base = self.base_index[index]

        if self.cnn_feature is not None:
            page_id, ind = self.page_id[index], self.tag_list[index]
            raw_cnn_feature = self.cnn_feature[page_id]
            assert ind.dim() == 1
            cnn_num, cnn_dim = raw_cnn_feature.size()
            ind[ind >= cnn_num] = cnn_num - 1
            ind[ind == -1] = cnn_num - 1
            ind = ind.unsqueeze(1).repeat([1, cnn_dim])
            cnn_feature = torch.gather(raw_cnn_feature, 0, ind)
            output.append(cnn_feature)

        if self.spatial_mask is not None and self.mask_method < 2:
            if self.direction == 'b':
                spa_mask = torch.zeros((4, self.shape[0], self.shape[0]), dtype=torch.long)
            else:
                spa_mask = torch.zeros((2, self.shape[0], self.shape[0]), dtype=torch.long)
            spa_mask[:, :base, :base + len(app_tags) + 1] = 1
            spa_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
            temp = form_spatial_mask(app_tags, self.spatial_mask[self.page_id[index]], self.direction)
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


class SliceDataset(BaseDataset):
    def __init__(self, file, offsets, html_trees=None, shape=None, training=True, mask_method=1, mask_dir=None,
                 direction=None, separate=False, cnn_feature_dir=None, loss_method='multi-soft'):
        self.file = open(file)
        self.offsets = offsets
        self.html_trees = html_trees
        self.shape = shape
        self.training = training
        self.mask_method = mask_method
        self.mask_dir = mask_dir
        self.direction = direction
        self.separate = separate
        self.cnn_feature_dir = cnn_feature_dir
        self.loss_method = loss_method
        self._init_spatial_mask()
        self._init_cnn_feature()
        if self.training:
            self.tensor_keys = ['input_ids', 'input_mask', 'segment_ids', 'answer_tid',
                                'depth', 'xpath_tags_seq', 'xpath_subs_seq']
        else:
            self.tensor_keys = ['input_ids', 'input_mask', 'segment_ids', 'feature_index',
                                'depth', 'xpath_tags_seq', 'xpath_subs_seq']

    # noinspection PyTypeChecker
    def __getitem__(self, index):
        anchor = self.offsets[index]
        self.file.seek(anchor, 0)
        feature = json.loads(self.file.readline())

        output = []
        for k in self.tensor_keys:
            if k == 'feature_index':
                output.append(torch.tensor(index, dtype=torch.long))
            elif k == 'answer_tid':
                output.append(torch.tensor(feature[k], dtype=torch.long if 'base' in self.loss_method else torch.float))
            else:
                output.append(torch.tensor(feature[k], dtype=torch.long))

        tag_to_token_index = feature['tag_to_token_index']
        app_tags = feature['app_tags']
        example_index = feature['example_index']
        html_tree = self.html_trees[example_index]
        base = feature['base_index']

        if self.cnn_feature is not None:
            page_id, ind = feature['page_id'], torch.tensor(feature["tag_list"], dtype=torch.long)
            raw_cnn_feature = self.cnn_feature[page_id]
            assert ind.dim() == 1
            cnn_num, cnn_dim = raw_cnn_feature.size()
            ind[ind >= cnn_num] = cnn_num - 1
            ind[ind == -1] = cnn_num - 1
            ind = ind.unsqueeze(1).repeat([1, cnn_dim])
            cnn_feature = torch.gather(raw_cnn_feature, 0, ind)
            output.append(cnn_feature)

        if self.spatial_mask is not None and self.mask_method < 2:
            if self.direction == 'b':
                spa_mask = torch.zeros((4, self.shape[0], self.shape[0]), dtype=torch.long)
            else:
                spa_mask = torch.zeros((2, self.shape[0], self.shape[0]), dtype=torch.long)
            spa_mask[:, :base, :base + len(app_tags) + 1] = 1
            spa_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
            temp = form_spatial_mask(app_tags, self.spatial_mask[feature['page_id']], self.direction)
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
        return len(self.offsets)


class GraphHtmlConfig(PretrainedConfig):
    def __init__(self,
                 args=None,
                 **kwargs):
        self.hidden_size = kwargs.pop("hidden_size", None)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", None)
        super().__init__(**kwargs)
        if args is not None:
            self.method = args.method
            self.model_type = args.model_type
            self.loss_method = args.loss_method
            self.num_hidden_layers = args.num_node_block
            self.max_depth_embeddings = args.max_depth_embeddings
            self.mask_method = args.mask_method
            self.cnn_feature_dim = args.cnn_feature_dim
            self.cnn_mode = args.cnn_mode


class Link(nn.Module):
    def __init__(self, method, config):
        super().__init__()
        self.method = method
        self.loss_method = config.loss_method
        self.add_position_embeddings = True if config.max_depth_embeddings is not None else False
        self.sequential_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("sequential_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.cnn_mode = config.cnn_mode
        self.cnn_feature_dim = config.cnn_feature_dim
        if self.add_position_embeddings:
            self.depth_embeddings = nn.Embedding(config.max_depth_embeddings, config.hidden_size, padding_idx=0)
        if self.cnn_mode == "once":
            self.scaling = nn.Linear(config.hidden_size + config.cnn_feature_dim, config.hidden_size)

    def forward(self, inputs, tag_to_token, tag_depth, cnn_feature=None):
        assert tag_to_token.dim() == 3
        modified_tag2token = self.deduce_direct_string(tag_to_token)
        outputs = torch.matmul(modified_tag2token, inputs)

        sequential_ids = self.sequential_ids[:, :outputs.size(1)]
        sequential_embeddings = self.sequential_embeddings(sequential_ids)
        outputs = outputs + sequential_embeddings
        if self.add_position_embeddings:
            depth_embeddings = self.depth_embeddings(tag_depth)
            outputs = outputs + depth_embeddings

        if self.cnn_mode == "once":
            outputs = torch.cat([outputs, cnn_feature], dim=2)
            outputs = self.scaling(outputs)

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


class VEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.scaling = nn.Linear(config.hidden_size + config.cnn_feature_dim, config.hidden_size)

    def forward(
        self,
        hidden_states,
        cnn_feature,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = torch.cat([hidden_states, cnn_feature], dim=2)
            hidden_states = self.scaling(hidden_states)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


def convert_mask_to_reality(mask, dtype=torch.float):
    mask = mask.to(dtype=dtype)
    mask = (1.0 - mask) * -10000.0
    return mask


class GraphHtmlBert(BertPreTrainedModel):
    def __init__(self, PTMForQA, config: GraphHtmlConfig, d='b'):
        super(GraphHtmlBert, self).__init__(config)
        self.method = config.method
        self.base_type = config.model_type
        self.loss_method = config.loss_method
        self.mask_method = config.mask_method
        self.cnn_mode = config.cnn_mode
        self.d = d

        self.ptm = getattr(PTMForQA, self.base_type)
        self.link = Link(self.method, config)
        self.num_gat_layers = config.num_hidden_layers
        if self.cnn_mode == 'each':
            self.gat = VEncoder(config)
        else:
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
            visual_feature=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
    ):

        if self.base_type == 'markuplm':
            outputs = self.ptm(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                xpath_tags_seq=xpath_tags_seq,
                xpath_subs_seq=xpath_subs_seq,
            )
            sequence_output = outputs[0]
            outputs = outputs[2:]
        else:
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

        gat_inputs = self.link(sequence_output, tag_to_tok, tag_depth, cnn_feature=visual_feature)
        if self.config.num_attention_heads == 12:
            if self.mask_method == 0:
                if self.d == 'b':
                    spa_mask = spa_mask.repeat(1, 2, 1, 1)
                else:
                    spa_mask = spa_mask.repeat(1, 4, 1, 1)
                if gat_mask.size(1) == 1:
                    gat_mask = gat_mask.repeat(1, 4, 1, 1)
                else:
                    gat_mask = gat_mask.repeat(1, 2, 1, 1)
                gat_mask = torch.cat([gat_mask, spa_mask], dim=1)
            elif self.mask_method == 1:
                gat_mask = spa_mask.repeat(1, 3, 1, 1)
            elif gat_mask.size(1) != 1:
                gat_mask = gat_mask.repeat(1, 6, 1, 1)
        elif self.config.num_attention_heads == 16:
            if self.mask_method == 0:
                spa_mask = spa_mask.repeat(1, 3, 1, 1)
                if gat_mask.size(1) == 1:
                    gat_mask = gat_mask.repeat(1, 4, 1, 1)
                else:
                    gat_mask = gat_mask.repeat(1, 2, 1, 1)
                gat_mask = torch.cat([gat_mask, spa_mask], dim=1)
            elif self.mask_method == 1:
                gat_mask = spa_mask.repeat(1, 4, 1, 1)
            elif gat_mask.size(1) != 1:
                gat_mask = gat_mask.repeat(1, 8, 1, 1)
        else:
            raise NotImplementedError()
        if head_mask is None:
            head_mask = [None] * self.num_gat_layers
        extended_gat_mask = convert_mask_to_reality(gat_mask)
        if self.cnn_mode == 'each':
            gat_outputs = self.gat(gat_inputs, visual_feature, attention_mask=extended_gat_mask, head_mask=head_mask)
        else:
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


class VConfig(PretrainedConfig):
    r"""
    The configuration class to store the configuration of V-PLM

    Arguments:
        method (str): the name of the method in use, choice: ['T-PLM', 'H-PLM', 'V-PLM'].
        model_type (str): the model type of the backbone PLM, currently support BERT and Electra.
        num_node_block (int): the number of the visual information enhanced self-attention block in use.
        cnn_feature_dim (int): the dimension of the provided cnn features.
        kwargs (dict): the other configuration which the configuration of the PLM in use needs.
    """
    def __init__(self,
                 method,
                 model_type1,
                 num_node_block,
                 cnn_feature_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.model_type1 = model_type1
        self.num_node_block = num_node_block
        self.cnn_feature_dim = cnn_feature_dim
        self.cat_hidden_size = self.hidden_size
        if self.method == 'V-PLM':
            self.cat_hidden_size += self.cnn_feature_dim


class VBlock(nn.Module):
    r"""
    the visual information enhanced self-attention block.
    """
    def __init__(self, config):
        super().__init__()
        self.method = config.method
        self.attention = BertAttention(config)
        self.dense = nn.Linear(config.cat_hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            inputs,
            visual_feature,
            attention_mask=None,
            head_mask=None
    ):
        if self.method == 'V-PLM':
            assert visual_feature.dim() == 3
            output = torch.cat([inputs, visual_feature], dim=2)
        else:
            output = inputs
        output = self.dense(output)
        output = self.dropout(output)
        output = self.LayerNorm(output + inputs)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dense.weight.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        output = self.attention(output, attention_mask=extended_attention_mask, head_mask=head_mask)[0]

        return output


class VPLM(BertPreTrainedModel):
    r"""
    the V-PLM model.

    Arguments:
        ptm: the Pretrained Language Model backbone in use, currently support BERT and Electra.
        config (VConfig): the configuration for V-PLM.
    """
    def __init__(self, ptm, config: VConfig):
        super(VPLM, self).__init__(config)
        self.base_type = config.model_type1
        self.ptm = getattr(ptm, self.base_type)
        self.struc = nn.ModuleList([VBlock(config) for _ in range(config.num_node_block)])
        self.qa_outputs = ptm.qa_outputs

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            visual_feature=None
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

        for i, layer in enumerate(self.struc):
            sequence_output = layer(sequence_output, visual_feature, attention_mask=attention_mask, head_mask=head_mask)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
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

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
