#!/usr/bin/python
# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertSelfAttention, BertAttention, \
    BertLayer, BertSelfOutput, BertIntermediate, BertOutput
from transformers import PretrainedConfig, AutoModelForQuestionAnswering

from markuplmft import MarkupLMForQuestionAnswering


class TIESelfAttention(BertSelfAttention):
    def __init__(self, config):
        super(TIESelfAttention, self).__init__(config)
        if self.position_embedding_type == "relation":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(config.max_rel_position_embeddings, self.attention_head_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relation":
            positional_embedding = self.distance_embedding(attention_mask)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TIEAttention(BertAttention):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = TIESelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


class TIELayer(BertLayer):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TIEAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


class TIEEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([TIELayer(config) for _ in range(config.num_gat_layers)])
        self.gradient_checkpointing = False


class TIEConfig(PretrainedConfig):
    def __init__(self,
                 args=None,
                 **kwargs):
        self.hidden_size = kwargs.pop("hidden_size", None)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", None)
        super().__init__(**kwargs)
        if args is not None:
            self.method = args.method
            self.loss_method = args.loss_method
            self.num_gat_layers = args.num_node_block
            self.max_depth_embeddings = args.max_depth_embeddings
            self.mask_method = args.mask_method
            self.cnn_feature_dim = args.cnn_feature_dim
            self.cnn_mode = args.cnn_mode
            self.max_rel_position_embeddings = args.max_rel_position_embeddings
            self.direction = args.directrion


class HTMLBasedPooling(nn.Module):
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

    def forward(self, inputs, tag_to_token, tag_depth, cnn_feature=None, xpath_embedding=None):
        assert tag_to_token.dim() == 3
        modified_tag2token = self.deduce_direct_string(tag_to_token)
        outputs = torch.matmul(modified_tag2token, inputs)

        sequential_ids = self.sequential_ids[:, :outputs.size(1)]
        sequential_embeddings = self.sequential_embeddings(sequential_ids)
        outputs = outputs + sequential_embeddings
        if self.add_position_embeddings:
            depth_embeddings = self.depth_embeddings(tag_depth)
            outputs = outputs + depth_embeddings
        if xpath_embedding is not None:
            outputs = outputs + xpath_embedding

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


def convert_mask_to_reality(mask, dtype=torch.float):
    mask = mask.to(dtype=dtype)
    mask = (1.0 - mask) * -10000.0
    return mask


class TIE(BertPreTrainedModel):
    def __init__(self, config: TIEConfig, init_plm=False):
        super(TIE, self).__init__(config)
        self.method = config.method
        self.base_type = config.model_type
        self.loss_method = config.loss_method
        self.mask_method = config.mask_method
        self.cnn_mode = config.cnn_mode
        self.direction = config.direction

        if init_plm:
            if self.base_type == 'markuplm':
                self.ptm = MarkupLMForQuestionAnswering.from_pretrained(config.name_or_path, config=config)
            else:
                self.ptm = AutoModelForQuestionAnswering.from_pretrained(config.name_or_path, config=config)
        else:
            if self.base_type == 'markuplm':
                self.ptm = MarkupLMForQuestionAnswering(config)
            else:
                self.ptm = AutoModelForQuestionAnswering.from_config(config)
        self.ptm = getattr(self.ptm, self.base_type)
        self.link = HTMLBasedPooling(self.method, config)
        self.num_gat_layers = config.num_gat_layers
        # if self.cnn_mode == 'each':
        #     self.gat = VEncoder(config)
        # else:
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
            xpath_tags_seq_tag=None,
            xpath_subs_seq_tag=None,
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

        if xpath_tags_seq_tag is not None :
            assert xpath_subs_seq_tag is not None
            xpath_embedding = self.ptm.embeddings.xpath_embeddings(xpath_tags_seq_tag, xpath_subs_seq_tag)
        else:
            xpath_embedding = None
        gat_inputs = self.link(sequence_output, tag_to_tok, tag_depth,
                               cnn_feature=visual_feature, xpath_embedding=xpath_embedding)
        if self.config.num_attention_heads == 12:
            if self.mask_method == 0:
                if self.direction == 'b':
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
                if self.direction == 'b':
                    spa_mask = spa_mask.repeat(1, 3, 1, 1)
                else:
                    spa_mask = spa_mask.repeat(1, 6, 1, 1)
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
            elif self.loss_method == 'bce-soft':
                answer_tid.clamp_(0, 1)
                loss_fct = torch.nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError('Loss method {} is not implemented yet'.format(self.loss_method))
            loss = loss_fct(tag_logits, answer_tid)
            outputs = (loss,) + outputs

        return outputs
        # (loss), tag_logits/probs, (hidden_states), (attentions)
