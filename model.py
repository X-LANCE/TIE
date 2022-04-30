#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertLayer
from transformers import PretrainedConfig, AutoConfig, AutoModelForQuestionAnswering

from markuplmft import MarkupLMForQuestionAnswering


class TIEEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_gat_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class TIEConfig(PretrainedConfig):
    def __init__(self,
                 args=None,
                 **kwargs):
        self.hidden_size = kwargs.pop("hidden_size", None)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", None)
        super().__init__(**kwargs)
        if args is not None:
            self.ptm_model_type = args.model_type
            self.ptm_name_or_path = args.model_name_or_path
            self.num_gat_layers = args.num_node_block
            self.mask_method = args.mask_method
            self.direction = args.direction
            self.name_or_path = args.model_name_or_path
            self.merge = args.merge is not None


class HTMLBasedPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sequential_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("sequential_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, inputs, tag_to_token):
        assert tag_to_token.dim() == 3
        modified_tag2token = self.deduce_direct_string(tag_to_token)
        outputs = torch.matmul(modified_tag2token, inputs)

        sequential_ids = self.sequential_ids[:, :outputs.size(1)]
        sequential_embeddings = self.sequential_embeddings(sequential_ids)
        outputs = outputs + sequential_embeddings

        return outputs

    @staticmethod
    def deduce_direct_string(tag_to_token):
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
    base_model_prefix = "ptm"

    def __init__(self, config: TIEConfig, init_plm=False):
        super(TIE, self).__init__(config)
        self.base_type = getattr(config, 'ptm_model_type', 'markuplm')
        self.mask_method = config.mask_method
        self.direction = config.direction
        self.merge = config.merge

        if init_plm:
            if self.base_type == 'markuplm':
                self.ptm = MarkupLMForQuestionAnswering.from_pretrained(config.name_or_path, config=config)
            else:
                ptm_config = AutoConfig.from_pretrained(config.ptm_name_or_path)
                self.ptm = AutoModelForQuestionAnswering.from_pretrained(config.ptm_name_or_path, config=ptm_config)
        else:
            if self.base_type == 'markuplm':
                self.ptm = MarkupLMForQuestionAnswering(config)
            else:
                ptm_config = AutoConfig.from_pretrained(config.ptm_name_or_path)
                ptm_config.vocab_size = config.vocab_size
                self.ptm = AutoModelForQuestionAnswering.from_config(ptm_config)
        self.ptm = getattr(self.ptm, self.base_type)
        self.link = HTMLBasedPooling(config)
        self.num_gat_layers = config.num_gat_layers
        self.gat = TIEEncoder(config)
        self.gat_outputs = nn.Linear(config.hidden_size, 1)
        if self.merge:
            self.qa_outputs = self.ptm.qa_outputs
        else:
            self.qa_outputs = None

    def forward(
            self,
            input_ids,
            attention_mask=None,
            dom_mask=None,
            spa_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            answer_tid=None,
            tag_to_tok=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            start_positions=None,
            end_positions=None,
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

        gat_inputs = self.link(sequence_output, tag_to_tok)

        if self.config.num_attention_heads == 12:
            if self.mask_method == 0:
                if self.direction == 'b':
                    spa_mask = spa_mask.repeat(1, 2, 1, 1)
                else:
                    spa_mask = spa_mask.repeat(1, 4, 1, 1)
                dom_mask = dom_mask.repeat(1, 4, 1, 1)
                gat_mask = torch.cat([dom_mask, spa_mask], dim=1)
            elif self.mask_method == 1:
                gat_mask = spa_mask.repeat(1, 3, 1, 1)
            else:
                gat_mask = dom_mask
        elif self.config.num_attention_heads == 16:
            if self.mask_method == 0:
                if self.direction == 'b':
                    spa_mask = spa_mask.repeat(1, 3, 1, 1)
                else:
                    spa_mask = spa_mask.repeat(1, 6, 1, 1)
                dom_mask = dom_mask.repeat(1, 4, 1, 1)
                gat_mask = torch.cat([dom_mask, spa_mask], dim=1)
            elif self.mask_method == 1:
                gat_mask = spa_mask.repeat(1, 4, 1, 1)
            else:
                gat_mask = dom_mask
        else:
            raise NotImplementedError()
        if head_mask is None:
            head_mask = [None] * self.num_gat_layers
        extended_gat_mask = convert_mask_to_reality(gat_mask)
        gat_outputs = self.gat(gat_inputs, attention_mask=extended_gat_mask, head_mask=head_mask)
        final_outputs = gat_outputs[0]
        tag_logits = self.gat_outputs(final_outputs)
        tag_logits = tag_logits.squeeze(-1)

        if self.qa_outputs is not None:
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            outputs = (start_logits, end_logits,) + outputs

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

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

        outputs = (tag_logits,) + outputs

        if answer_tid is not None:
            if len(answer_tid.size()) > 1:
                answer_tid = answer_tid.squeeze(-1)
            ignored_index = tag_logits.size(1)
            answer_tid.clamp_(0, ignored_index)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = loss_fct(tag_logits, answer_tid)
            outputs = (loss,) + outputs

        return outputs
        # (loss), tag_logits/probs, (total_loss), (start_logits), (end_logits), (hidden_states), (attentions)
