# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function

# import re
import json
import logging
import math
import collections
from io import open
from tqdm import tqdm
from os import path as osp
import numpy as np
import random

import bs4
from bs4 import BeautifulSoup as bs

from copy import deepcopy

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
from utils_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 doc_tokens,
                 qas_id,
                 question_text=None,
                 html_code=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 tok_to_orig_index=None,
                 orig_to_tok_index=None,
                 all_doc_tokens=None,
                 tok_to_tags_index=None,
                 tags_to_tok_index=None,
                 orig_tags=None,
                 orig_tag_position=None,
                 tok_to_parent_index=None,
                 parent_name=None,
                 parent_depth=None):
        self.doc_tokens = doc_tokens
        self.qas_id = qas_id
        self.question_text = question_text
        self.html_code = html_code
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.tok_to_orig_index = tok_to_orig_index
        self.orig_to_tok_index = orig_to_tok_index
        self.all_doc_tokens = all_doc_tokens
        self.tok_to_tags_index = tok_to_tags_index
        self.tags_to_tok_index = tags_to_tok_index
        self.orig_tags = orig_tags
        self.orig_tag_position = orig_tag_position
        self.tok_to_parent_index = tok_to_parent_index
        self.parent_name = parent_name
        self.parent_depth = parent_depth

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.end_position:
            s += ", end_position: %d" % self.end_position
        if self.is_impossible:
            s += ", is_impossible: %r" % self.is_impossible
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 page_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 tag_ids=None,
                 tags=None,
                 common_parent_ids=None,
                 common_parent_depth=None,
                 token_to_tag_index=None,
                 tag_to_token_index=None,
                 is_impossible=None,
                 tag_position=None):
        self.unique_id = unique_id
        self.page_id = page_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.tag_ids = tag_ids
        self.tags = tags
        self.common_parent_ids = common_parent_ids
        self.common_parent_depth = common_parent_depth
        self.token_to_tag_index = token_to_tag_index
        self.tag_to_token_index = tag_to_token_index
        self.is_impossible = is_impossible
        self.tag_position = tag_position


def html_escape(html):
    html = html.replace('&quot;', '"')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&nbsp;', ' ')
    return html


def form_tree():
    ...


def read_squad_examples(input_file, is_training, version_2_with_negative, tokenizer,
                        method=1, visual_down_sample=None, simplify=False):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    if method == 6:
        method = 4
    if method in [3, 4, 5]:
        tokenizer, _ = tokenizer

    def delete_codes(h):
        changes = True
        while changes:
            changes = False
            for node in h.html.descendants:
                if type(node) == bs4.element.NavigableString:
                    continue
                cnt, ns = 0, False
                for c in node.contents:
                    if type(c) != bs4.element.NavigableString:
                        cnt += 1
                        tag_children = c
                    elif c.strip():
                        ns = True
                if ns:
                    continue
                if cnt == 1:
                    node.replace_with(tag_children)
                    changes = True
                if cnt == 0:
                    node.decompose()
                    changes = True
        return h

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def html_to_text(h):
        tag_list = set()
        for element in h.descendants:
            if type(element) == bs4.element.Tag:
                element.attrs = {}
                temp = str(element).split()
                tag_list.add(temp[0])
                tag_list.add(temp[-1])
        return html_escape(str(h)), tag_list

    def adjust_offset(offset, text):
        text_list = text.split()
        cnt, adjustment = 0, []
        for t in text_list:
            if not t:
                continue
            if t[0] == '<' and t[-1] == '>':
                adjustment.append(offset.index(cnt))
            else:
                cnt += 1
        add = 0
        adjustment.append(len(offset))
        for i in range(len(offset)):
            while i >= adjustment[add]:
                add += 1
            offset[i] += add
        return offset

    def e_id_to_t_id(e_id, html):
        t_id = 0
        for element in html.descendants:
            if type(element) == bs4.element.NavigableString and element.strip():
                t_id += 1
            if type(element) == bs4.element.Tag:
                if int(element.attrs['tid']) == e_id:
                    break
        return t_id

    def calc_num_from_raw_text_list(t_id, l):
        n_char = 0
        for i in range(t_id):
            n_char += len(l[i]) + 1
        return n_char

    def word_to_tag_from_text(tokens, h):
        cnt, path = -1, []
        w2t, t2w, tags = [], [], []
        for ind in range(len(tokens) - 2):
            t = tokens[ind]
            if len(t) < 2:
                w2t.append(path[-1])
                continue
            if t[0] == '<' and t[-2] == '/':
                cnt += 1
                w2t.append(cnt)
                tags.append(t)
                t2w.append({'start': ind, 'end': ind})
                continue
            if t[0] == '<' and t[1] != '/':
                cnt += 1
                path.append(cnt)
                tags.append(t)
                t2w.append({'start': ind})
            w2t.append(path[-1])
            if t[0] == '<' and t[1] == '/':
                num = path.pop()
                t2w[num]['end'] = ind
        w2t.append(cnt + 1)
        w2t.append(cnt + 2)
        tags.append('<no>')
        tags.append('<yes>')
        t2w.append({'start': len(tokens) - 2, 'end': len(tokens) - 2})
        t2w.append({'start': len(tokens) - 1, 'end': len(tokens) - 1})
        assert len(w2t) == len(tokens)
        assert len(tags) == len(t2w), (len(tags), len(t2w))
        assert len(path) == 0, h
        return w2t, t2w, tags

    def load_tag_position(raw, html):
        left = None
        tag_position = [[0, 0, visual_down_sample, visual_down_sample], [0, 0, visual_down_sample, visual_down_sample]]
        i = html.descendants

        def next_tag():
            try:
                temp = next(i)
                while type(temp) != bs4.element.Tag:
                    temp = next(i)
                return temp
            except StopIteration:
                return None

        _ = next(i)
        _ = next(i)
        _ = next(i)
        e = next_tag()
        for k, v in raw.items():
            p = v['rect']
            if left is None:
                left, up, width, height = p['x'], p['y'], p['width'], p['height']
            while int(e['tid']) < int(k):
                tag_position.append([-1, -1, -1, -1])
                e = next_tag()
                if e is None:
                    break
            if int(e['tid']) != int(k):
                continue
            l = (p['x'] - left) / width * visual_down_sample
            u = (p['y'] - up) / height * visual_down_sample
            r = l + p['width'] / width * visual_down_sample
            d = u + p['height'] / height * visual_down_sample
            l = min(max(0, l), visual_down_sample)
            u = min(max(0, u), visual_down_sample)
            r = min(max(0, r), visual_down_sample)
            d = min(max(0, d), visual_down_sample)
            tag_position.append([l, u, r, d])
            e = next_tag()
            if e is None:
                break
        while e is not None:
            tag_position.append([-1, -1, -1, -1])
            e = next_tag()
        return tag_position

    def calc_common_parent(s_t_offset, h):
        curr = s_t_offset[0]
        curr_node = h.find(tid=curr)
        curr_parents = [curr_node] + list(curr_node.parents)
        parent_name, parent_depth, index, cnt = [], [], [0], 0
        for ind in s_t_offset[1:-2]:
            if curr == ind:
                index.append(cnt)
                continue
            cnt += 1
            temp_node = h.find(tid=ind)
            temp_parents = [temp_node] + list(temp_node.parents)
            for depth in range(-1, - min(len(curr_parents), len(temp_parents)) - 1, -1):
                if curr_parents[depth] != temp_parents[depth]:
                    depth += 1
                    break
            name = '<' + curr_parents[depth].name + '>'
            parent_name.append(name)
            parent_depth.append(- depth - 1)
            index.append(cnt)
            curr, curr_node, curr_parents = ind, temp_node, temp_parents
        index.append(cnt + 1)
        index.append(cnt + 2)
        parent_name.append('<html>')
        parent_name.append('<html>')
        parent_name.append('<html>')
        parent_depth.append(1)
        parent_depth.append(1)
        parent_depth.append(1)
        return index, parent_name, parent_depth

    examples = []
    all_tag_list = set()
    for entry in input_data:
        domain = entry["domain"]
        for website in entry["websites"]:

            # Generate Doc Tokens
            page_id = website["page_id"]
            curr_dir = osp.join(osp.dirname(input_file), domain, page_id[0:2], 'processed_data')
            html_file = open(osp.join(curr_dir, page_id + '.html')).read()
            html_code = bs(html_file)
            raw_text_list = [s.strip() for s in html_code.strings if s.strip()]
            page_text = ' '.join(raw_text_list)
            length = -1
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in page_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        length += 1
                    prev_is_whitespace = False
                char_to_word_offset.append(length)
            char_to_word_offset.append(length + 1)
            char_to_word_offset.append(length + 2)

            real_text, tag_list = html_to_text(bs(html_file))
            all_tag_list = all_tag_list | tag_list
            char_to_word_offset = adjust_offset(char_to_word_offset, real_text)
            doc_tokens = real_text.split()
            doc_tokens.append('no')
            doc_tokens.append('yes')
            doc_tokens = [i for i in doc_tokens if i]
            assert len(doc_tokens) == char_to_word_offset[-1] + 1, (len(doc_tokens), char_to_word_offset[-1])

            if simplify:
                for qa in website["qas"]:
                    qas_id = qa["id"]
                    example = SquadExample(doc_tokens=doc_tokens, qas_id=qas_id)
                    examples.append(example)
            else:
                # Tokenize all doc tokens
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    sub_tokens = tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)

                # Generate extra information for features
                tok_to_tags_index, tags_to_tok_index, orig_tags = word_to_tag_from_text(all_doc_tokens, html_code)
                # if method == 4:
                #     raw_position = json.load(open(osp.join(curr_dir, page_id + '.json')))
                #     orig_tag_position = load_tag_position(raw_position, html_code)
                #     orig_tag_position.append([-1, -1, -1, -1])
                #     orig_tag_position.append([-1, -1, -1, -1])
                #     assert len(orig_tag_position) == len(orig_tags)
                # else:
                #     orig_tag_position = None
                # if method == 5:
                #     simple_code = delete_codes(bs(html_file))
                #     tok_to_parent_index, parent_name, parent_depth = calc_common_parent(tok_to_tags_index,
                #                                                                         simple_code)
                # else:
                #     tok_to_parent_index, parent_name, parent_depth = None, None, None

                # Process each qas, which is mainly calculate the answer position
                for qa in website["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False

                    if is_training:
                        if version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            if answer["element_id"] == -1:
                                num_char = len(char_to_word_offset) - 2
                            else:
                                num_char = calc_num_from_raw_text_list(e_id_to_t_id(answer["element_id"], html_code),
                                                                       raw_text_list)
                            answer_offset = num_char + answer["answer_start"]
                            answer_length = len(orig_answer_text) if answer["element_id"] != -1 else 1
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join([w for w in doc_tokens[start_position:(end_position + 1)]
                                                    if w[0] != '<' or w[-1] != '>'])
                            cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer of question %s: '%s' vs. '%s'",
                                               qa['id'], actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        doc_tokens=doc_tokens,
                        qas_id=qas_id,
                        question_text=question_text,
                        html_code=html_code,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                        tok_to_orig_index=tok_to_orig_index,
                        orig_to_tok_index=orig_to_tok_index,
                        all_doc_tokens=all_doc_tokens,
                        tok_to_tags_index=tok_to_tags_index,
                        tags_to_tok_index=tags_to_tok_index,
                        orig_tags=orig_tags,
                        # orig_tag_position=orig_tag_position,
                        # tok_to_parent_index=tok_to_parent_index,
                        # parent_name=parent_name,
                        # parent_depth=parent_depth
                    )
                    examples.append(example)
    return examples, all_tag_list


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 max_tag_length=600, pad_tag_label=0,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False,
                                 method=1, sample_size=None, visual_down_sample=None):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    # if method == 6:
    #     method = 4

    features = []

    if sample_size is not None:
        sampled_index = random.sample(range(0, len(examples)), sample_size)
    else:
        sampled_index = None

    ##############
    mt = 0

    for (example_index, example) in enumerate(tqdm(examples)):

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        if sample_size is not None:
            if example_index not in sampled_index:
                continue

        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = example.orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = example.orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(example.all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                example.all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(example.all_doc_tokens):
            length = len(example.all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(example.all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            token_to_tag_index = []

            tag_to_token_index, tags = [], deepcopy(example.orig_tags)
            q, c, s = None, None, None
            # if method == 4:
            #     tag_position = deepcopy(example.orig_tag_position)
            # else:
            #     tag_position = None
            # if method == 5:
            #     common_parent_name, common_parent_depth = [], []
            #     special = '[UNK]'
            # else:
            #     common_parent_name, common_parent_depth, common_parent_ids, special = None, None, None, None

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0
                token_to_tag_index.append(len(tags))
                tags.append('[CLS]')
                c = [0, 0]
                # if method == 4:
                #     tag_position.append([-1, -1, -1, -1])
                # if method == 5:
                #     common_parent_name.append(special)
                #     common_parent_depth.append(0)

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)
                token_to_tag_index += [len(tags)] * len(query_tokens)
                tags.append('<question>')
                q = [len(tokens) - len(query_tokens), len(tokens) - 1]
                # if method == 4:
                #     tag_position.append([-1, -1, -1, -1])
                # if method == 5:
                #     common_parent_name += [special] * len(query_tokens)
                #     common_parent_depth += [0] * len(query_tokens)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
                token_to_tag_index.append(len(tags))
                tags.append('[SEP]')
                s = [len(tokens) - 1, len(tokens) - 1]
                # if method == 4:
                #     tag_position.append([-1, -1, -1, -1])
                # if method == 5:
                #     common_parent_name.append(special)
                #     common_parent_depth.append(0)

            # Paragraph
            for i in range(len(example.tags_to_tok_index)):
                start = example.tags_to_tok_index[i]['start']
                end = example.tags_to_tok_index[i]['end']
                if end < doc_span.start:
                    tag_to_token_index.append(None)
                elif start >= doc_span.start + doc_span.length:
                    tag_to_token_index.append(None)
                elif start > end:
                    tag_to_token_index.append(None)
                else:
                    start = max(start, doc_span.start) - doc_span.start + len(tokens)
                    end = min(end, doc_span.start + doc_span.length - 1) - doc_span.start + len(tokens)
                    tag_to_token_index.append([start, end])
            if len(tag_to_token_index) > max_tag_length - 4:
                raise ValueError('Max tag length is not big enough')
            for i in [c, q, s]:
                if i is not None:
                    tag_to_token_index.append(i)
            assert len(tags) == len(tag_to_token_index)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = example.tok_to_orig_index[split_token_index]
                token_to_tag_index.append(example.tok_to_tags_index[split_token_index])
                # if method == 5:
                #     common_parent_name.append(example.parent_name[example.tok_to_parent_index[split_token_index]])
                #     common_parent_depth.append(example.parent_depth[example.tok_to_parent_index[split_token_index]])

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(example.all_doc_tokens[split_token_index])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
                token_to_tag_index.append(len(tags))
                tags.append('[SEP]')
                tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])
                # if method == 4:
                #     tag_position.append([-1, -1, -1, -1])
                # if method == 5:
                #     common_parent_name.append(special)
                #     common_parent_depth.append(0)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)
                token_to_tag_index += [len(tags)] * len(query_tokens)
                tags.append('<question>')
                tag_to_token_index.append([len(tokens) - len(query_tokens), len(tokens) - 1])
                # if method == 4:
                #     tag_position.append([-1, -1, -1, -1])
                # if method == 5:
                #     common_parent_name += [special] * len(query_tokens)
                #     common_parent_depth += [0] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)
            token_to_tag_index.append(len(tags))
            tags.append('[SEP]')
            tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])
            # if method == 4:
            #     tag_position.append([-1, -1, -1, -1])
            # if method == 5:
            #     common_parent_name.append(special)
            #     common_parent_depth.append(0)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token
                token_to_tag_index.append(len(tags))
                tags.append('[CLS]')
                tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])
                # if method == 4:
                #     tag_position.append([-1, -1, -1, -1])
                # if method == 5:
                #     common_parent_name.append(special)
                #     common_parent_depth.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # tags = html_tokenizer.tokenize(' '.join(tags))
            # tag_ids = html_tokenizer.convert_tokens_to_ids(tags)
            assert len(tags) == len(tag_to_token_index)
            # if method == 4:
            #     assert len(tag_position) == len(tags)
            # while len(tag_position) < max_tag_length:
            #     tag_ids.append(pad_tag_label)
            #     if method == 4:
            #         tag_position.append([-1, -1, -1, -1])

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)
                token_to_tag_index.append(max_tag_length - 1)
                # if method == 5:
                #     common_parent_name.append('[UNK]')
                #     common_parent_depth.append(0)

            # if method == 4:
            #     tag_position = np.array(tag_position)
            #     tag_position = np.floor(tag_position)
            #     tag_position[tag_position == visual_down_sample] = visual_down_sample - 1
            #     tag_position[tag_position == -1] = visual_down_sample
            # if method == 5:
            #     common_parent_name = html_tokenizer.tokenize(' '.join(common_parent_name))
            #     common_parent_ids = html_tokenizer.convert_tokens_to_ids(common_parent_name)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(token_to_tag_index) == max_seq_length
            # if method == 4:
            #     assert (tag_position >= 0).all()
            #     assert (tag_position <= visual_down_sample).all()
            # if method == 5:
            #     assert len(common_parent_ids) == max_seq_length
            #     assert len(common_parent_depth) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    if sequence_a_is_doc:
                        doc_offset = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            ###############
            cnt = 0
            for t in tag_to_token_index:
                if t is not None:
                    cnt += 1
            if cnt > mt:
                mt = cnt

            # if example_index < 20:
            #     logger.info("*** Example ***")
            #     logger.info("unique_id: %s" % (unique_id))
            #     logger.info("example_index: %s" % (example_index))
            #     logger.info("doc_span_index: %s" % (doc_span_index))
            #     logger.info("tokens: %s" % " ".join(tokens))
            #     logger.info("token_to_orig_map: %s" % " ".join([
            #         "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            #     logger.info("token_is_max_context: %s" % " ".join([
            #         "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
            #     ]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info(
            #         "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     if is_training and span_is_impossible:
            #         logger.info("impossible example")
            #     if is_training and not span_is_impossible:
            #         answer_text = " ".join(tokens[start_position:(end_position + 1)])
            #         logger.info("start_position: %d" % (start_position))
            #         logger.info("end_position: %d" % (end_position))
            #         logger.info(
            #             "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    page_id=example.qas_id[:-5],
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    tag_to_token_index=tag_to_token_index,
                    token_to_tag_index=token_to_tag_index,
                    # tag_ids=tag_ids,
                    tags=tags,
                    # common_parent_ids=common_parent_ids,
                    # common_parent_depth=common_parent_depth,
                    is_impossible=span_is_impossible,
                    # tag_position=tag_position
                ))
            unique_id += 1

    print(mt)
    raise SystemError('Mission Complete')
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file, output_tag_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold, write_pred):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "tag_ids"])

    all_predictions = collections.OrderedDict()
    all_tag_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    tag_ids = set(feature.token_to_tag_index[start_index: end_index + 1])
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            tag_ids=list(tag_ids)))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    tag_ids=[-1]))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "tag_ids"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    tag_ids=pred.tag_ids))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        tag_ids=[-1]))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, tag_ids=[-1]))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, tag_ids=[-1]))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["tag_ids"] = entry.tag_ids
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            best = nbest_json[0]["text"].split()
            best = ' '.join([w for w in best if w[0] != '<' or w[-1] != '>'])
            all_predictions[example.qas_id] = best
            all_tag_predictions[example.qas_id] = nbest_json[0]["tag_ids"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
                all_tag_predictions[example.qas_id] = []
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
                all_tag_predictions[example.qas_id] = best_non_null_entry.tag_ids
        all_nbest_json[example.qas_id] = nbest_json

    if write_pred:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        with open(output_tag_prediction_file, 'w') as writer:
            writer.write(json.dumps(all_tag_predictions, indent=4) + '\n')

        if version_2_with_negative:
            with open(output_null_log_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    else:
        returns = [all_predictions, all_tag_predictions]
        if version_2_with_negative:
            returns += [scores_diff_json]
        else:
            returns += [None]
        return returns

    return all_predictions


# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
                                           ["unique_id", "start_top_log_probs", "start_top_index",
                                            "end_top_log_probs", "end_top_index", "cls_logits"])


def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                               max_answer_length, output_prediction_file,
                               output_nbest_file,
                               output_null_log_odds_file, orig_data_file,
                               start_n_top, end_n_top, version_2_with_negative,
                               tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
         "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_tag_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            # 
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case,
                                        verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                                 end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    with open(orig_data_file, "r", encoding='utf-8') as reader:
        orig_data = json.load(reader)["data"]

    qid_to_has_ans = make_qid_to_has_ans(orig_data)
    # has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    # no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw, coin_raw = get_raw_scores(orig_data, all_predictions, all_tag_predictions)
    out_eval = {}

    find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json, qid_to_has_ans)

    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
