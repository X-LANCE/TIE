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

import json
import logging
import math
import collections
from io import open
from os import path as osp
import random

from tqdm import tqdm
import numpy as np
import bs4
from bs4 import BeautifulSoup as bs
from transformers.models.bert.tokenization_bert import BasicTokenizer, whitespace_tokenize
from lxml import etree
from markuplmft.data.tag_utils import tags_dict

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 doc_tokens,
                 qas_id,
                 html_tree=None,
                 question_text=None,
                 orig_answer_text=None,
                 answer_tid=None,
                 start_position=None,
                 end_position=None,
                 tok_to_orig_index=None,
                 orig_to_tok_index=None,
                 all_doc_tokens=None,
                 tok_to_tags_index=None,
                 tags_to_tok_index=None,
                 orig_tags=None,
                 tag_depth=None,
                 xpath_tag_map=None,
                 xpath_subs_map=None,):
        self.doc_tokens = doc_tokens
        self.qas_id = qas_id
        self.html_tree = html_tree
        self.question_text = question_text
        self.orig_answer_text = orig_answer_text
        self.answer_tid = answer_tid
        self.start_position = start_position
        self.end_position = end_position
        self.tok_to_orig_index = tok_to_orig_index
        self.orig_to_tok_index = orig_to_tok_index
        self.all_doc_tokens = all_doc_tokens
        self.tok_to_tags_index = tok_to_tags_index
        self.tags_to_tok_index = tags_to_tok_index
        self.orig_tags = orig_tags
        self.tag_depth = tag_depth
        self.xpath_tag_map = xpath_tag_map
        self.xpath_subs_map = xpath_subs_map

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.answer_tid:
            s += ", answer_tid: %d" % self.answer_tid
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
                 paragraph_len,
                 answer_tid=None,
                 start_position=None,
                 end_position=None,
                 token_to_tag_index=None,
                 tag_to_token_index=None,
                 app_tags=None,
                 tag_list=None,
                 depth=None,
                 base_index=None,
                 is_impossible=None,
                 xpath_tags_seq=None,
                 xpath_subs_seq=None,):
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
        self.paragraph_len = paragraph_len
        self.answer_tid = answer_tid
        self.start_position = start_position
        self.end_position = end_position
        self.token_to_tag_index = token_to_tag_index
        self.tag_to_token_index = tag_to_token_index
        self.app_tags = app_tags
        self.tag_list = tag_list
        self.depth = depth
        self.base_index = base_index
        self.is_impossible = is_impossible
        self.xpath_tags_seq = xpath_tags_seq
        self.xpath_subs_seq = xpath_subs_seq

    def to_json(self):
        return json.dumps(self.__dict__)


def html_escape(html):
    html = html.replace('&quot;', '"')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&nbsp;', ' ')
    return html


def get_xpath4tokens(html_fn: str, unique_tids: set):
    xpath_map = {}
    tree = etree.parse(html_fn, etree.HTMLParser())
    nodes = tree.xpath('//*')
    for node in nodes:
        tid = node.attrib.get("tid")
        if int(tid) in unique_tids:
            xpath_map[int(tid)] = tree.getpath(node)
    xpath_map[len(nodes)] = "/html"
    xpath_map[len(nodes) + 1] = "/html"
    return xpath_map


def get_xpath_and_treeid4tokens(html_code, unique_tids, max_depth):
    unknown_tag_id = len(tags_dict)
    pad_tag_id = unknown_tag_id + 1
    max_width = 1000
    width_pad_id = 1001

    pad_x_tag_seq = [pad_tag_id] * max_depth
    pad_x_subs_seq = [width_pad_id] * max_depth

    def xpath_soup(element):

        xpath_tags = []
        xpath_subscripts = []
        tree_index = []
        child = element if element.name else element.parent
        for parent in child.parents:  # type: bs4.element.Tag
            siblings = parent.find_all(child.name, recursive=False)
            para_siblings = parent.find_all(True, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child))

            tree_index.append(next(i for i, s in enumerate(para_siblings, 0) if s is child))
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        tree_index.reverse()
        return xpath_tags, xpath_subscripts, tree_index

    xpath_tag_map = {}
    xpath_subs_map = {}

    for tid in unique_tids:
        element = html_code.find(attrs={'tid': tid})
        if element is None:
            xpath_tags = pad_x_tag_seq
            xpath_subscripts = pad_x_subs_seq

            xpath_tag_map[tid] = xpath_tags
            xpath_subs_map[tid] = xpath_subscripts
            continue

        xpath_tags, xpath_subscripts, tree_index = xpath_soup(element)

        assert len(xpath_tags) == len(xpath_subscripts)
        assert len(xpath_tags) == len(tree_index)

        if len(xpath_tags) > max_depth:
            xpath_tags = xpath_tags[-max_depth:]
            xpath_subscripts = xpath_subscripts[-max_depth:]

        xpath_tags = [tags_dict.get(name, unknown_tag_id) for name in xpath_tags]
        xpath_subscripts = [min(i, max_width) for i in xpath_subscripts]

        # we do not append them to max depth here

        xpath_tags += [pad_tag_id] * (max_depth - len(xpath_tags))
        xpath_subscripts += [width_pad_id] * (max_depth - len(xpath_subscripts))

        xpath_tag_map[tid] = xpath_tags
        xpath_subs_map[tid] = xpath_subscripts

    return xpath_tag_map, xpath_subs_map


def read_squad_examples(input_file, is_training, tokenizer, base_mode, max_depth=50, sample_size=None, simplify=False):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def html_to_text_list(h):
        tag_num, text_list = 0, []
        for element in h.descendants:
            if (type(element) == bs4.element.NavigableString) and (element.strip()):
                text_list.append(element.strip())
            if type(element) == bs4.element.Tag:
                tag_num += 1
        return text_list, tag_num + 2  # + 2 because we treat the additional 'yes' and 'no' as two special tags.

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

    def word_tag_offset(html):
        w_t, t_w, tags, tags_tids = [], [], [], []
        for element in html.descendants:
            if type(element) == bs4.element.Tag:
                content = ' '.join(list(element.strings)).split()
                t_w.append({'start': len(w_t), 'len': len(content)})
                tags.append('<' + element.name + '>')
                tags_tids.append(element['tid'])
            elif type(element) == bs4.element.NavigableString and element.strip():
                text = element.split()
                tid = element.parent['tid']
                ind = tags_tids.index(tid)
                for _ in text:
                    w_t.append(ind)
        t_w.append({'start': len(w_t), 'len': 1})
        t_w.append({'start': len(w_t) + 1, 'len': 1})
        w_t.append(len(tags))
        w_t.append(len(tags) + 1)
        tags.append('<no>')
        tags.append('<yes>')
        return w_t, t_w, tags

    def subtoken_tag_offset(html, s_tok, tok_s):
        w_t, t_w, tags = word_tag_offset(html)
        s_t, t_s = [], []
        unique_tids = set()
        for i in range(len(s_tok)):
            s_t.append(w_t[s_tok[i]])
            unique_tids.add(w_t[s_tok[i]])
        for i in t_w:
            try:
                t_s.append({'start': tok_s[i['start']], 'end': tok_s[i['start'] + i['len']] - 1})
            except IndexError:
                assert i == t_w[-1]
                t_s.append({'start': tok_s[i['start']], 'end': len(s_tok) - 1})
        return s_t, t_s, tags, unique_tids

    def calculate_depth(html_code):
        def _calc_depth(tag, depth):
            for t in tag.contents:
                if type(t) != bs4.element.Tag:
                    continue
                tag_depth.append(depth)
                _calc_depth(t, depth + 1)

        tag_depth = []
        _calc_depth(html_code, 1)
        tag_depth += [2, 2]
        return tag_depth

    # def check_for_index(t2w, token, h):
    #     for ind in range(len(t2w) - 2):
    #         e = h.find(tid=ind)
    #         raw = ' '.join([s.strip() for s in e.strings if s.strip()])
    #         content = token[t2w[ind]['start']:t2w[ind]['end'] + 1]
    #         content = ' '.join([s for s in content if s[0] != '<' or s[-1] != '>'])
    #         assert content == raw, 'Not the same in {}\n{}\n{}\n{}'.format(ind, h, content, raw)

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

            raw_text_list, tag_num = html_to_text_list(html_code)
            page_text = ' '.join(raw_text_list)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in page_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            doc_tokens.append('no')
            char_to_word_offset.append(len(doc_tokens) - 1)
            doc_tokens.append('yes')
            char_to_word_offset.append(len(doc_tokens) - 1)

            if base_mode != 'markuplm':
                real_text, tag_list = html_to_text(bs(html_file))
                all_tag_list = all_tag_list | tag_list
                char_to_word_offset = adjust_offset(char_to_word_offset, real_text)
                doc_tokens = real_text.split()
                doc_tokens.append('no')
                doc_tokens.append('yes')
                doc_tokens = [i for i in doc_tokens if i]
                assert len(doc_tokens) == char_to_word_offset[-1] + 1, (len(doc_tokens), char_to_word_offset[-1])
            else:
                tag_list = []

            if simplify:
                for qa in website["qas"]:
                    qas_id = qa["id"]
                    tag_depth = calculate_depth(html_code)
                    example = SquadExample(doc_tokens=doc_tokens, qas_id=qas_id,
                                           html_tree=html_code, tag_depth=tag_depth)
                    examples.append(example)
            else:
                # Tokenize all doc tokens
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    if token in tag_list:
                        sub_tokens = [token]
                    else:
                        sub_tokens = tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)

                # Generate extra information for features
                if base_mode != 'markuplm':
                    tok_to_tags_index, tags_to_tok_index, orig_tags = word_to_tag_from_text(all_doc_tokens,
                                                                                            bs(html_file))
                    xpath_tag_map, xpath_subs_map = None, None
                else:
                    tok_to_tags_index, tags_to_tok_index, orig_tags, unique_tids = subtoken_tag_offset(
                                                                                            html_code,
                                                                                            tok_to_orig_index,
                                                                                            orig_to_tok_index)

                    xpath_tag_map, xpath_subs_map = get_xpath_and_treeid4tokens(html_code,
                                                                                unique_tids,
                                                                                max_depth=max_depth)

                assert tok_to_tags_index[-1] == tag_num - 1, (tok_to_tags_index[-1], tag_num - 1)
                # check_for_index(tags_to_tok_index, all_doc_tokens, bs(html_file))

                # Process each qas, which is mainly calculate the answer position
                for qa in website["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    answer_tid = None
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    tag_depth = calculate_depth(html_code)

                    if is_training:
                        if len(qa["answers"]) != 1:
                            raise ValueError("For training, each question should have exactly 1 answer.")
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        if answer["element_id"] == -1:
                            answer_tid = len(orig_tags) - 2 + answer["answer_start"]
                            num_char = len(char_to_word_offset) - 2
                        else:
                            answer_tid = answer["element_id"]
                            num_char = calc_num_from_raw_text_list(e_id_to_t_id(answer["element_id"], html_code),
                                                                   raw_text_list)
                        answer_offset = num_char + answer["answer_start"]
                        answer_length = len(orig_answer_text) if answer["element_id"] != -1 else 1
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        node_text = doc_tokens[tok_to_orig_index[tags_to_tok_index[answer_tid]['start']]:
                                               tok_to_orig_index[tags_to_tok_index[answer_tid]['end']] + 1]
                        node_text = ' '.join([s for s in node_text if s[0] != '<' or s[-1] != '>'])
                        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                        if node_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer of question %s: '%s' vs. '%s'",
                                           qa['id'], node_text, cleaned_answer_text)
                            continue
                        actual_text = " ".join([w for w in doc_tokens[start_position:(end_position + 1)]
                                                if w[0] != '<' or w[-1] != '>'])
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer of question %s: '%s' vs. '%s'",
                                           qa['id'], actual_text, cleaned_answer_text)
                            continue

                    example = SquadExample(
                        doc_tokens=doc_tokens,
                        qas_id=qas_id,
                        html_tree=html_code,
                        question_text=question_text,
                        orig_answer_text=orig_answer_text,
                        answer_tid=answer_tid,
                        start_position=start_position,
                        end_position=end_position,
                        tok_to_orig_index=tok_to_orig_index,
                        orig_to_tok_index=orig_to_tok_index,
                        all_doc_tokens=all_doc_tokens,
                        tok_to_tags_index=tok_to_tags_index,
                        tags_to_tok_index=tags_to_tok_index,
                        orig_tags=orig_tags,
                        tag_depth=tag_depth,
                        xpath_tag_map=xpath_tag_map,
                        xpath_subs_map=xpath_subs_map,
                    )
                    examples.append(example)

    if sample_size is not None:
        sampled_index = random.sample(range(0, len(examples)), sample_size)
        sampled_index.sort()
        examples = [examples[ind] for ind in sampled_index]

    return examples, all_tag_list


def convert_examples_to_features(examples, tokenizer, loss_method, max_seq_length, max_tag_length,
                                 doc_stride, max_query_length, is_training, soft_remain, soft_decay,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 max_depth=50):
    """Loads a data file into a list of `InputBatch`s."""

    def label_generating(html_tree, origin_answer_tid, app_tags, base):
        if loss_method == 'base' or not is_training:
            return origin_answer_tid
        t, path = 1, [origin_answer_tid]
        marker = np.zeros(max_tag_length, dtype=np.float)
        marker[origin_answer_tid] = t
        if origin_answer_tid != 0:
            tag = html_tree.find(tid=app_tags[origin_answer_tid - base])
            if tag is None:
                marker[origin_answer_tid] = soft_remain
                marker[base] = 1 - soft_remain
                path.append(base)
            else:
                for p in tag.parents:
                    if type(p) == bs4.BeautifulSoup:
                        break
                    if int(p['tid']) in app_tags:
                        path.append(app_tags.index(int(p['tid'])) + base)
                        marker[app_tags.index(int(p['tid'])) + base] = t
                        t *= soft_decay
                temp = marker.sum() - 1
                if temp != 0:
                    marker[origin_answer_tid] = temp * soft_remain / (1 - soft_remain)
                    marker /= marker.sum()
        answer_tid = marker.tolist()
        return answer_tid

    unique_id = 1000000000
    features = []

    for (example_index, example) in enumerate(tqdm(examples)):

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        xpath_tag_map = example.xpath_tag_map
        xpath_subs_map = example.xpath_subs_map

        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_answer_tid = None
        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_answer_tid = example.answer_tid
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
            tag_to_token_index = []
            token_to_tag_index = []
            depth = []
            tag_list = []

            # CLS
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            tag_to_token_index.append([0, 0])
            token_to_tag_index.append(-1)
            depth.append(0)
            tag_list.append(-1)

            # Query
            for i in range(len(query_tokens)):
                tag_to_token_index.append([len(tokens) + i, len(tokens) + i])
                token_to_tag_index.append(-1)
                tag_list.append(-1)
            tokens += query_tokens
            segment_ids += [sequence_a_segment_id] * len(query_tokens)
            depth += [0] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])
            token_to_tag_index.append(-1)
            depth.append(0)
            tag_list.append(-1)

            # Paragraph
            app_tags = []
            for i in range(len(example.tags_to_tok_index)):
                start = example.tags_to_tok_index[i]['start']
                end = example.tags_to_tok_index[i]['end']
                if end < doc_span.start:
                    continue
                    # tag_to_token_index.append(None)
                elif start >= doc_span.start + doc_span.length:
                    continue
                    # tag_to_token_index.append(None)
                elif start > end:
                    continue
                    # tag_to_token_index.append(None)
                else:
                    start = max(start, doc_span.start) - doc_span.start + len(tokens)
                    end = min(end, doc_span.start + doc_span.length - 1) - doc_span.start + len(tokens)
                    tag_to_token_index.append([start, end])
                    app_tags.append(i)
                    tag_list.append(i)
            for ind in app_tags:
                depth.append(example.tag_depth[ind])
            # if len(app_tags) > max_tag_length - 4:
            if len(app_tags) > max_tag_length - len(query_tokens) - 3:
                raise ValueError('Max tag length is not big enough to contain {}'.format(len(app_tags)))
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = example.tok_to_orig_index[split_token_index]
                token_to_tag_index.append(example.tok_to_tags_index[split_token_index])
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(example.all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])
            token_to_tag_index.append(-1)
            depth.append(0)
            tag_list.append(-1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # mask generating
            base = len(query_tokens) + 2
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0)
                segment_ids.append(pad_token_segment_id)
                token_to_tag_index.append(-1)
            while len(depth) < max_tag_length:
                depth.append(0)
                tag_list.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(token_to_tag_index) == max_seq_length
            assert len(depth) == max_tag_length
            assert len(tag_list) == max_tag_length

            span_is_impossible = False
            answer_tid = None
            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    answer_tid = 0
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    assert tok_answer_tid in app_tags
                    offset = len(query_tokens) + 2
                    answer_tid = app_tags.index(tok_answer_tid) + offset
                    start_position = tok_start_position - doc_start + offset
                    end_position = tok_end_position - doc_start + offset

            pad_x_tag_seq = [216] * max_depth
            pad_x_subs_seq = [1001] * max_depth
            xpath_tags_seq = [xpath_tag_map.get(tid, pad_x_tag_seq) for tid in token_to_tag_index]  # ok
            xpath_subs_seq = [xpath_subs_map.get(tid, pad_x_subs_seq) for tid in token_to_tag_index]  # ok
            answer_tid = label_generating(example.html_tree, answer_tid, app_tags, base)
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
                    paragraph_len=paragraph_len,
                    answer_tid=answer_tid,
                    start_position=start_position,
                    end_position=end_position,
                    token_to_tag_index=token_to_tag_index,
                    tag_to_token_index=tag_to_token_index,
                    app_tags=app_tags,
                    tag_list=tag_list,
                    depth=depth,
                    base_index=base,
                    is_impossible=span_is_impossible,
                    xpath_tags_seq=xpath_tags_seq,
                    xpath_subs_seq=xpath_subs_seq,
                ))
            unique_id += 1

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


RawTagResult = collections.namedtuple("RawResult", ["unique_id", "tag_logits"])


def write_tag_predictions(all_examples, all_features, all_results, n_best_tag_size,
                          output_tag_prediction_file, output_nbest_file, write_pred):
    logger.info("Writing tag predictions to: %s" % output_tag_prediction_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "tag_index", "tag_logit", "tag_id"])

    all_tag_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            possible_values = [ind for ind in range(feature.base_index,
                                                    feature.base_index + len(feature.app_tags))]
            tag_indexes = _get_best_tags(result.tag_logits, n_best_tag_size, possible_values)
            for ind in range(len(tag_indexes)):
                tag_index = tag_indexes[ind]
                if tag_index == 0:
                    continue
                tag_logit = result.tag_logits[tag_index]
                tag_id = feature.app_tags[tag_index - feature.base_index]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        tag_index=tag_index,
                        tag_logit=tag_logit,
                        tag_id=tag_id))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: x.tag_logit,
            reverse=True)

        _NBestPrediction = collections.namedtuple(
            "NBestPrediction", ["tag_logit", "tag_id"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_tag_size:
                break
            if pred.tag_index > 0:  # this is a non-null prediction
                if pred.tag_id in seen_predictions:
                    continue
                seen_predictions[pred.tag_id] = True
            else:
                seen_predictions[-1] = True

            nbest.append(_NBestPrediction(tag_logit=pred.tag_logit, tag_id=pred.tag_id))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NBestPrediction(tag_logit=0.0, tag_id=-1))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.tag_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["probability"] = probs[i]
            output["tag_logit"] = entry.tag_logit
            output["tag_id"] = entry.tag_id
            nbest_json.append(output)
        assert len(nbest_json) >= 1

        best_tag = nbest_json[0]["tag_id"]
        all_tag_predictions[example.qas_id] = best_tag
        all_nbest_json[example.qas_id] = nbest_json

    if write_pred:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        with open(output_tag_prediction_file, 'w') as writer:
            writer.write(json.dumps(all_tag_predictions, indent=4) + '\n')

    return None, all_tag_predictions


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def write_predictions_provided_tag(all_examples, all_features, all_results, n_best_size, max_answer_length,
                                   do_lower_case, output_prediction_file, input_tag_prediction_file,
                                   output_refined_tag_prediction_file, output_nbest_file, verbose_logging, write_pred):
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    def _get_tag_id(ind2tok, start_ind, end_ind, base, ind2tag):
        tag_ind = -1
        for ind in range(base, len(ind2tok)):
            if (start_ind >= ind2tok[ind][0]) and (end_ind <= ind2tok[ind][1]):
                tag_ind = ind
        tag_ind -= base
        assert tag_ind >= 0
        return ind2tag[tag_ind]

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index",
         "tag_index", "start_index", "end_index",
         "tag_logit", "start_logit", "end_logit",
         "tag_id"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_refined_tag_predictions = collections.OrderedDict()
    all_tag_predictions = json.load(open(input_tag_prediction_file))

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        nb_tag_pred = all_tag_predictions[example.qas_id]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            for item in nb_tag_pred:
                tag_pred = item['tag_id']
                if tag_pred not in feature.app_tags:
                    continue
                tag_index = feature.app_tags.index(tag_pred) + feature.base_index
                left_bound, right_bound = feature.tag_to_token_index[tag_index]
                start_indexes = _get_best_indexes(result.start_logits[left_bound:right_bound + 1], n_best_size)
                end_indexes = _get_best_indexes(result.end_logits[left_bound:right_bound + 1], n_best_size)
                start_indexes = [ind + left_bound for ind in start_indexes]
                end_indexes = [ind + left_bound for ind in end_indexes]
                tag_logit = item['tag_logit']
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        tag_ids = _get_tag_id(feature.tag_to_token_index,
                                              start_index, end_index,
                                              feature.base_index, feature.app_tags)
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                tag_index=tag_index,
                                start_index=start_index,
                                end_index=end_index,
                                tag_logit=tag_logit,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                                tag_id=tag_ids))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NBestPrediction = collections.namedtuple(
            "NBestPrediction", ["text", "tag_logit", "start_logit", "end_logit", "tag_id"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
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
            if '{} with the tag_id of {}'.format(final_text, str(pred.tag_index)) in seen_predictions:
                continue
            seen_predictions['{} with the tag_id of {}'.format(final_text, str(pred.tag_index))] = True

            nbest.append(
                _NBestPrediction(
                    text=final_text,
                    tag_logit=pred.tag_logit,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    tag_id=pred.tag_id))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NBestPrediction(
                    text='empty',
                    start_logit=0.0,
                    end_logit=0.0,
                    tag_logit=0.0,
                    tag_id=-1))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["tag_logit"] = entry.tag_logit
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["tag_id"] = entry.tag_id
            nbest_json.append(output)
        assert len(nbest_json) >= 1

        best_text = nbest_json[0]["text"].split()
        best_text = ' '.join([w for w in best_text if w[0] != '<' or w[-1] != '>'])
        all_predictions[example.qas_id] = best_text
        best_tag = nbest_json[0]["tag_id"]
        all_refined_tag_predictions[example.qas_id] = best_tag
        all_nbest_json[example.qas_id] = nbest_json

    if write_pred:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        with open(output_refined_tag_prediction_file, 'w') as writer:
            writer.write(json.dumps(all_refined_tag_predictions, indent=4) + '\n')

    return all_predictions, all_refined_tag_predictions


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


def _get_best_tags(logits, n_best_size, possible_values):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if index_and_score[i][0] not in possible_values:
            continue
        if len(best_indexes) >= n_best_size:
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


def form_tree_mask(app, tree, separate=False, accelerate=True):
    def _unit(node):
        for ch in node.contents:
            if type(ch) != bs4.element.Tag:
                continue
            curr = int(ch['tid'])
            if curr not in app:
                continue
            curr = app.index(curr)
            if len(path) > 0:
                children[path[-1], curr] = 1
            if accelerate:
                for par in path:
                    if separate:
                        adj_up[curr, par] = 1
                        adj_down[par, curr] = 1
                    else:
                        adj[par, curr] = 1
                        adj[curr, par] = 1
            else:
                if len(path) > 0:
                    if separate:
                        adj_up[curr, path[-1]] = 1
                        adj_down[path[-1], curr] = 1
                    else:
                        adj[path[-1], curr] = 1
                        adj[curr, path[-1]] = 1
            path.append(curr)
            _unit(ch)
            path.pop()

    path = []
    children = np.zeros((len(app), len(app)), dtype=np.int)
    if separate:
        adj_up = np.zeros((len(app), len(app)), dtype=np.int)
        adj_down = np.zeros((len(app), len(app)), dtype=np.int)
        ind = np.diag_indices_from(adj_up)
        adj_up[ind] = 1
        adj_down[ind] = 1
        adj_up[:, 0] = 1
        adj_down[0] = 1
        _unit(tree)
        return adj_up, adj_down, children
    else:
        adj = np.zeros((len(app), len(app)), dtype=np.int)
        ind = np.diag_indices_from(adj)
        adj[0] = 1
        adj[:, 0] = 1
        adj[ind] = 1
        _unit(tree)
        return adj, children


def form_spatial_mask(app, rel, direction='b'):
    def _form_direction_mask(rel, d):
        mask = np.zeros((len(app), len(app)), dtype=np.int)
        reverse_mask = np.zeros((len(app), len(app)), dtype=np.int)
        ind = np.diag_indices_from(mask)
        mask[ind] = 1
        for k, v in rel[d].items():
            try:
                curr = app.index(int(k))
            except ValueError:
                continue
            for t in v:
                try:
                    ter = app.index(int(t))
                except ValueError:
                    continue
                mask[curr, ter] = 1
                reverse_mask[ter, curr] = 1
        return mask, reverse_mask
    o = []
    if direction in ['b', 'h']:
        r, rr = _form_direction_mask(rel, 'rw')
        l, rl = _form_direction_mask(rel, 'lw')
        l[rr == 1] = 1
        r[rl == 1] = 1
        o += [r, l]
    if direction in ['b', 'v']:
        u, ru = _form_direction_mask(rel, 'uw')
        d, rd = _form_direction_mask(rel, 'dw')
        u[rd == 1] = 1
        d[ru == 1] = 1
        o += [u, d]
    return np.stack(o)
