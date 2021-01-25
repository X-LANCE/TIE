""" Official evaluation script for SQuAD version 2.0.
    Modified by XLNet authors to update `find_best_threshold` scripts for SQuAD V2.0

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import collections
import json
from copy import deepcopy

import numpy as np
import os
import re
import string
import sys
from bs4 import BeautifulSoup


class EVAL_OPTS:
    def __init__(self, data_file, pred_file, tag_pred_file, result_file='',
                 out_file="",  out_image_dir=None, verbose=False):
        self.data_file = data_file
        self.pred_file = pred_file
        self.tag_pred_file = tag_pred_file
        self.result_file = result_file
        self.out_file = out_file
        self.out_image_dir = out_image_dir
        self.verbose = verbose


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
    parser.add_argument('tag_pred_file', metavar='tag_pred.json', help='Model predictions.')
    parser.add_argument('--result-file', '-r', metavar='qas_eval.json')
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--na-prob-file', '-n', metavar='na_prob.json',
                        help='Model estimates of probability of no answer.')
    parser.add_argument('--na-prob-thresh', '-t', type=float, default=1.0,
                        help='Predict "" if no-answer probability exceeds this (default = 1.0).')
    parser.add_argument('--out-image-dir', '-p', metavar='out_images', default=None,
                        help='Save precision-recall curves to directory.')
    parser.add_argument('--verbose', '-v', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_pages_list(dataset):
    pages_list = []
    last_page = None
    for domain in dataset:
        for w in domain['websites']:
            for qa in w['qas']:
                if last_page != qa['id'][:4]:
                    last_page = qa['id'][:4]
                    pages_list.append(last_page)
    return pages_list


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for domain in dataset:
        for w in domain['websites']:
            for qa in w['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def generate_qa_list_for_types(dataset, t) -> list:
    qa_list_for_type = []
    for domain in dataset:
        for w in domain['websites']:
            for qa in w['qas']:
                if qa['type'] == t:
                    qa_list_for_type.append(qa['id'])
    return qa_list_for_type


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_coin(f, t_gold, addition, t_pred, output_exact=False):
    h = BeautifulSoup(open(f))
    p_gold, e_gold = set(), h.find(tid=t_gold)
    if e_gold is None:
        e_pred, e_prev = h.find(tid=t_pred), h.find(tid=t_pred - 1)
        if (e_pred is not None) or \
           (addition == 1 and e_prev is not None) or \
           (addition == 0 and e_prev is None):
            return (0, 0.) if output_exact else 0.
        else:
            return (1, 1.) if output_exact else 1.
    else:
        if int(t_gold) == int(t_pred):
            return (1, 1.) if output_exact else 1.
        p_gold.add(e_gold['tid'])
        for e in e_gold.parents:
            if int(e['tid']) < 2:
                break
            p_gold.add(e['tid'])
        p_pred, e_pred = set(), h.find(tid=t_pred)
        if e_pred is None:
            return 0
        else:
            p_pred.add(e_pred['tid'])
            if e_pred.name != 'html':
                for e in e_pred.parents:
                    if int(e['tid']) < 2:
                        break
                    p_pred.add(e['tid'])
        if output_exact:
            return 0, len(p_gold & p_pred) / len(p_gold | p_pred)
        else:
            return len(p_gold & p_pred) / len(p_gold | p_pred)


def get_raw_scores(dataset, preds, tag_preds, data_dir):
    if preds is None:
        exact_scores = {}
        coin_scores = {}
        for websites in dataset:
            for w in websites['websites']:
                f = os.path.join(data_dir, websites['domain'], w['page_id'][0:2], 'processed_data',
                                 w['page_id'] + '.html')
                for qa in w['qas']:
                    qid = qa['id']
                    gold_tag_answers = [a['element_id'] for a in qa['answers']]
                    additional_tag_information = [a['answer_start'] for a in qa['answers']]
                    if qid not in preds:
                        print('Missing prediction for %s' % qid)
                        continue
                    t_pred = tag_preds[qid]
                    # Take max over all gold answers
                    exact_scores[qid], coin_scores[qid] = max(compute_coin(f, t, a, t_pred, output_exact=True)
                                                              for t, a in zip(gold_tag_answers,
                                                                              additional_tag_information))
        return exact_scores, coin_scores, coin_scores
    exact_scores = {}
    f1_scores = {}
    coin_scores = {}
    for websites in dataset:
        for w in websites['websites']:
            f = os.path.join(data_dir, websites['domain'], w['page_id'][0:2], 'processed_data',
                             w['page_id'] + '.html')
            for qa in w['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                gold_tag_answers = [a['element_id'] for a in qa['answers']]
                additional_tag_information = [a['answer_start'] for a in qa['answers']]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred, t_pred = preds[qid], tag_preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
                coin_scores[qid] = max(compute_coin(f, t, a, t_pred)
                                       for t, a in zip(gold_tag_answers, additional_tag_information))
    return exact_scores, f1_scores, coin_scores


def make_eval_dict(exact_scores, f1_scores, coin_scores, qid_list=None):
    if qid_list is None:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('coin', 100.0 * sum(coin_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        if total == 0:
            return collections.OrderedDict([
                ('exact', 0),
                ('f1', 0),
                ('coin', 0),
                ('total', 0),
            ])
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('coin', 100.0 * sum(coin_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh
    main_eval['has_ans_exact'] = has_ans_exact
    main_eval['has_ans_f1'] = has_ans_f1


def main(OPTS):
    data_dir = os.path.dirname(OPTS.data_file)
    with open(OPTS.data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    if isinstance(OPTS.pred_file, str):
        with open(OPTS.pred_file) as f:
            preds = json.load(f)
    else:
        preds = OPTS.pred_file
    if isinstance(OPTS.tag_pred_file, str):
        with open(OPTS.tag_pred_file) as f:
            tag_preds = json.load(f)
    else:
        tag_preds = OPTS.tag_pred_file
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact, f1, coin = get_raw_scores(dataset, preds, tag_preds, data_dir)
    out_eval = make_eval_dict(exact, f1, coin)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact, f1, coin, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact, f1, coin, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
    print(json.dumps(out_eval, indent=2))
    pages_list, write_eval = make_pages_list(dataset), deepcopy(out_eval)
    for p in pages_list:
        pages_ans_qids = [k for k, _ in qid_to_has_ans.items() if p in k]
        page_eval = make_eval_dict(exact, f1, coin, qid_list=pages_ans_qids)
        merge_eval(write_eval, page_eval, p)
    if OPTS.result_file:
        with open(OPTS.result_file, 'w') as f:
            w = {}
            for k, v in qid_to_has_ans.items():
                w[k] = {'exact': exact[k], 'f1': f1[k], 'coin': coin[k]}
            json.dump(w, f)
    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(write_eval, f)
    return out_eval


if __name__ == '__main__':
    OPTS = parse_args()
    if OPTS.out_image_dir:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    main(OPTS)
