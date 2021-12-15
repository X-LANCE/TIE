import argparse
import json
import os

import bs4
from bs4 import BeautifulSoup as bs


def parent_calc(h, t_pred):
    p = None
    for t in t_pred:
        p_pred, e_pred = set(), h.find(tid=t)
        if e_pred is not None:
            p_pred.add(int(e_pred['tid']))
            for e in e_pred.parents:
                if type(e) != bs4.element.Tag:
                    assert type(e) == bs4.BeautifulSoup
                    break
                p_pred.add(int(e['tid']))
        else:
            return -1
        if p is None:
            p = p_pred
        else:
            p = p & p_pred
    return max(p)


def path_calc(h, gold_tid, pred_tid, addition):
    tag = h.find(tid=gold_tid)
    if tag is None:
        if type(pred_tid) == list:
            if len(pred_tid) > 1:
                return 0.
            pred_tid = pred_tid[0]
        if h.find(tid=pred_tid) is not None:
            return 0.
        if h.find(tid=pred_tid - 1) is not None and addition == 1:
            return 0.
        if h.find(tid=pred_tid - 1) is None and addition == 0:
            return 0.
        return 1.
    path = [int(gold_tid)]
    for p in tag.parents:
        if type(p) != bs4.element.Tag:
            assert type(p) == bs4.BeautifulSoup
            break
        path.append(int(p['tid']))
    if type(pred_tid) == list:
        pred_tid = parent_calc(h, pred_tid)
    return int(int(pred_tid) in path)


parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--pred_file', type=str, required=True,
                    help='the tag prediction file')
args = parser.parse_args()

dataset = json.load(open(args.data_file))['data']
tag_preds = json.load(open(args.pred_file))
print('Result of {}'.format(args.pred_file))
cnt, total_path_match = 0, 0.
for websites in dataset:
    for w in websites['websites']:
        f = os.path.join(args.data_path, websites['domain'], w['page_id'][0:2], 'processed_data',
                         w['page_id'] + '.html')
        html = bs(open(f))
        for qa in w['qas']:
            qid = qa['id']
            gold_tag_answers = [a['element_id'] for a in qa['answers']]
            additional_tag_information = [a['answer_start'] for a in qa['answers']]
            tag_pred = tag_preds[qid]
            total_path_match += max([path_calc(html, gt, tag_pred, at) for gt, at in zip(gold_tag_answers,
                                                                                         additional_tag_information)])
            cnt += 1
path_match = total_path_match / cnt
print(path_match)

