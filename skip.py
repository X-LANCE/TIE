import argparse
import json
import os

import bs4
from tqdm import tqdm

from utils_evaluate import EVAL_OPTS, main as evaluate


def find_yn(domain, page_id):
    web_id = page_id[2:4]
    page = page_id[2:]
    f = 'data/{}/{}/processed_data/{}.html'.format(domain, web_id, page)
    h = bs4.BeautifulSoup(open(f))
    tag_id = None
    for e in h.descendants:
        if not isinstance(e, bs4.element.Tag):
            continue
        tag_id = e['tid']
    return int(tag_id), h


def main(args):
    data = json.load(open(args.data_file))
    tag_pred = json.load(open(args.tag_pred))
    pred = {}
    for d in tqdm(data['data']):
        for w in d['websites']:
            curr_page, max_tag_id, html = None, None, None
            for q in w['qas']:
                if curr_page is None or curr_page != q['id'][:-5]:
                    curr_page = q['id'][:-5]
                    max_tag_id, html = find_yn(d['domain'], curr_page)
                answer_id = tag_pred[q['id']]
                tag = html.find(tid=answer_id)
                if tag is not None:
                    pred[q['id']] = ' '.join([s.strip() for s in tag.strings if s.strip() != ''])
                else:
                    pred[q['id']] = 'yes' if answer_id - max_tag_id == 2 else 'no'
    opt = EVAL_OPTS(data_file=args.data_file,
                    pred_file=pred,
                    tag_pred_file=tag_pred,
                    result_file=os.path.join(args.out_dir, 'qas_eval_results.json'),
                    out_file=os.path.join(args.out_dir, 'eval_matrix_results'))
    evaluate(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--tag_pred', required=True, type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    args = parser.parse_args()
    main(args)
