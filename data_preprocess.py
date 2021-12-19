import argparse
import json
import os

from tqdm import tqdm
from bs4 import element, BeautifulSoup as bs


def find_direct_contents(e):
    empty = True
    for ch in e.contents:
        if isinstance(ch, element.NavigableString) and ch.strip():
            empty = False
    return empty


def generate_valid_points(h, r):
    points = {}
    for e in h.descendants:
        if not isinstance(e, element.Tag):
            continue
        if find_direct_contents(e):
            continue
        key = e['tid']
        if key not in r or 'rect' not in r[key]:
            continue
        if 0.0 in list(r[key]['rect'].values()):
            continue
        # bb = r[key]['rect']
        # x = bb['x'] + bb['width'] / 2
        # y = bb['y'] + bb['height'] / 2
        points[key] = r[key]['rect']
    return points


def rect_process(args):
    print('Start rect process!!!')
    for d, _, fs in os.walk(args.root_dir):
        print('Start process dir:', d)
        for f in tqdm(fs):
            if not f.endswith('.html'):
                continue
            page_id = f.split('.')[0]
            html = bs(open(os.path.join(d, f)))
            rect = json.load(open(os.path.join(d, page_id + '.json')))
            position = generate_valid_points(html, rect)
            with open(os.path.join(d, page_id + '.points.json'), 'w') as o:
                json.dump(position, o)
        print('Successfully finish dir:', d)


def mask_process(args):
    def clamp(num, left, right):
        if num < left:
            return left
        if num > right:
            return right
        return num

    def add_edge(d, key, ter):
        if key not in d:
            d[key] = []
        d[key].append(ter)

    print('Start mask process!!!')
    for d, _, fs in os.walk(args.root_dir):
        print('Start process dir:', d)
        for f in tqdm(fs):
            if not f.endswith('.points.json'):
                continue
            rect = json.load(open(os.path.join(d, f)))
            # rect_x = json.load(open(os.path.join(d, f)))
            # min_x, min_y = rect_x['2']['x'], rect_x[2]['y']
            # max_x, max_y = rect_x['2']['x'] + rect_x['2']['width'], rect_x[2]['y'] + rect_x[2]['height']
            # rect_y = deepcopy(rect_x)
            # rect_x = sorted(rect_x.items(), key=lambda x: x[1]['x'])
            # rect_y = sorted(rect_y.items(), key=lambda x: x[1]['y'])
            tuple_rect = list(rect.items())
            h, v = {}, {}
            l, r, u, dd = {}, {}, {}, {}
            lw, rw, uw, dw = {}, {}, {}, {}
            for i in range(len(tuple_rect)):
                curr = tuple_rect[i][1]
                curr_x = (curr['x'], curr['x'] + curr['width'])
                curr_y = (curr['y'], curr['y'] + curr['height'])
                for j in range(len(tuple_rect)):
                    if i == j:
                        continue
                    att = tuple_rect[j][1]
                    att_x = (att['x'], att['x'] + att['width'])
                    att_x = (clamp(att_x[0], *curr_x), clamp(att_x[1], *curr_x))
                    width = att_x[1] - att_x[0]
                    att_y = (att['y'], att['y'] + att['height'])
                    att_y = (clamp(att_y[0], *curr_y), clamp(att_y[1], *curr_y))
                    height = att_y[1] - att_y[0]
                    if width / curr['width'] >= args.percentage:
                        add_edge(v, tuple_rect[i][0], tuple_rect[j][0])
                        if att['y'] + att['height'] <= curr_y[0]:
                            add_edge(u, tuple_rect[i][0], tuple_rect[j][0])
                        if att['y'] >= curr_y[1]:
                            add_edge(dd, tuple_rect[i][0], tuple_rect[j][0])
                        if att['y'] <= curr_y[0]:
                            add_edge(uw, tuple_rect[i][0], tuple_rect[j][0])
                        if att['y'] + att['height'] >= curr_y[1]:
                            add_edge(dw, tuple_rect[i][0], tuple_rect[j][0])
                    if height / curr['height'] >= args.percentage:
                        add_edge(h, tuple_rect[i][0], tuple_rect[j][0])
                        if att['x'] + att['width'] <= curr_x[0]:
                            add_edge(l, tuple_rect[i][0], tuple_rect[j][0])
                        if att['x'] >= curr_x[1]:
                            add_edge(r, tuple_rect[i][0], tuple_rect[j][0])
                        if att['x'] <= curr_x[0]:
                            add_edge(lw, tuple_rect[i][0], tuple_rect[j][0])
                        if att['x'] + att['width'] >= curr_x[1]:
                            add_edge(rw, tuple_rect[i][0], tuple_rect[j][0])
            with open(os.path.join(d, f.split('.')[0] + '.spatial.json'), 'w') as o:
                json.dump({'h': h, 'v': v, 'l': l, 'r': r, 'u': u, 'd': dd, 'lw': lw, 'rw': rw, 'uw': uw, 'dw': dw}, o)
        print('Successfully finish dir:', d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str)
    parser.add_argument('--task', required=True, choices=['rect', 'mask', 'rect_mask'])
    parser.add_argument('--percentage', default=0.5, type=float)
    args = parser.parse_args()

    if 'rect' in args.task:
        rect_process(args)
    if 'mask' in args.task:
        mask_process(args)


if __name__ == '__main__':
    main()
