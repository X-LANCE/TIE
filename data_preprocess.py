import argparse
import json
import os

from tqdm import tqdm
from bs4 import element, BeautifulSoup as bs


def find_direct_contents(node):
    r"""
    Find the direct contents of the inputted node.
    """
    empty = True
    for ch in node.contents:
        if isinstance(ch, element.NavigableString) and ch.strip():
            empty = False
    return empty


def generate_valid_points(html, rect):
    r"""
    Specified the tags which is valid for NPR graphs. The condition is as follows:
        whose direct contents has text tokens;
        which has valid rect recorded in the meta-information json file provided by the dataset;
        which has a non-zero width and a non-zero height.
        
    Arguments:
        html (BeautifulSoup): the Beautiful Soup object of the web page.
        rect (dict): the meta-information provided by the dataset.
    """
    points = {}
    for e in html.descendants:
        if not isinstance(e, element.Tag):
            continue
        if find_direct_contents(e):
            continue
        key = e['tid']
        if key not in rect or 'rect' not in rect[key]:
            continue
        if 0.0 in [rect[key]['rect']['width'], rect[key]['rect']['height']]:
            continue
        points[key] = rect[key]['rect']
    return points


def rect_process(args):
    r"""
    Specified the tags which is valid for NPR graphs. The condition is as follows:
        whose direct contents has text tokens;
        which has valid rect recorded in the meta-information json file provided by the dataset;
        which has a non-zero width and a non-zero height.
    """
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
            with open(os.path.join(d, page_id + '.' + '.html' + '.points.json'), 'w') as o:
                json.dump(position, o)
        print('Successfully finish dir:', d)


def mask_process(args):
    r"""
    Generated the NPR relations between tags and store them as dicts to save disk space.
    """
    def add_edge(d, key, ter):
        if key not in d:
            d[key] = []
        d[key].append(ter)

    print('Start mask process!!!')
    for d, _, fs in os.walk(args.root_dir):
        print('Start process dir:', d)
        for f in tqdm(fs):
            if not f.endswith('.html' + '.points.json'):
                continue
            rect = json.load(open(os.path.join(d, f)))
            tuple_rect = list(rect.items())
            left, right, up, down = {}, {}, {}, {}
            for i in range(len(tuple_rect)):
                curr = tuple_rect[i][1]
                curr_x = (curr['x'], curr['x'] + curr['width'])
                curr_y = (curr['y'], curr['y'] + curr['height'])
                for j in range(i + 1, len(tuple_rect)):
                    att = tuple_rect[j][1]
                    att_x = (att['x'], att['x'] + att['width'])
                    width = min(att_x[1], curr_x[1]) - max(att_x[0], curr_x[0])
                    att_y = (att['y'], att['y'] + att['height'])
                    height = min(att_y[1], curr_y[1]) - max(att_y[0], curr_y[0])
                    if width / min(curr['width'], att['width']) >= args.percentage:
                        if (att_y[0] <= curr_y[0]) or (att_y[1] <= curr_y[1]):
                            add_edge(up, tuple_rect[i][0], tuple_rect[j][0])
                            add_edge(down, tuple_rect[j][0], tuple_rect[i][0])
                        if (att_y[0] >= curr_y[0]) or (att_y[1] >= curr_y[1]):
                            add_edge(down, tuple_rect[i][0], tuple_rect[j][0])
                            add_edge(up, tuple_rect[j][0], tuple_rect[i][0])
                    if height / min(curr['height'], att['height']) >= args.percentage:
                        if (att_x[0] <= curr_x[0]) or (att_x[1] <= curr_x[1]):
                            add_edge(left, tuple_rect[i][0], tuple_rect[j][0])
                            add_edge(right, tuple_rect[j][0], tuple_rect[i][0])
                        if (att_x[0] >= curr_x[0]) or (att_x[1] >= curr_x[1]):
                            add_edge(right, tuple_rect[i][0], tuple_rect[j][0])
                            add_edge(left, tuple_rect[j][0], tuple_rect[i][0])
            with open(os.path.join(d, f.split('.')[0] + '.html' + '.new.spatial.json'), 'w') as o:
                json.dump({'left': left, 'right': right, 'up': up, 'down': down}, o)
        print('Successfully finish dir:', d)
# def mask_process(args):
#     r"""
#     Generated the NPR relations between tags and store them as dicts to save disk space.
#     """
#     def clamp(num, left, right):
#         if num < left:
#             return left
#         if num > right:
#             return right
#         return num
#
#     def add_edge(d, key, ter):
#         if key not in d:
#             d[key] = []
#         d[key].append(ter)
#
#     print('Start mask process!!!')
#     for d, _, fs in os.walk(args.root_dir):
#         print('Start process dir:', d)
#         for f in tqdm(fs):
#             if not f.endswith('.html' + '.points.json'):
#                 continue
#             rect = json.load(open(os.path.join(d, f)))
#             tuple_rect = list(rect.items())
#             left, right, up, down = {}, {}, {}, {}
#             for i in range(len(tuple_rect)):
#                 curr = tuple_rect[i][1]
#                 curr_x = (curr['x'], curr['x'] + curr['width'])
#                 curr_y = (curr['y'], curr['y'] + curr['height'])
#                 for j in range(len(tuple_rect)):
#                     if i == j:
#                         continue
#                     att = tuple_rect[j][1]
#                     att_x = (att['x'], att['x'] + att['width'])
#                     att_x = (clamp(att_x[0], *curr_x), clamp(att_x[1], *curr_x))
#                     width = att_x[1] - att_x[0]
#                     att_y = (att['y'], att['y'] + att['height'])
#                     att_y = (clamp(att_y[0], *curr_y), clamp(att_y[1], *curr_y))
#                     height = att_y[1] - att_y[0]
#                     if width / curr['width'] >= args.percentage:
#                         if att['y'] <= curr_y[0]:
#                             add_edge(up, tuple_rect[i][0], tuple_rect[j][0])
#                         if att['y'] + att['height'] >= curr_y[1]:
#                             add_edge(down, tuple_rect[i][0], tuple_rect[j][0])
#                     if height / curr['height'] >= args.percentage:
#                         if att['x'] <= curr_x[0]:
#                             add_edge(left, tuple_rect[i][0], tuple_rect[j][0])
#                         if att['x'] + att['width'] >= curr_x[1]:
#                             add_edge(right, tuple_rect[i][0], tuple_rect[j][0])
#             with open(os.path.join(d, f.split('.')[0] + '.' + '.html' + '.spatial.json'), 'w') as o:
#                 json.dump({'left': left, 'right': right, 'up': up, 'down': down}, o)
#         print('Successfully finish dir:', d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str,
                        help='The root directory of the raw WebSRC dataset, which contains the HTML files.')
    parser.add_argument('--task', required=True, choices=['rect', 'mask', 'rect_mask'],
                        help='The task to conduct. rect: generate valid points; mask: generate NPR graph based on valid'
                             'points; rect_mask: both of the above.')
    parser.add_argument('--percentage', default=0.5, type=float,
                        help='Hyper-parameter gamma in equation 1.')
    args = parser.parse_args()

    if 'rect' in args.task:
        rect_process(args)
    if 'mask' in args.task:
        mask_process(args)


if __name__ == '__main__':
    main()
