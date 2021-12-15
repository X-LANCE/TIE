import json
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--split', type=str, required=True,
                    help='path of the json file contain the split of dev and test')
args = parser.parse_args()

if not os.path.exists(os.path.join(args.path, 'dev_results')):
    raise FileExistsError('There is not a folder call \'dev_results\''
                          ' which contains the dev results in the provided path')

split = json.load(open(args.split))
assert len(split['dev']) == 10
assert len(split['test']) == 10

if not os.path.exists(os.path.join(args.path, 'final')):
    os.makedirs(os.path.join(args.path, 'final'))

for f in os.listdir(args.path):
    if not f.startswith('qas_eval_results'):
        continue
    suffix = f[:-5].split('_')[-1]
    print('Processing checkpoint {}'.format(suffix))
    final_results = {'dev': {'exact': 0., 'f1': 0., 'pos': 0., 'total': 0},
                     'test': {'exact': 0., 'f1': 0., 'pos': 0., 'total': 0}}
    qas = json.load(open(os.path.join(args.path, f)))
    qas.update(json.load(open(os.path.join(args.path, 'dev_results', f))))
    for k, v in qas.items():
        if k[:4] in split['dev']:
            final_results['dev']['exact'] += v['exact']
            final_results['dev']['f1'] += v['f1']
            final_results['dev']['pos'] += v['coin']
            final_results['dev']['total'] += 1
        elif k[:4] in split['test']:
            final_results['test']['exact'] += v['exact']
            final_results['test']['f1'] += v['f1']
            final_results['test']['pos'] += v['coin']
            final_results['test']['total'] += 1
        else:
            raise ValueError('Website {} is neither in dev not in test'.format(k[:4]))
    for s in ['dev', 'test']:
        for e in ['exact', 'f1', 'pos']:
            final_results[s][e] /= final_results[s]['total']
    suffix = f[:-5].split('_')[-1]
    w = open(os.path.join(args.path, 'final', 'results_{}.json'.format(suffix)), 'w')
    json.dump(final_results, w)
