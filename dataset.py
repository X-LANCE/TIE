import os
import json

import torch
import numpy as np
from torch.utils.data import Dataset

from utils import form_tree_mask, form_spatial_mask


# noinspection PyUnresolvedReferences
class BaseDataset(Dataset):
    def _init_spatial_mask(self, mask_dir):
        if mask_dir is None:
            self.spatial_mask = None
            return
        if isinstance(mask_dir, dict):
            self.spatial_mask = mask_dir
            return
        self.spatial_mask = {}
        for d, _, fs in os.walk(mask_dir):
            for f in fs:
                if not f.endswith('.spatial.json'):
                    continue
                domain = d.split('/')[-3][:2]
                page = f.split('.')[0]
                self.spatial_mask[domain + page] = json.load(open(os.path.join(d, f)))
        return

    def _init_cnn_feature(self):
        if self.cnn_feature_dir is None:
            self.cnn_feature = None
            print('None!')
            return
        self.cnn_feature = {}
        cnn_feature_dir = os.walk(self.cnn_feature_dir)
        for d, _, fs in cnn_feature_dir:
            for f in fs:
                if f.split('.')[-1] != 'npy':
                    continue
                domain = d.split('/')[-3][:2]
                page = f.split('.')[0]
                temp = torch.as_tensor(np.load(os.path.join(d, f)), dtype=torch.float)
                self.cnn_feature[domain + page] = torch.cat([temp, torch.zeros_like(temp[0]).unsqueeze(0)], dim=0)
        print(len(self.cnn_feature))
        return


class SubDataset(BaseDataset):
    def __init__(self, examples, evaluate, total, base_name, args):
        self.examples = examples
        self.evaluate = evaluate
        self.total = total
        self.base_name = base_name
        self.input_file = args.predict_file if evaluate else args.train_file
        self.args = args

        self.current_part = 1
        self.passed_index = 0
        self.dataset = None
        self.all_html_trees = [e.html_tree for e in self.examples]
        self._read_data()
        self._init_spatial_mask(os.path.dirname(self.input_file) if self.args.mask_method < 2 else None)

    def _read_data(self):
        del self.dataset
        features = torch.load('{}_sub_{}'.format(self.base_name, self.current_part))
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_tag_depth = torch.tensor([f.depth for f in features], dtype=torch.long)
        all_app_tags = [f.app_tags for f in features]
        # all_tag_lists = torch.tensor([f.tag_list for f in features], dtype=torch.long)
        all_example_index = [f.example_index for f in features]
        all_base_index = [f.base_index for f in features]
        all_tag_to_token = [f.tag_to_token_index for f in features]
        all_page_id = [f.page_id for f in features]

        if self.args.model_type == 'markuplm':
            all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in features], dtype=torch.long)
            all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in features], dtype=torch.long)
            if self.args.add_xpath:
                all_xpath_tags_seq_tag = torch.tensor([f.xpath_tags_seq_tag for f in features], dtype=torch.long)
                all_xpath_subs_seq_tag = torch.tensor([f.xpath_subs_seq_tag for f in features], dtype=torch.long)
            else:
                all_xpath_tags_seq_tag, all_xpath_subs_seq_tag = None, None
        else:
            all_xpath_tags_seq, all_xpath_subs_seq, all_xpath_tags_seq_tag, all_xpath_subs_seq_tag = None, None, None, None

        if self.evaluate:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                                        all_tag_depth, all_xpath_tags_seq, all_xpath_subs_seq,
                                        all_xpath_tags_seq_tag, all_xpath_subs_seq_tag,
                                        # tag_list=all_tag_lists,
                                        gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index,
                                        tag2tok=all_tag_to_token,
                                        shape=(self.args.max_tag_length, self.args.max_seq_length),
                                        training=False, page_id=all_page_id, mask_method=self.args.mask_method,
                                        mask_dir=self.spatial_mask,
                                        separate=self.args.separate_mask, cnn_feature_dir=self.args.cnn_feature_dir,
                                        direction=self.args.direction)
        else:
            all_answer_tid = torch.tensor([f.answer_tid for f in features],
                                          dtype=torch.long if self.args.loss_method == 'base' else torch.float)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_answer_tid, all_tag_depth,
                                        all_xpath_tags_seq, all_xpath_subs_seq,
                                        all_xpath_tags_seq_tag, all_xpath_subs_seq_tag,
                                        # tag_list=all_tag_lists,
                                        gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index,
                                        tag2tok=all_tag_to_token,
                                        shape=(self.args.max_tag_length, self.args.max_seq_length),
                                        training=True, page_id=all_page_id, mask_method=self.args.mask_method,
                                        mask_dir=self.spatial_mask,
                                        separate=self.args.separate_mask, cnn_feature_dir=self.args.cnn_feature_dir,
                                        direction=self.args.direction)

    def __getitem__(self, index):
        if index - self.passed_index < 0:
            self.passed_index = 0
            self.current_part = 1
            self._read_data()
        elif index - self.passed_index >= len(self.dataset):
            self.passed_index += len(self.dataset)
            self.current_part += 1
            self._read_data()
        return self.dataset[index - self.passed_index]

    def __len__(self):
        return self.total


class StrucDataset(BaseDataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, tag_list=None, gat_mask=None, base_index=None, tag2tok=None, shape=None, training=True,
                 page_id=None, mask_method=1, mask_dir=None, direction=None, separate=False, cnn_feature_dir=None):
        tensors = tuple(tensor for tensor in tensors if tensor is not None)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.tag_list = tag_list
        self.gat_mask = gat_mask
        self.base_index = base_index
        self.tag2tok = tag2tok
        self.shape = shape
        self.training = training
        self.page_id = page_id
        self.mask_method = mask_method
        self.direction = direction
        self.separate = separate
        self.cnn_feature_dir = cnn_feature_dir
        self._init_spatial_mask(mask_dir)
        self._init_cnn_feature()

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]

        tag_to_token_index = self.tag2tok[index]
        app_tags = self.gat_mask[0][index]
        example_index = self.gat_mask[1][index]
        html_tree = self.gat_mask[2][example_index]
        base = self.base_index[index]

        if self.cnn_feature is not None:
            page_id, ind = self.page_id[index], self.tag_list[index]
            raw_cnn_feature = self.cnn_feature[page_id]
            assert ind.dim() == 1
            cnn_num, cnn_dim = raw_cnn_feature.size()
            ind[ind >= cnn_num] = cnn_num - 1
            ind[ind == -1] = cnn_num - 1
            ind = ind.unsqueeze(1).repeat([1, cnn_dim])
            cnn_feature = torch.gather(raw_cnn_feature, 0, ind)
            output.append(cnn_feature)

        if self.spatial_mask is not None and self.mask_method < 2:
            if self.direction == 'b':
                spa_mask = torch.zeros((4, self.shape[0], self.shape[0]), dtype=torch.long)
            else:
                spa_mask = torch.zeros((2, self.shape[0], self.shape[0]), dtype=torch.long)
            spa_mask[:, :base, :base + len(app_tags) + 1] = 1
            spa_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
            temp = form_spatial_mask(app_tags, self.spatial_mask[self.page_id[index]], self.direction)
            spa_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp)
            output.append(spa_mask)

        gat_mask = torch.zeros((1, self.shape[0], self.shape[0]), dtype=torch.long)
        gat_mask[:, :base, :base + len(app_tags) + 1] = 1
        gat_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
        temp = torch.tensor(form_tree_mask(app_tags, html_tree, separate=self.separate,
                                           accelerate=self.mask_method != 3))
        if self.separate:  # TODO multiple mask and variable total head number implementation
            gat_mask = gat_mask.repeat(2, 1, 1)
            gat_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0:2])
            tree_children = torch.tensor(temp[2])
        else:
            gat_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0])
            tree_children = torch.tensor(temp[1])
        output.append(gat_mask)

        children = torch.zeros((self.shape[0], self.shape[0]), dtype=torch.long)
        tree_children[tree_children.sum(dim=1) == 0, 0] = 1
        children[base:base + len(app_tags), base:base + len(app_tags)] = tree_children
        children[base, 0] = 1
        output.append(children)

        pooling_matrix = np.zeros(self.shape, dtype=np.double)
        for i in range(len(tag_to_token_index)):
            temp = tag_to_token_index[i]
            pooling_matrix[i][temp[0]: temp[1] + 1] = 1 / (temp[1] - temp[0] + 1)
        pooling_matrix = torch.tensor(pooling_matrix, dtype=torch.float)
        output.append(pooling_matrix)

        return tuple(item for item in output)

    def __len__(self):
        return len(self.tensors[0])


class SliceDataset(BaseDataset):
    def __init__(self, file, offsets, html_trees=None, shape=None, training=True, mask_method=1, mask_dir=None,
                 direction=None, separate=False, cnn_feature_dir=None, loss_method='multi-soft'):
        self.file = open(file)
        self.offsets = offsets
        self.html_trees = html_trees
        self.shape = shape
        self.training = training
        self.mask_method = mask_method
        self.mask_dir = mask_dir
        self.direction = direction
        self.separate = separate
        self.cnn_feature_dir = cnn_feature_dir
        self.loss_method = loss_method
        self._init_spatial_mask()
        self._init_cnn_feature()
        if self.training:
            self.tensor_keys = ['input_ids', 'input_mask', 'segment_ids', 'answer_tid',
                                'depth', 'xpath_tags_seq', 'xpath_subs_seq']
        else:
            self.tensor_keys = ['input_ids', 'input_mask', 'segment_ids', 'feature_index',
                                'depth', 'xpath_tags_seq', 'xpath_subs_seq']

    # noinspection PyTypeChecker
    def __getitem__(self, index):
        anchor = self.offsets[index]
        self.file.seek(anchor)
        feature = json.loads(self.file.readline())

        output = []
        for k in self.tensor_keys:
            if k == 'feature_index':
                output.append(torch.tensor(index, dtype=torch.long))
            elif k == 'answer_tid':
                output.append(torch.tensor(feature[k], dtype=torch.long if 'base' in self.loss_method else torch.float))
            else:
                output.append(torch.tensor(feature[k], dtype=torch.long))

        tag_to_token_index = feature['tag_to_token_index']
        app_tags = feature['app_tags']
        example_index = feature['example_index']
        html_tree = self.html_trees[example_index]
        base = feature['base_index']

        if self.cnn_feature is not None:
            page_id, ind = feature['page_id'], torch.tensor(feature["tag_list"], dtype=torch.long)
            raw_cnn_feature = self.cnn_feature[page_id]
            assert ind.dim() == 1
            cnn_num, cnn_dim = raw_cnn_feature.size()
            ind[ind >= cnn_num] = cnn_num - 1
            ind[ind == -1] = cnn_num - 1
            ind = ind.unsqueeze(1).repeat([1, cnn_dim])
            cnn_feature = torch.gather(raw_cnn_feature, 0, ind)
            output.append(cnn_feature)

        if self.spatial_mask is not None and self.mask_method < 2:
            if self.direction == 'b':
                spa_mask = torch.zeros((4, self.shape[0], self.shape[0]), dtype=torch.long)
            else:
                spa_mask = torch.zeros((2, self.shape[0], self.shape[0]), dtype=torch.long)
            spa_mask[:, :base, :base + len(app_tags) + 1] = 1
            spa_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
            temp = form_spatial_mask(app_tags, self.spatial_mask[feature['page_id']], self.direction)
            spa_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp)
            output.append(spa_mask)

        gat_mask = torch.zeros((1, self.shape[0], self.shape[0]), dtype=torch.long)
        gat_mask[:, :base, :base + len(app_tags) + 1] = 1
        gat_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
        temp = torch.tensor(form_tree_mask(app_tags, html_tree, separate=self.separate,
                                           accelerate=self.mask_method != 3))
        if self.separate:  # TODO multiple mask and variable total head number implementation
            gat_mask = gat_mask.repeat(2, 1, 1)
            gat_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0:2])
            tree_children = torch.tensor(temp[2])
        else:
            gat_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0])
            tree_children = torch.tensor(temp[1])
        output.append(gat_mask)

        children = torch.zeros((self.shape[0], self.shape[0]), dtype=torch.long)
        tree_children[tree_children.sum(dim=1) == 0, 0] = 1
        children[base:base + len(app_tags), base:base + len(app_tags)] = tree_children
        children[base, 0] = 1
        output.append(children)

        pooling_matrix = np.zeros(self.shape, dtype=np.double)
        for i in range(len(tag_to_token_index)):
            temp = tag_to_token_index[i]
            pooling_matrix[i][temp[0]: temp[1] + 1] = 1 / (temp[1] - temp[0] + 1)
        pooling_matrix = torch.tensor(pooling_matrix, dtype=torch.float)
        output.append(pooling_matrix)

        return tuple(item for item in output)

    def __len__(self):
        return len(self.offsets)