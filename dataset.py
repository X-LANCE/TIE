import os
import json

import torch
import numpy as np
from torch.utils.data import Dataset

from utils import form_tree_mask, form_spatial_mask


# noinspection PyUnresolvedReferences
class _BaseDataset(Dataset):
    def _init_spatial_mask(self, mask_dir, method):
        if method < 2:
            suffix = '.spatial.json'
            if self.simplify:
                suffix = '.simp' + suffix
            else:
                suffix = '.html' + suffix
        elif method > 3:
            suffix = '.advanced.json'
        else:
            self.spatial_mask = None
            return
        print(suffix)
        if isinstance(mask_dir, dict):
            self.spatial_mask = mask_dir
            return
        self.spatial_mask = {}
        for d, _, fs in os.walk(mask_dir):
            for f in fs:
                if not f.endswith(suffix):
                    continue
                domain = d.split('/')[-3][:2]
                page = f.split('.')[0]
                self.spatial_mask[domain + page] = json.load(open(os.path.join(d, f)))
        return


class SubDataset(_BaseDataset):
    def __init__(self, examples, evaluate, total, base_name, args):
        self.examples = examples
        self.evaluate = evaluate
        self.total = total
        self.base_name = base_name
        self.input_file = args.predict_file if evaluate else args.train_file
        self.args = args
        self.simplify = args.simplify

        self.current_part = 1
        self.passed_index = 0
        self.dataset = None
        self.all_html_trees = [e.html_tree for e in self.examples]
        self._init_spatial_mask(os.path.dirname(self.input_file), self.args.mask_method)
        self._read_data()

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
        else:
            all_xpath_tags_seq, all_xpath_subs_seq = None, None

        if self.evaluate:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                                        all_tag_depth, all_xpath_tags_seq, all_xpath_subs_seq,
                                        gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index,
                                        tag2tok=all_tag_to_token,
                                        shape=(self.args.max_tag_length, self.args.max_seq_length),
                                        training=False, page_id=all_page_id, mask_method=self.args.mask_method,
                                        mask_dir=self.spatial_mask,
                                        direction=self.args.direction)
        else:
            all_answer_tid = torch.tensor([f.answer_tid for f in features],
                                          dtype=torch.long if self.args.loss_method == 'base' else torch.float)
            if self.args.merge is not None:
                all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
                all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            else:
                all_start_positions, all_end_positions = None, None
            self.dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_answer_tid, all_tag_depth,
                                        all_xpath_tags_seq, all_xpath_subs_seq,
                                        all_start_positions, all_end_positions,
                                        gat_mask=(all_app_tags, all_example_index, self.all_html_trees),
                                        base_index=all_base_index,
                                        tag2tok=all_tag_to_token,
                                        shape=(self.args.max_tag_length, self.args.max_seq_length),
                                        training=True, page_id=all_page_id, mask_method=self.args.mask_method,
                                        mask_dir=self.spatial_mask,
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


class StrucDataset(_BaseDataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, tag_list=None, gat_mask=None, base_index=None, tag2tok=None, shape=None, training=True,
                 page_id=None, mask_method=1, mask_dir=None, direction=None):
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
        self._init_spatial_mask(mask_dir, self.mask_method)

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]

        tag_to_token_index = self.tag2tok[index]
        app_tags = self.gat_mask[0][index]
        example_index = self.gat_mask[1][index]
        html_tree = self.gat_mask[2][example_index]
        base = self.base_index[index]

        if self.mask_method < 2:
            if self.direction == 'b':
                spa_mask = torch.zeros((4, self.shape[0], self.shape[0]), dtype=torch.long)
            else:
                spa_mask = torch.zeros((2, self.shape[0], self.shape[0]), dtype=torch.long)
            spa_mask[:, :base, :base + len(app_tags) + 1] = 1
            spa_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
            temp = form_spatial_mask(app_tags, self.spatial_mask[self.page_id[index]], self.direction)
            spa_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp)
            output.append(spa_mask)

        temp = torch.tensor(form_tree_mask(app_tags, html_tree, accelerate=self.mask_method != 3))
        dom_mask = torch.zeros((1, self.shape[0], self.shape[0]), dtype=torch.long)
        dom_mask[:, :base, :base + len(app_tags) + 1] = 1
        dom_mask[:, base + len(app_tags), :base + len(app_tags) + 1] = 1
        dom_mask[:, base:base + len(app_tags), base:base + len(app_tags)] = torch.tensor(temp[0])
        output.append(dom_mask)

        pooling_matrix = np.zeros(self.shape, dtype=np.double)
        for i in range(len(tag_to_token_index)):
            temp = tag_to_token_index[i]
            pooling_matrix[i][temp[0]: temp[1] + 1] = 1 / (temp[1] - temp[0] + 1)
        pooling_matrix = torch.tensor(pooling_matrix, dtype=torch.float)
        output.append(pooling_matrix)

        return tuple(item for item in output)

    def __len__(self):
        return len(self.tensors[0])
