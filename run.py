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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import timeit

from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer, BertTokenizer, AlbertTokenizer, ElectraTokenizer,
    get_linear_schedule_with_warmup,
)


from model import GraphHtmlConfig, GraphHtmlBert, StrucDataset, SubDataset
from utils import read_squad_examples, convert_examples_to_features, RawResult, write_predictions


# The following import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils_evaluate import EVAL_OPTS, main as evaluate_on_squad

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.separate_read:
        assert args.local_rank == -1
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if args.separate_lr:
        optimizer_grouped_parameters_ptm = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('ptm') and
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if n.startswith('ptm') and
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not n.startswith('ptm') and
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('ptm') and
                        any(nd in n for nd in no_decay) and 'ptm' not in n],
             'weight_decay': 0.0}
        ]
        optimizer_ptm = AdamW(optimizer_grouped_parameters_ptm, lr=args.learning_rate / 10, eps=args.adam_epsilon)
        scheduler_ptm = get_linear_schedule_with_warmup(optimizer_ptm, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=t_total)
    elif args.separate_train:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not n.startswith('ptm') and
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('ptm') and
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids'      : batch[0],
                      'attention_mask' : batch[1],
                      'token_type_ids' : batch[2],
                      'answer_tid'     : batch[3],
                      'start_positions': batch[4],
                      'end_positions'  : batch[5],
                      'tag_depth'      : batch[6],
                      'gat_mask'       : batch[-3],
                      'children'       : batch[-2],
                      'tag_to_tok'     : batch[-1]}
            outputs = model(**inputs)
            loss = args.loss_gamma * outputs[0] + outputs[1]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                if args.separate_lr:
                    optimizer_ptm.step()
                    scheduler_ptm.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, prefix=str(global_step), write_pred=False)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", write_pred=True):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, split=args.evaluate_split)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids'      : batch[0],
                      'attention_mask' : batch[1],
                      'token_type_ids' : batch[2],
                      'tag_depth'      : batch[4],
                      'gat_mask'       : batch[-3],
                      'children'       : batch[-2],
                      'tag_to_tok'     : batch[-1]}
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.loss_method in ['hierarchy', 'multi']:
                result = RawResult(unique_id=unique_id,
                                   tag_logits=(to_list(outputs[1][i]), to_list(outputs[2][i])),
                                   start_logits=to_list(outputs[3][i]),
                                   end_logits=to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id=unique_id,
                                   tag_logits=to_list(outputs[0][i]),
                                   start_logits=to_list(outputs[1][i]),
                                   end_logits=to_list(outputs[2][i]))
            all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_tag_prediction_file = os.path.join(args.output_dir, "tag_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    output_result_file = os.path.join(args.output_dir, "qas_eval_results_{}.json".format(prefix))
    output_file = os.path.join(args.output_dir, "eval_matrix_results_{}".format(prefix))

    returns = write_predictions(args.loss_method, examples, features, all_results, args.n_best_size, 1,
                                args.max_answer_length, args.do_lower_case, output_prediction_file,
                                output_tag_prediction_file, output_nbest_file, args.verbose_logging,
                                write_pred=write_pred)  # TODO n best tag size greater than 1
    if not write_pred:
        output_prediction_file, output_tag_prediction_file = returns

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 tag_pred_file=output_tag_prediction_file,
                                 result_file=output_result_file if write_pred else None,
                                 out_file=output_file)
    results = evaluate_on_squad(evaluate_options)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, split='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    base_cached_features_file = os.path.join(os.path.dirname(input_file), 'cached', 'cached_{}_{}_{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    cached_features_file = '{}_{}'.format(base_cached_features_file, args.loss_method)
    gat_cached_features_file = base_cached_features_file + '_gat'

    if os.path.exists(cached_features_file) and not args.overwrite_cache and not args.enforce:
        logger.info("Loading features from cached file %s", cached_features_file)
        if args.separate_read and not evaluate:
            total = torch.load(cached_features_file + '_total')
            features = None
        else:
            features = torch.load(cached_features_file)
        examples, tag_list = read_squad_examples(input_file=input_file,
                                                 is_training=not evaluate,
                                                 tokenizer=tokenizer,
                                                 simplify=True)
        if not evaluate:
            tag_list = list(tag_list)
            tag_list.sort()
            tokenizer.add_tokens(tag_list)
    else:
        logger.info("Creating features from dataset file at %s", input_file)

        if not evaluate:
            examples, tag_list = read_squad_examples(input_file=input_file,
                                                     is_training=not evaluate,
                                                     tokenizer=tokenizer,
                                                     simplify=True)
            tag_list = list(tag_list)
            tag_list.sort()
            tokenizer.add_tokens(tag_list)

        examples, _ = read_squad_examples(input_file=input_file,
                                          is_training=not evaluate,
                                          tokenizer=tokenizer,
                                          simplify=False)

        if not os.path.exists(gat_cached_features_file):
            os.makedirs(gat_cached_features_file)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                loss_method=args.loss_method,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                max_tag_length=args.max_tag_length,
                                                soft_remain=args.soft_remain,
                                                soft_decay=args.soft_decay,
                                                is_training=not evaluate,
                                                sample_size=args.sample_size,
                                                save_dir=gat_cached_features_file,
                                                no_save=args.enforce)
        if args.local_rank in [-1, 0] and not args.enforce:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            if args.separate_read and not evaluate:
                random.shuffle(features)
                total = len(features)
                num = ((total // 2) // 64) * 64
                torch.save(features[:num], cached_features_file + '_sub1')
                torch.save(features[num:], cached_features_file + '_sub2')
                torch.save(total, cached_features_file + '_total')

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    if args.resave:
        random.shuffle(features)
        total = len(features)
        num = ((total // 2) // 64) * 64
        torch.save(features[:num], cached_features_file + '_sub1')
        torch.save(features[num:], cached_features_file + '_sub2')
        torch.save(total, cached_features_file + '_total')
        raise SystemError('Mission complete!')

    if args.separate_read and not evaluate:
        dataset = SubDataset(examples, evaluate, total, cached_features_file, 2,
                             (args.max_tag_length, args.max_seq_length), args.loss_method, args.separate_mask)
        if evaluate:
            dataset = (dataset, examples, features)
        return dataset

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_tag_depth = torch.tensor([f.depth for f in features], dtype=torch.long)
    all_app_tags = [f.app_tags for f in features]
    all_example_index = [f.example_index for f in features]
    all_html_trees = [e.html_tree for e in examples]
    all_base_index = [f.base_index for f in features]
    all_tag_to_token = [f.tag_to_token_index for f in features]

    if evaluate:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index, all_tag_depth,
                               gat_mask=(all_app_tags, all_example_index, all_html_trees), base_index=all_base_index,
                               tag2tok=all_tag_to_token, shape=(args.max_tag_length, args.max_seq_length),
                               training=False, separate=args.separate_mask)
    else:
        all_answer_tid = torch.tensor([f.answer_tid for f in features],
                                      dtype=torch.long if args.loss_method != 'soft' else torch.float)
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_answer_tid, all_start_positions, all_end_positions, all_tag_depth,
                               gat_mask=(all_app_tags, all_example_index, all_html_trees), base_index=all_base_index,
                               tag2tok=all_tag_to_token, shape=(args.max_tag_length, args.max_seq_length),
                               training=True, separate=args.separate_mask)

    if evaluate:
        dataset = (dataset, examples, features)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the bert or electra models provided by huggingface")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output "
                             "file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending "
                             "with step number")
    parser.add_argument('--eval_from_checkpoint', type=int, default=0,
                        help="Only evaluate the checkpoints with prefix larger than or equal to it, beside the final "
                             "checkpoint with no prefix")
    parser.add_argument('--eval_to_checkpoint', type=int, default=None,
                        help="Only evaluate the checkpoints with prefix smaller than it, beside the final checkpoint "
                             "with no prefix")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--method', type=str, default="base", choices=['base', 'init_direct'])
    parser.add_argument('--enforce', action='store_true')
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--max_tag_length', type=int, default=384)

    # Struc_Config parameters
    parser.add_argument('--num_node_block', type=int, default=3)

    parser.add_argument('--separate_lr', action='store_true')
    parser.add_argument('--separate_train', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='the folder of the checkpoint is provided')
    parser.add_argument('--separate_read', action='store_true')
    parser.add_argument('--resave', action='store_true')
    parser.add_argument('--evaluate_split', type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--max_depth_embeddings', type=str, default=None,
                        help='Set to the max depth embedding for node if want to use the position embeddings')
    parser.add_argument('--loss_method', type=str, default='base', choices=['base', 'soft', 'hierarchy'])
    parser.add_argument('--soft_remain', type=float, default=0.8)
    parser.add_argument('--soft_decay', type=float, default=0.5)
    parser.add_argument('--loss_gamma', type=float, default=1)
    parser.add_argument('--separate_mask', action='store_true')
    parser.add_argument('--resume_from_HPLM', type=str, default=None,
                        help='the path of the folder contains the state dict file and the tokenizer')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path,
                                                          from_tf=bool('.ckpt' in args.model_name_or_path),
                                                          config=config, cache_dir=args.cache_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is
    # set. Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running
    # `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, split='train')
        model.resize_token_embeddings(len(tokenizer))
        if args.resume_from_HPLM is not None:
            model.load_state_dict(torch.load(os.path.join(args.resume_from_HPLM, 'pytorch_model.bin')))
            if args.model_type == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            elif args.model_type == 'electra':
                tokenizer = ElectraTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            else:
                raise NotImplementedError()
        tokenizer.save_pretrained(args.output_dir)
        html_config = GraphHtmlConfig(args, **config.__dict__)
        model = GraphHtmlBert(model, html_config)
        if args.resume is not None:
            model.load_state_dict(torch.load(args.resume), strict=False)
        model.to(args.device)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if args.model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        elif args.model_type == 'electra':
            tokenizer = ElectraTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        else:
            raise NotImplementedError()

        bert_config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                 cache_dir=args.cache_dir)
        bert_model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path,
                                                                   from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                   config=bert_config, cache_dir=args.cache_dir)
        bert_model.resize_token_embeddings(len(tokenizer))

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            try:
                int(global_step)
            except ValueError:
                global_step = ""
            if global_step and int(global_step) < args.eval_from_checkpoint:
                continue
            if global_step and args.eval_to_checkpoint and int(global_step) >= args.eval_to_checkpoint:
                continue
            html_config = GraphHtmlConfig(args, **config.__dict__)
            model = GraphHtmlBert(bert_model, html_config)
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))  # confirmed correct
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
