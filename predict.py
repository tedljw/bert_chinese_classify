# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging
import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from train import convert_examples_to_features
from train import MyPro,ClassificationProcessor
from flask import Flask
from flask import request

import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


ph = None
return_text = True


def init_model(args):
    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'mypro': MyPro, 'classify': ClassificationProcessor}
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    processor.get_train_examples(args.data_dir)

    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank), num_labels=len(label_list))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_save_pth, map_location='cpu')['state_dict'])
    else:
        model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])

    return model, processor, args, label_list, tokenizer, device


class PredictProcessor :
    def __init__(self, model, processor, args, label_list, tokenizer, device):
        self.model = model
        self.processor = processor
        self.args = args
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

    def predict(self, text_list):
        result = []
        print(text_list)
        test_examples = self.processor.get_ifrn_examples(text_list)
        print("test_examples", test_examples[0].text_a)

        test_features = convert_examples_to_features(
            test_examples, self.label_list, self.args.max_seq_length, self.tokenizer, show_exp=False)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.args.eval_batch_size)

        for idx, (input_ids, input_mask, segment_ids) in enumerate(test_dataloader):
            item = {}
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            text = test_examples[idx].text_a
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                print("logits: ", logits)
                logits = F.softmax(logits, dim=1)
                print("logits2: ", logits)
                pred = logits.max(1)[1]
                print("pred: ",pred)
                logits = logits.detach().cpu().numpy()[0].tolist()
                print("logits3", logits)
                if return_text:
                    item['text'] = text
                item['label'] = pred.item()
                item['scores'] = {0: logits[0], 1: logits[1]}
                result.append(item)
        return result


if __name__ == "__main__":
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()
    # required parameters
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default='./data',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='bert-base-chinese',
                        type=str,
                        # required = True,
                        help="choose [bert-base-chinese] mode.")
    parser.add_argument("--task_name",
                        default='MyPro',
                        type=str,
                        # required = True,
                        help="The name of the task to train.")

    parser.add_argument("--model_save_pth",
                        default='checkpoints/bert_classification.pth',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")

    # other parameters
    parser.add_argument("--max_seq_length",
                        default=22,
                        type=int,
                        help="字符串最大长度")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")

    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="验证时batch大小")

    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="用不用CUDA")

    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus.")

    parser.add_argument("--seed",
                        default=777,
                        type=int,
                        help="初始化时的随机数种子")

    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")

    args = parser.parse_args()
    print('[INFO]Init model started.')
    model, processor, args, label_list, tokenizer, device = init_model(args)
    print('[INFO]Init model finished.')
    ph = PredictProcessor(model, processor, args, label_list, tokenizer, device)
    result = ph.predict("请问我的蓝蜂如何手柄使用")
    for res in result:
        print(res)




