import argparse
import json
import logging
import os
import random
from io import open
import numpy as np
from loguru import logger

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
from easydict import EasyDict as edict

import pdb
import sys
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_pretrained_bert.optimization import WarmupLinearSchedule

from vilbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain, ForwardModelsVal
from vilbert.optimization import BertAdam, Adam, Adamax
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import vilbert.utils as utils
import torch.distributed as dist

from vilbert.datasets.retreival_dataset import CiderDataset
from vilbert.vilbert import BertConfig
from vilbert.vilbert import VILBertForVLTasks

from torch.utils.data import random_split
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

from args import get_parser
from utils_hard_negative import print_f1_scores

def create_init_cider_model(args, device):
    timeStamp = '_' + args.vilbert_config_file.split('/')[1].split('.')[0]
    savePath = os.path.join(args.vilbert_output_dir, timeStamp)
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    config = BertConfig.from_json_file(args.vilbert_config_file)

    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)
        print('\n', file=f)
        print(config, file=f)

    if not os.path.exists(args.vilbert_output_dir):
        os.makedirs(args.vilbert_output_dir)

    model = VILBertForVLTasks.from_pretrained(args.vilbert_from_pretrained, config, num_labels=1, default_gpu=True)

    model.to(device)

    return model

def train_cider_predictor(args, model, writer, logger, device, epochId, i, j):
    bert_weight_name = json.load(open("../vilbert_beta/config/" + args.vilbert_bert_model + "_weight_name.json", "r"))
    tokenizer = BertTokenizer.from_pretrained(args.vilbert_bert_model, do_lower_case=True)

    dataset = CiderDataset(args.captions_path, args.tsv_path, args.cider_path, tokenizer)
    coco_val_dataset = CiderDataset(args.val_captions_path, args.tsv_path, args.val_cider_path, tokenizer)
    
    train_dataloader = DataLoader(dataset, batch_size=args.vilbert_batch_size, shuffle=True)
    coco_val_dataloader = DataLoader(coco_val_dataset, batch_size=args.vilbert_batch_size, shuffle=False)
    
    num_train_optimization_steps = (
        (len(dataset) // args.vilbert_batch_size) * args.num_train_epochs
    )
    
    #####################################
    # Already existing code 
    #####################################

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        
    optimizer_grouped_parameters = []
    lr = args.vilbert_learning_rate
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'vil_prediction' in key:
                lr = 1e-4
            else:
                # if args.vision_scratch:
                #     if key[12:] in bert_weight_name:
                #         lr = args.vilbert_learning_rate
                #     else:
                #         lr = 1e-4
                # else:
                lr = args.vilbert_learning_rate
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]
    
    #####################################
    # End of already existing code
    #####################################

    criterion = nn.MSELoss()

    vilbert_optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.vilbert_learning_rate,
        # warmup=args.warmup_proportion,
        t_total=num_train_optimization_steps,
        schedule='warmup_constant',
    )

    model.train()

    for batch in tqdm(train_dataloader):
        i += 1
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, captions, _, input_mask, segment_ids, co_attention_mask, image_id, y = batch
        _, vil_logit, _, _, _, _, _ = \
            model(captions, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)
        loss = torch.sqrt(criterion(vil_logit.squeeze(-1), y.to(device)))
        writer.add_scalar('Vilbert_train_loss', loss, i)
        loss.backward()
        vilbert_optimizer.step()
        model.zero_grad()
        vilbert_optimizer.zero_grad()

    model.eval()
    
    coco_actual_values = []
    coco_predicted_values = []
    for batch in coco_val_dataloader:
        j += 1
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, captions, _, input_mask, segment_ids, co_attention_mask, image_id, y = batch
        _, vil_logit, _, _, _, _, _ = \
            model(captions, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)
        coco_actual_values += y.tolist()
        loss = torch.sqrt(criterion(vil_logit.squeeze(-1), y.to(device)))
        coco_predicted_values += vil_logit.squeeze(-1).tolist()
        writer.add_scalar('Vilbert_val_loss', loss, j)

    correlation_value = np.corrcoef(coco_predicted_values, coco_actual_values)
    logger.debug("Correlation is " + str(correlation_value))

    # Save a trained model
    model_to_save = (model.module if hasattr(model, "module") else model)

    if not os.path.exists(args.vilbert_output_dir):
        os.makedirs(args.vilbert_output_dir)
    output_model_file = os.path.join(args.vilbert_output_dir, "pytorch_model_" + str(epochId) + ".bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    return model, correlation_value, i, j