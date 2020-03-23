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
from utils import print_f1_scores, add_summary_value, eval_cider_and_append_values
from cider_predictor_train import create_init_cider_model, train_cider_predictor

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()

def main():
    parser = get_parser()
    args = parser.parse_args()
    assert len(args.vilbert_output_dir) > 0
    assert torch.cuda.is_available()

    logger.add(args.vilbert_output_dir + '/output_log.log')
    
    device = torch.device("cuda")
    correlation_values_for_predictor = {}

    vilbert_train_iteration = 0
    vilbert_val_iteration = 0

    cider_predictor = create_init_cider_model(args, device)
    
    for epochId in range(args.num_train_epochs):
        logger.debug("Epoch number " + str(epochId))
        logger.debug("Starting cider predictor training")
        cider_predictor, correlation_value, vilbert_train_iteration, vilbert_val_iteration = train_cider_predictor(args, cider_predictor, writer, logger, device, epochId, vilbert_train_iteration, vilbert_val_iteration)
        correlation_values_for_predictor[epochId] = correlation_value
        logger.debug("Finished cider predictor training")

if __name__ == "__main__":
    main()