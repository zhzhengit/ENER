# encoding: utf-8
import argparse
from ast import arg
from cgi import test
import os
from pickletools import optimize
from re import X
from typing import Dict
import torch
from dataloaders.taggerner_dataset import get_labels, get_dataloader
from models.framework import FewShotNERFramework
from dataloaders.spanner_dataset import get_span_labels, get_loader
from models.bert_model_spanner import BertNER
from transformers import AutoTokenizer
from models.config_tagger import BertTaggerConfig
from models.config_spanner import BertNerConfig
import random
import logging

from models.bert_tagger import BertTagger

from models.Evidential_woker import Span_Evidence, Tagger_Evidence
from utils import get_logger, span_parser, tagger_parser
logger = logging.getLogger(__name__)
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    args: argparse.Namespace
    parser = argparse.ArgumentParser(description="Training")

    # basic argument&value
    
    parser.add_argument("--paradigm", default="seqlab", type=str, help="data dir")
    parser.add_argument('--seed', default=-1, type=int, help='')
    args = parser.parse_args()

    if args.seed == -1:
        seed = '%08d' % (random.randint(0, 100000000))
        seed_num = int(seed)
    else:
        seed_num = int(args.seed)

    print('random_int:', seed_num)
    print("Seed num:", seed_num)
    setup_seed(seed_num)

    logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

    if args.paradigm == 'span':
        args = span_parser()
        num_labels = args.n_class
        task_idx2label = None
        args.label2idx_list, args.morph2idx_list = get_span_labels(args)
        bert_config = BertNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         model_dropout=args.model_dropout)
        model = BertNER.from_pretrained(args.bert_config_dir,
                                                        config=bert_config,
                                                        args=args)
        model.cuda()
        
        train_data_loader = get_loader(args, args.data_dir, "train", True)
        dev_data_loader = get_loader(args, args.data_dir,"dev", False)
        test_data_loader = get_loader(args, args.data_dir,"test", False)
        
        test_data_loader_typos = get_loader(args, args.data_dir_typos, "test", False)
        test_data_loader_oov = get_loader(args, args.data_dir_oov,"test", False)
        test_data_loader_ood = get_loader(args, args.data_dir_ood,"test", False)
        edl = Span_Evidence(args, num_labels)

    elif args.paradigm == 'seqlab':
        args = tagger_parser()
        task_labels = get_labels(args.data_sign)
        task_idx2label = {label_idx : label_item for label_idx, label_item in enumerate(get_labels(args.data_sign))}
        num_labels = len(task_labels)
        bert_config = BertTaggerConfig.from_pretrained(args.bert_config_dir,
                                                       hidden_dropout_prob=args.bert_dropout,
                                                       attention_probs_dropout_prob=args.bert_dropout,
                                                       num_labels=num_labels,
                                                       classifier_dropout=args.classifier_dropout,
                                                       classifier_sign=args.classifier_sign,
                                                       classifier_act_func=args.classifier_act_func,
                                                       classifier_intermediate_hidden_size=args.classifier_intermediate_hidden_size)
        tokenizer = AutoTokenizer.from_pretrained(args.bert_config_dir, use_fast=False, do_lower_case=args.do_lowercase)
        model = BertTagger.from_pretrained(args.bert_config_dir, config=bert_config)
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

        model = BertTagger.from_pretrained(args.bert_config_dir, config=bert_config)
        model.cuda()
        train_data_loader = get_dataloader(args,tokenizer,'train',True)
        dev_data_loader = get_dataloader(args,tokenizer,'dev',False)
        test_data_loader = get_dataloader(args,tokenizer,'test',False)
        
        test_data_loader_typos = get_dataloader(args,tokenizer,'typos',False)
        test_data_loader_oov = get_dataloader(args,tokenizer,'oov',False)
        test_data_loader_ood = get_dataloader(args,tokenizer,'ood',False)

        edl = Tagger_Evidence(args, num_labels)

    logger = get_logger(args, seed_num)
    framework = FewShotNERFramework(args, 
                                    logger, 
                                    task_idx2label,
                                    train_data_loader, 
                                    dev_data_loader, 
                                    test_data_loader, 
                                    test_data_loader_typos, 
                                    test_data_loader_oov, 
                                    test_data_loader_ood, 
                                    edl, 
                                    seed_num, 
                                    num_labels=num_labels)
    framework.train(model)

    # if args.load_ckpt:
    #     model = torch.load(args.results_dir+ args.etrans_func + str(seed_num)+'_net_model.pkl')
    #     framework.inference(model)
    
    logger.info("end! 🎉")

if __name__ == '__main__':
    main()

