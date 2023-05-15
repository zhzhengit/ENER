import logging
import time
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_logger(args, seed):
    path =args.logger_dir + args.etrans_func + "{}_{}.txt"
    pathname = path.format(seed, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def tagger_parser():
    args: argparse.Namespace
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--train_batch_size", type=int, default=16, help="training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="testing batch size")
    parser.add_argument("--bert_dropout", type=float, default=0.2, help="bert dropout rate")
    parser.add_argument("--classifier_sign", type=str, default="multi_nonlinear")
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--classifier_act_func", type=str, default="gelu")
    parser.add_argument("--classifier_intermediate_hidden_size", type=int, default=1024)
    parser.add_argument("--chinese", action="store_true", help="is chinese dataset")
    parser.add_argument("--optimizer", choices=["adamw", "torch.adam"], default="torch.adam", help="optimizer type")
    parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
    parser.add_argument("--output_dir", type=str, default="", help="the path for saving intermediate model checkpoints.")
    parser.add_argument("--data_sign", type=str, default="en_conll03", help="data signature for the dataset.")
    parser.add_argument("--polydecay_ratio", type=float, default=4, help="ratio for polydecay learing rate scheduler.")
    parser.add_argument("--do_lowercase", action="store_true", )
    parser.add_argument("--data_file_suffix", type=str, default=".char.bmes")
    parser.add_argument("--lr_scheulder", type=str, default="polydecay")
    parser.add_argument("--lr_mini", type=float, default=-1)
    parser.add_argument("--warmup_proportion", default=0.01, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument('--iteration', default=10, type=int, help='num of iteration')
    parser.add_argument('--accumulate_grad_batche', default=4, type=int, help='')
    parser.add_argument("--paradigm", default="seqlab", type=str)

    parser.add_argument("--data_dir", default='data/tagger_data' ,type=str)
    parser.add_argument("--results_dir", default='model/results' ,type=str)
    parser.add_argument("--logger_dir", default="log/", type=str)
    parser.add_argument("--early_stop", type=int, default=3, help="batch size")
    
    parser.add_argument('--load_ckpt',type=str2bool, default=False, help='')
    
    parser.add_argument("--max_keep_ckpt", default=3, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--bert_config_dir", default="user/", type=str, help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--max_length", type=int, default=256, help="max length of dataset")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.02, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", default=-1, type=int, help="set random seed for reproducing results.")
    parser.add_argument('--gpu',type=str2bool, default=True, help='')
    
    parser.add_argument("--loss", default='edl', type=str, help='train cost function')
    parser.add_argument('--etrans_func', default='exp', type=str, help='type of evidence')
    parser.add_argument('--with_uc',type=str2bool, default=False, help='')
    parser.add_argument('--with_iw', type=str2bool, default=False, help='')
    parser.add_argument('--with_kl', type=str2bool, default=True, help='')
    parser.add_argument('--MCD', type=str2bool, default=False, help='')
    parser.add_argument("--use_span_weight", type=str2bool, default=False, help="range: [0,1.0], the weight of negative span for the loss.")

    parser.add_argument('--annealing_start', default=0.01, type=float, help='num of random')
    parser.add_argument('--annealing_step', default=10, type=float, help='num of random')

    args = parser.parse_args()
    return args

def span_parser():
    args: argparse.Namespace
    parser = argparse.ArgumentParser(description="Training")

    # basic argument&value
    parser.add_argument("--data_dir", default="data/conll03", type=str)
    parser.add_argument("--data_dir_typos", default="data/conll03/typos", type=str)
    parser.add_argument("--data_dir_oov", default="data/conll03/oov", type=str)
    parser.add_argument("--data_dir_ood", default="data/conll03/ood", type=str)
    parser.add_argument("--results_dir", default="model_results/", type=str)
    parser.add_argument("--logger_dir", default="log/", type=str)
    parser.add_argument("--paradigm", default="span", type=str)

    parser.add_argument('--gpu', type=str2bool, default=True, help='gpu')
    parser.add_argument('--iteration', default=30, type=int, help='num of iteration')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--etrans_func', default='exp', type=str, help='type of evidence')
    parser.add_argument("--loss", default='edl', type=str, help='train cost function')
    parser.add_argument('--lr_scheulder', default='linear', type=str, help='(linear,StepLR,OneCycleLR,polydecay)')
    parser.add_argument('--with_uc',type=str2bool, default=True, help='')
    parser.add_argument('--with_iw', type=str2bool, default=False, help='')
    parser.add_argument('--with_kl', type=str2bool, default=True, help='')
    parser.add_argument('--MCD', type=str2bool, default=False, help='')
    parser.add_argument("--lr_mini", type=float, default=-1)
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--polydecay_ratio", type=float, default=4, help="ratio for polydecay learing rate scheduler.")

    parser.add_argument('--annealing_start', default=0.01, type=float, help='num of random')
    parser.add_argument('--annealing_step', default=10, type=float, help='num of random')

    parser.add_argument('--regr', default=2, type=float, help='')
    parser.add_argument('--early_stop', default=5, type=float, help='early stop')
    parser.add_argument('--clip_grad', type=str2bool, default=False, help='clip grad')
    
    parser.add_argument('--load_ckpt', type=str2bool, default=False, help='save model')
    parser.add_argument("--bert_config_dir", default="", type=str, help="bert config dir")
    parser.add_argument("--bert_max_length", default=128, type=int, help="max length of dataset")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-5, type=float,help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--model_dropout", type=float, default=0.2, help="model dropout rate")
    parser.add_argument("--bert_dropout", type=float, default=0.15, help="bert dropout rate")
    parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw")
    parser.add_argument("--dataname", default="conll03", help="the name of a dataset")
    parser.add_argument("--max_spanLen", type=int, default=4, help="max span length")
    parser.add_argument("--n_class", type=int, default=5, help="the classes of a task")
    parser.add_argument("--data_sign", type=str, default="en_onto", help="data signature for the dataset.")
    parser.add_argument("--classifier_sign", type=str, default="multi_nonlinear")
    parser.add_argument("--classifier_act_func", type=str, default="gelu")

    parser.add_argument('--ignore_index', type=int, default=-1,help='label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_tokenLen', default=True, type=str2bool, help='use the token length (after the bert tokenizer process) as a feature',nargs='?',choices=['yes (default)', True, 'no', False])
    parser.add_argument("--tokenLen_emb_dim", default=60, type=int, help="the embedding dim of a span")
    parser.add_argument('--span_combination_mode', default='x,y', help='Train data in format defined by --data-io param.')
    parser.add_argument('--use_spanLen', type=str2bool, default=True, help='use the span length as a feature', nargs='?',choices=['yes (default)', True, 'no', False])
    parser.add_argument("--spanLen_emb_dim", type=int, default=100, help="the embedding dim of a span length")
    parser.add_argument('--use_morph', type=str2bool, default=True, help='use the span length as a feature', nargs='?',choices=['yes (default)', True, 'no', False])
    parser.add_argument("--morph_emb_dim", default=100, type=int,  help="the embedding dim of the morphology feature.")
    parser.add_argument('--morph2idx_list', type=list, help='a list to store a pair of (morph, index).', )
    parser.add_argument('--label2idx_list', type=list, help='a list to store a pair of (label, index).',)
    parser.add_argument('--param_name', type=str, default='param_name', help='a prexfix for a param file name', )
    parser.add_argument('--best_dev_f1', type=float, default=0.0, help='best_dev_f1 value', )
    parser.add_argument("--use_span_weight", type=str2bool, default=False
                        , help="range: [0,1.0], the weight of negative span for the loss.")
    parser.add_argument("--neg_span_weight", type=float,default=0.8,help="range: [0,1.0], the weight of negative span for the loss.")
    args = parser.parse_args()
    
    return args
