from ast import arg
import imp
from re import A
import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD
import time
import prettytable as pt
import sys
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from metrics.function_metrics import span_f1_prune, uncer_auroc, ECE_Scores, compute_tagger_span_f1, transform_predictions_to_labels, auc_roc, auc_ruc

class FewShotNERFramework:

    def __init__(self, args, logger, task_idx2label, train_data_loader, val_data_loader, test_data_loader, test_data_loader_typos, test_data_loader_oov, test_data_loader_ood, edl, seed_num, num_labels):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.logger = logger
        # self.gpu = True
        self.seed = seed_num
        self.args = args
        self.eps = 1e-10
        self.learning_rate = args.lr
        self.load_ckpt=args.load_ckpt
        self.optimizer = args.optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss = args.loss
        self.annealing_start = 1e-6
        self.epoch_num = args.iteration
        self.edl = edl
        self.test_data_loader_typos = test_data_loader_typos
        self.test_data_loader_oov = test_data_loader_oov
        self.test_data_loader_ood = test_data_loader_ood
        self.num_labels = num_labels
        self.task_idx2label = task_idx2label
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def metric(self,
                model,
                eval_dataset,
                mode): 
            '''
            model: a FewShotREModel instance
            B: Batch size
            N: Num of classes for each batch
            K: Num of instances for each class in the support set
            Q: Num of instances for each class in the query set
            eval_iter: Num of iterations
            ckpt: Checkpoint path. Set as None if using current model parameters.
            return: Accuracy
            '''

            pred_cnt = 0 # pred entity cnt
            label_cnt = 0 # true label entity cnt
            correct_cnt = 0 # correct predicted entity cnt
            correct_list = []
            scores_list = []
            correct_list_roc = []
            scores_list_roc = []
            
            with torch.no_grad():
                for it, data in enumerate(eval_dataset):
                    
                    gold_tokens_list = []
                    pred_scores_list = []
                    pred_list = []
                    
                    if self.args.paradigm == 'span':
                        tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs =  data
                        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,
                                real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                        attention_mask = (tokens != 0).long()
                        logits = model(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids)
                        predicts, uncertainty = self.edl.pred(logits)
                        correct, tmp_pred_cnt, tmp_label_cnt, roc_correct, roc_scores = span_f1_prune(predicts, span_label_ltoken, real_span_mask_ltoken)
                        _, _, _, ruc_correct, ruc_scores = uncer_auroc(predicts, uncertainty, span_label_ltoken, real_span_mask_ltoken, mode)
                        pred_cls, pred_scores, tgt_cls = self.edl.ece_value(logits, span_label_ltoken, real_span_mask_ltoken)

                        pred_cnt += tmp_pred_cnt
                        label_cnt += tmp_label_cnt
                        correct_cnt += correct

                    elif self.args.paradigm == 'seqlab':
                        token_input_ids, token_type_ids, attention_mask, sequence_labels, is_wordpiece_mask = data
                        batch_size = token_input_ids.shape[0]
                        logits = model(token_input_ids.cuda(), token_type_ids=token_type_ids.cuda(), attention_mask=attention_mask.cuda())
                        logits = logits.reshape(batch_size, -1, self.num_labels)
                        predicts, uncertainty = self.edl.pred(logits)
                        sequence_pred_lst = transform_predictions_to_labels(logits.view(batch_size, -1, self.num_labels), is_wordpiece_mask, self.task_idx2label, input_type="logit")
                        sequence_gold_lst = transform_predictions_to_labels(sequence_labels, is_wordpiece_mask, self.task_idx2label, input_type="label")
                        tp, fp, fn = compute_tagger_span_f1(sequence_pred_lst, sequence_gold_lst)
                        roc_correct, roc_scores = auc_roc(predicts, uncertainty, sequence_labels)
                        ruc_correct, ruc_scores = auc_ruc(predicts, uncertainty, sequence_labels)
                        pred_cls, pred_scores, tgt_cls = self.edl.ece_value(logits, sequence_labels)

                        correct_cnt += tp
                        pred_cnt += (tp+fn)
                        label_cnt += (tp+fp)
                        
                    else:
                        return ValueError
                        
                    # tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs = data
                    # loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights,
                    #         real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                    # attention_mask = (tokens != 0).long()
                    # logits = model(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids)
                    # pred, uncertainty = self.edl.pred(logits)

                    pred_list.append(pred_cls)
                    pred_scores_list.append(pred_scores)
                    gold_tokens_list.append(tgt_cls)
                    # pred_cls, pred_scores, tgt_cls = self.edl.ece_value(logits, span_label_ltoken, real_span_mask_ltoken)
                    correct_list_roc += roc_correct
                    scores_list_roc += roc_scores

                    correct_list += ruc_correct
                    scores_list += ruc_scores
                    
                    
                gold_tokens_cat = torch.cat(gold_tokens_list, dim=0)
                pred_scores_cat = torch.cat(pred_scores_list, dim=0)
                pred_cat = torch.cat(pred_list, dim=0)
                
                ece = ECE_Scores(pred_cat, gold_tokens_cat, pred_scores_cat)
                precision = correct_cnt / pred_cnt + 0.0
                recall = correct_cnt / label_cnt + 0.0
                f1 = 2 * precision * recall / (precision + recall+float("1e-8"))
                
                # if mode != 'ori':
                fpr, tpr, thresholds = metrics.roc_curve(correct_list_roc, scores_list_roc, pos_label=1)
                auc =  metrics.auc(fpr, tpr)

                # self.logger.info('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}, ece: {4:3.4f}'\
                #         .format(it + 1, precision, recall, f1, ece))
                # self.logger.info('「」:{0:4}'.format(auc))

                return precision, recall, f1, ece, auc, correct_list, scores_list
    
    def eval(self,
            model,
            correct_lis_ori=0.0,
            scores_list_ori=0.0,
            mode=None,
            test_mode=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        if mode == 'dev':
            self.logger.info("Use val dataset")
            precision, recall, f1, ece, aucroc, correct_list, scores_list = self.metric(model, self.val_data_loader, mode='dev')
            
            _, _, _, _, _, _, _ = self.metric(model, self.test_data_loader, mode='ori')

            self.logger.info('{} Label F1 {}'.format("dev", f1))
            table = pt.PrettyTable(["{}".format("dev"), "Precision", "Recall", 'F1', 'ECE'])
            table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
            # table.add_row(["AUC-ROC"] + ["{:3.4f}".format(aucroc)])
            self.logger.info("\n{}".format(table))
            return f1

        elif mode == 'test' and test_mode == None:
            self.logger.info("Use test dataset")
            precision, recall, f1, ece, aucroc, correct_list, scores_list = self.metric(model, self.test_data_loader, mode='ori')
            
            self.logger.info('{} Label F1 {}'.format("test", f1))
            table = pt.PrettyTable(["{}".format("test"), "Precision", "Recall", 'F1', 'ECE'])
            table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
            # table.add_row(["AUC-ROC"] + ["{:3.4f}".format(aucroc)])
            self.logger.info("\n{}".format(table))
            
            return f1, correct_list, scores_list

        elif mode == 'test' and test_mode != None:
            
            if test_mode == 'typos':
                self.logger.info("Use typos test dataset")
                precision, recall, f1, ece, aucroc, correct_lis_trans, scores_list_trans = self.metric(model, self.test_data_loader_typos, mode='trans')

            elif test_mode == 'oov':
                self.logger.info("Use oov test dataset")
                precision, recall, f1, ece, aucroc, correct_lis_trans, scores_list_trans = self.metric(model, self.test_data_loader_oov, mode='trans')

            elif test_mode == 'ood':
                self.logger.info("Use ood test dataset")
                precision, recall, f1, ece, aucroc, correct_lis_trans, scores_list_trans = self.metric(model, self.test_data_loader_ood, mode='trans')

            correct_list = correct_lis_trans + correct_lis_ori
            scores_list = scores_list_trans + scores_list_ori
            fpr, tpr, thresholds = metrics.roc_curve(correct_list, scores_list, pos_label=1)
            auc_ruc =  metrics.auc(fpr, tpr)
            
            self.logger.info('{} Label F1 {}'.format(test_mode, f1))
            table = pt.PrettyTable(["{}".format(test_mode), "Precision", "Recall", 'F1', 'ECE'])
            table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
            self.logger.info("\n{}".format(table))
            
            table = pt.PrettyTable(["{}".format(test_mode), "「AUC-ROC」", "「AUC-RUC」"])
            table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [aucroc, auc_ruc]])
            self.logger.info("\n{}".format(table))
        
    def train(self,
              model
              ): 
        self.logger.info("Start training...")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.learning_rate,
                              eps=self.args.adam_epsilon,)
            
        elif self.optimizer == "sgd":
            optimizer = SGD(optimizer_grouped_parameters, self.learning_rate, momentum=0.9)
        
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                        lr=self.args.lr,
                                        eps=self.args.adam_epsilon,
                                        weight_decay=self.args.weight_decay)

        t_total = len(self.train_data_loader) * self.args.iteration
        warmup_steps = int(self.args.warmup_proportion * t_total)

        if self.args.lr_scheulder == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) 
        elif self.args.lr_scheulder == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 1e-8)
        elif self.args.lr_scheulder == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.learning_rate, pct_start=float(self.args.warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheulder == "polydecay":
            if self.args.lr_mini == -1:
                lr_mini = self.args.lr / self.args.polydecay_ratio
            else:
                lr_mini = self.args.lr_mini
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError

        model.train()
        # Training
        best_f1 = 0.0
        best_step = 0
        iter_loss = 0.0

        for idx in range(self.args.iteration):
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0
            epoch_start = time.time()
            self.logger.info("training...")

            for it in range(len(self.train_data_loader)):
                loss = 0
                if self.args.paradigm == 'span':
                    tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs =  next(iter(self.train_data_loader))
                    loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,
                            real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                    attention_mask = (tokens != 0).long()
                    logits = model(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids)
                    loss, pred = self.edl.loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx)
                    correct, tmp_pred_cnt, tmp_label_cnt, _, _ = span_f1_prune(pred, span_label_ltoken, real_span_mask_ltoken)
                    
                    pred_cnt += tmp_pred_cnt
                    label_cnt += tmp_label_cnt
                    correct_cnt += correct

                elif self.args.paradigm == 'seqlab':
                    token_input_ids, token_type_ids, attention_mask, sequence_labels, is_wordpiece_mask = next(iter(self.train_data_loader))
                    batch_size = token_input_ids.shape[0]
                    logits = model(token_input_ids.cuda(), token_type_ids=token_type_ids.cuda(), attention_mask=attention_mask.cuda())
                    loss, _ = self.edl.loss(logits, sequence_labels, attention_mask, idx)
                    sequence_pred_lst = transform_predictions_to_labels(logits.view(batch_size, -1, self.num_labels), is_wordpiece_mask, self.task_idx2label, input_type="logit")
                    sequence_gold_lst = transform_predictions_to_labels(sequence_labels, is_wordpiece_mask, self.task_idx2label, input_type="label")
                    tp, fp, fn = compute_tagger_span_f1(sequence_pred_lst, sequence_gold_lst)
                    
                    correct_cnt += tp
                    pred_cnt += (tp+fn)
                    label_cnt += (tp+fp)
                    
                else:
                    return ValueError
                
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()
                iter_loss += self.item(loss.data)

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            
            precision = correct_cnt / pred_cnt + 0.
            recall = correct_cnt / label_cnt + 0.
            f1 = 2 * precision * recall / (precision + recall+float("1e-8"))

            self.logger.info("Time '%.2f's" % epoch_cost)

            self.logger.info('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                .format(idx+1, iter_loss, precision, recall, f1) + '\r')

            if (idx + 1) % 1 == 0:
                f1 = self.eval(model, mode = 'dev', correct_lis_ori=0.0, scores_list_ori=0.0, test_mode = None)
                # _ = self.eval(model, mode = 'test', test_mode = None)
                self.inference(model)
                if f1>best_f1:
                    best_step = idx + 1
                    best_f1 = f1
                    if self.args.load_ckpt:
                        torch.save(model, self.args.results_dir+ self.args.etrans_func + str(self.seed)+'_net_model.pkl')

                if (idx+1) > best_step + self.args.early_stop:
                    self.logger.info('earlt stop!')
                    return

                # results_dir = self.args.results_dir + str(idx+1) + '_result.txt'
                # sent_num = len(predict_results)
                # fout = open(results_dir, 'w', encoding='utf-8')
                # for idx in range(sent_num):
                #     sent_length = len(predict_results[idx])
                #     for idy in range(sent_length):
                #         results += (str(int(predict_results[idx][idy])), " ", str(int(batch_soft[idx][idy])), " ", str(float(prob_results[idx][idy])), " ", str(float(uk[idx][idy])), '\n')
                #         fout.write("".join(results))
                #         results = []
                #     fout.write('\n')
                # fout.close()

            iter_loss = 0.
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0
    
    def inference(self, model):
        f1, correct_lis_ori, scores_list_ori = self.eval(model, mode = 'test', correct_lis_ori=0.0, scores_list_ori=0.0, test_mode = None)

        # _ = self.eval(model, mode = 'test', test_mode = None)
        
        # self.eval(model, mode = 'test', correct_lis_ori = correct_lis_ori, scores_list_ori = scores_list_ori, test_mode = 'typos')
        # self.eval(model, mode = 'test', correct_lis_ori = correct_lis_ori, scores_list_ori = scores_list_ori, test_mode = 'oov')
        # self.eval(model, mode = 'test', correct_lis_ori = correct_lis_ori, scores_list_ori = scores_list_ori, test_mode = 'ood')