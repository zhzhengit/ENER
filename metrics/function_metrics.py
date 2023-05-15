# encoding: utf-8


import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel,RobertaModel

from models.classifier import MultiNonLinearClassifier, SingleLinearClassifier
from torch.nn import functional as F

def transform_predictions_to_labels(sequence_input_lst, wordpiece_mask, idx2label_map, input_type="logit"):

    wordpiece_mask = wordpiece_mask.detach().cpu().numpy().tolist()
    if input_type == "logit":
        label_sequence = torch.argmax(F.softmax(sequence_input_lst, dim=2), dim=2).detach().cpu().numpy().tolist()
    elif input_type == "prob":
        label_sequence = torch.argmax(sequence_input_lst, dim=2).detach().cpu().numpy().tolist()
    elif input_type == "label":
        label_sequence = sequence_input_lst.detach().cpu().numpy().tolist()
    else:
        raise ValueError
    output_label_sequence = []
    for tmp_idx_lst, tmp_label_lst in enumerate(label_sequence):
        tmp_wordpiece_mask = wordpiece_mask[tmp_idx_lst]
        tmp_label_seq = []
        for tmp_idx, tmp_label in enumerate(tmp_label_lst):
            if tmp_wordpiece_mask[tmp_idx] != -100:
                tmp_label_seq.append(idx2label_map[tmp_label])
            else:
                tmp_label_seq.append(-100)
        output_label_sequence.append(tmp_label_seq)
    return output_label_sequence

def compute_tagger_span_f1(sequence_pred_lst, sequence_gold_lst):
    sum_true_positive, sum_false_positive, sum_false_negative = 0, 0, 0

    for seq_pred_item, seq_gold_item in zip(sequence_pred_lst, sequence_gold_lst):
        gold_entity_lst = get_entity_from_bmes_lst(seq_gold_item)
        pred_entity_lst = get_entity_from_bmes_lst(seq_pred_item)

        true_positive_item, false_positive_item, false_negative_item = count_confusion_matrix(pred_entity_lst, gold_entity_lst)
        sum_true_positive += true_positive_item
        sum_false_negative += false_negative_item
        sum_false_positive += false_positive_item

    batch_confusion_matrix = torch.tensor([sum_true_positive, sum_false_positive, sum_false_negative], dtype=torch.long)
    return batch_confusion_matrix


def count_confusion_matrix(pred_entities, gold_entities):
    true_positive, false_positive, false_negative = 0, 0, 0
    for span_item in pred_entities:
        if span_item in gold_entities:
            true_positive += 1
            gold_entities.remove(span_item)
        else:
            false_positive += 1
    # these entities are not predicted.
    for span_item in gold_entities:
        false_negative += 1
    return true_positive, false_positive, false_negative


def get_entity_from_bmes_lst(label_list):

    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        if label_list[i] != -100:
            current_label = label_list[i].upper()
        else:
            continue

        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def num_prob(pred_list, gold_list, prob_list):
    
    entity_count = 0
    prob_count = [0]*10
    gold_count = [0]*10
    count = [0]*10
    
    for idx in range(len(gold_list)):
        if gold_list[idx] != 0:
            entity_count += 1
            prob_list[idx] = float(prob_list[idx])
            gold_count[(int((prob_list[idx]-0.01)*10))]+=1

            if gold_list[idx].cuda() == pred_list[idx]:
                count[(int((prob_list[idx]-0.01)*10))]+=1

    for i in range(len(count)):
        prob_count[i]=round((count[i]/(gold_count[i]+1e-20)),2)

    return gold_count, count, prob_count, entity_count

def ECE_Scores(pred_list, gold_list, prob_list):

    ece=0
    gold_count, count, prob_count, entity_count = num_prob(pred_list, gold_list, prob_list)
    for i in range(len(count)):
        ece += abs(prob_count[i]-(0.1*i+0.05))*count[i]/entity_count

    ece = round((ece),10)
    return ece

def span_f1_prune(pred,span_label_ltoken,real_span_mask_ltoken):

    predicts = torch.max(pred, 2)[1]
    probs = pred.max(-1)[0]
    scores = []
    correct_auc = []
    
    sent_num = len(probs)
    for idx in range(sent_num):
        sent_length = len(probs[idx])
        for idy in range(sent_length):
            if span_label_ltoken[idx][idy] != 0 and real_span_mask_ltoken[idx][idy] != 0 :
                scores.append(probs[idx][idy].cpu().detach().numpy())
                if predicts[idx][idy] == span_label_ltoken[idx][idy]:
                    correct_auc.append(1)
                else:
                    correct_auc.append(0)

    span_label_ltoken = span_label_ltoken.cpu()
    predicts = predicts.cpu()
    pred_label_mask = (predicts!=0)  # (bs, n_span)
    all_correct = predicts == span_label_ltoken
    correct = all_correct*pred_label_mask*real_span_mask_ltoken
    correct = torch.where(correct==0, 0, 1)
    all_correct = correct.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(predicts!=0 )
    total_golden = torch.sum(span_label_ltoken!=0)

    return correct_pred, total_pred, total_golden, correct_auc, scores

def auc_roc(pred, uncertainty, sequence_label):

    probs, predicts = torch.max(pred, -1)
    probs = probs.cpu()
    predicts = predicts.cpu()
    scores = []
    correct_auc = []
    
    sent_num = len(probs)
    for idx in range(sent_num):
        sent_length = len(probs[idx])
        for idy in range(sent_length):
            if sequence_label[idx][idy] != -100 and sequence_label[idx][idy] != 0:
                # if  sequence_label[idx][idy] != predicts[idx][idy] and mode == 'oov':
                if  sequence_label[idx][idy] == predicts[idx][idy]:
                    correct_auc.append(1)
                    scores.append(probs[idx][idy].cpu().detach().numpy())
                else:
                # elif  sequence_label[idx][idy] == predicts[idx][idy] and mode == 'ori':
                    correct_auc.append(0)
                    scores.append(probs[idx][idy].cpu().detach().numpy())

    return correct_auc, scores

def auc_ruc(pred, uncertainty, sequence_label):

    probs = uncertainty
    predicts = torch.max(pred, -1)[1].cpu()
    scores = []
    correct_auc = []
    
    sent_num = len(probs)
    for idx in range(sent_num):
        sent_length = len(probs[idx])
        for idy in range(sent_length):
            if sequence_label[idx][idy] != -100 and sequence_label[idx][idy] != 0:
                # if  sequence_label[idx][idy] != predicts[idx][idy] and mode == 'oov':
                if  sequence_label[idx][idy] != predicts[idx][idy]:
                    correct_auc.append(1)
                    scores.append(probs[idx][idy].cpu().detach().numpy())
                elif  sequence_label[idx][idy] == predicts[idx][idy]:
                # elif  sequence_label[idx][idy] == predicts[idx][idy] and mode == 'ori':
                    correct_auc.append(0)
                    scores.append(probs[idx][idy].cpu().detach().numpy())

    return correct_auc, scores

def uncer_auroc(pred, uncertainty, span_label_ltoken, real_span_mask_ltoken, mode):

    probs = uncertainty
    predicts = torch.max(pred, 2)[1].cpu()
    scores = []
    correct_auc = []
    
    sent_num = len(probs)
    for idx in range(sent_num):
        sent_length = len(probs[idx])
        for idy in range(sent_length):
            if real_span_mask_ltoken[idx][idy] != 0:
                if  predicts[idx][idy] != span_label_ltoken[idx][idy] and mode == 'trans':
                    
                    correct_auc.append(1)
                    scores.append(probs[idx][idy].cpu().detach().numpy())
                    
                elif predicts[idx][idy] == span_label_ltoken[idx][idy] and span_label_ltoken[idx][idy] != 0 and mode == 'ori': 
                    correct_auc.append(0)
                    scores.append(probs[idx][idy].cpu().detach().numpy())
                    
    span_label_ltoken = span_label_ltoken.cpu()
    predicts = predicts.cpu()
    pred_label_mask = (predicts!=0)  # (bs, n_span)
    all_correct = predicts == span_label_ltoken
    correct = all_correct*pred_label_mask*real_span_mask_ltoken
    correct = torch.where(correct==0, 0, 1)
    all_correct = correct.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(predicts!=0 )
    total_golden = torch.sum(span_label_ltoken!=0)

    return correct_pred, total_pred, total_golden, correct_auc, scores

def get_predict_prune(label2idx_list, all_span_word, words,predicts_new,span_label_ltoken,all_span_idxs):

    idx2label = {}
    for labidx in label2idx_list:
        lab, idx = labidx
        idx2label[int(idx)] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,predicts_new,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
    return batch_preds

def num_prob(pred_list, gold_list, prob_list):
    
    entity_count = 0
    prob_count = [0]*10
    gold_count = [0]*10
    count = [0]*10
    
    for idx in range(len(gold_list)):
        if gold_list[idx] != 0:
            entity_count += 1
            prob_list[idx] = float(prob_list[idx])
            gold_count[(int((prob_list[idx]-0.01)*10))]+=1

            if gold_list[idx].cuda() == pred_list[idx]:
                count[(int((prob_list[idx]-0.01)*10))]+=1

    for i in range(len(count)):
        prob_count[i]=round((count[i]/(gold_count[i]+1e-20)),2)

    return gold_count, count, prob_count, entity_count