import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules import CrossEntropyLoss

def evidence_trans(y, mode):
    if mode == 'softplus':
        return F.softplus(y)
    elif mode == 'exp':
        return torch.exp(y)
    elif mode == 'relu':
        return F.relu(y)
    elif mode == 'softmax':
        classifier = torch.nn.Softmax(dim=-1)
        return classifier(y)
    
def get_tagger_one_hot(label, N, off_value, on_value, input_mask, gpu):

    label_new = torch.clamp(label,min=0)
    label_mask= torch.unsqueeze(torch.where(label>=0,1,0), dim=2)
    
    size = list(label.size())
    size.append(N)
    label = label_new.view(-1)
    ones = torch.sparse.torch.eye(N) * on_value
    
    ones = ones.index_select(0, label.cpu())
    ones += off_value
    ones = ones.view(*size)
    if gpu:
        ones = ones.cuda()
    ones = ones*label_mask
    
    return ones

def get_one_hot(label, N, off_value, on_value, gpu, paradigm):
    label_mask= torch.unsqueeze(torch.where(label>=0,1,0), dim=2)
    size = list(label.size())
    size.append(N)
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N) * on_value
    
    ones = ones.index_select(0, label.cpu())
    ones += off_value
    ones = ones.view(*size)
    if gpu:
        ones = ones.cuda()
    if paradigm == 'seqlab':
        ones = ones*label_mask

    return ones

class Span_Evidence(nn.Module):
    """Evidential MSE Loss."""
    def __init__(self, args, num_classes):

        super().__init__()
        self.paradigm = args.paradigm
        self.num_classes = num_classes
        self.etrans_func = args.etrans_func
        self.loss_type = args.loss
        self.annealing_start = args.annealing_start
        self.annealing_step = args.annealing_step
        self.use_span_weight = args.use_span_weight
        self.cross_entropy = CrossEntropyLoss(reduction='none')
        self.total_epoch = args.iteration
        self.eps = 1e-10
        self.with_kldiv = args.with_kl
        self.with_iw = args.with_iw
        self.with_uc = args.with_uc
        self.gpu = args.gpu

    def kl_divergence(self, alpha, beta):

        S_alpha = torch.sum(alpha, dim=2, keepdim=False)
        S_beta = torch.sum(beta, dim=2, keepdim=False)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=2, keepdim=False)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=2, keepdim=False) - torch.lgamma(S_beta)
        dg0 = torch.digamma(torch.sum(alpha, dim=2, keepdim=True))
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=2, keepdim=False) + lnB + lnB_uni

        return kl

    def ce_loss(self, logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx):
        '''

        :param all_span_rep: shape: (bs, n_span, n_class)
        :param span_label_ltoken:
        :param real_span_mask_ltoken:
        :return:
        '''
        span_label_ltoken = span_label_ltoken.cuda()
        real_span_mask_ltoken = real_span_mask_ltoken.cuda()
        pred = evidence_trans(logits, 'softmax')
        batch_size, n_span = span_label_ltoken.size()
        all_span_rep1 = logits.view(-1,self.num_classes)
        span_label_ltoken1 = span_label_ltoken.view(-1)
        
        loss = self.cross_entropy(all_span_rep1, span_label_ltoken1)
        loss = loss.view(batch_size, n_span)

        if self.use_span_weight: # when training we should multiply the span-weight
            span_weight = loadall[6]
            loss = loss*span_weight.cuda()
        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())
        loss= torch.mean(loss)
        # print("loss: ", loss)

        return loss, pred

    def edl_loss(self, logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx):
        '''
        :param all_span_rep: shape: (bs, n_span, n_class)
        :param span_label_ltoken:
        :param real_span_mask_ltoken:
        :return:
        '''
        # predict = self.classifier(all_span_rep) # shape: (bs, n_span, n_class)
        annealing_coef_step, annealing_coef_exp = self.compute_annealing_coef(idx)
        span_label_ltoken = span_label_ltoken.cuda()
        real_span_mask_ltoken = real_span_mask_ltoken.cuda()

        soft_output = get_one_hot(span_label_ltoken, self.num_classes, 0, 1, self.gpu, paradigm='span') # one-hot label
        alpha = evidence_trans(logits, self.etrans_func) + 1
        evidence = alpha -1
        S = torch.sum(alpha, dim=2, keepdim=True)  # Dirichlet strength
        bu = evidence/S

        alp = evidence * (1-soft_output) + 1 # items of evidence to be punished
        beta = torch.ones_like(alp) # Uniform Distribution

        _ , pred_cls = torch.max(alpha / S, 2, keepdim=False)
        uncertainty = self.num_classes / torch.sum(alpha, dim=2, keepdim=False)
        acc_match = torch.eq(pred_cls, span_label_ltoken).float()

        lam = (1-bu) if self.with_iw == True else 1
        A = torch.sum(lam * soft_output * (torch.digamma(S) - torch.digamma(alpha)), dim=2, keepdim=False) 
        B = self.kl_divergence(alp, beta) * annealing_coef_step if self.with_kldiv == True else 0
        C = (1 - acc_match) * torch.log(uncertainty + 1e-20) * annealing_coef_step if self.with_uc == True else 0

        loss = (A + B - C)

        if self.use_span_weight: # when training we should multiply the span-weight
            span_weight = loadall[6]
            loss = loss*span_weight.cuda()

        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())
        loss= loss.sum()

        return loss, alpha/S

    def compute_annealing_coef(self, epoch):

        epoch_num, total_epoch = epoch, self.total_epoch
        # annealing coefficient
        annealing_coef_step = torch.min(torch.tensor(
            1.0, dtype=torch.float32), torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32))

        annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
        annealing_coef_exp = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * epoch_num)

        return annealing_coef_step, annealing_coef_exp

    def loss(self, logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx):
        if self.loss_type=='edl':
            return(self.edl_loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx))
        elif self.loss_type=='edl_mse':
            return(self.mes_loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx))
        elif self.loss_type=='ce':
            return(self.ce_loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx))
    
    def pred(self, logits):
        if self.etrans_func=='softmax':
            pred = evidence_trans(logits, 'softmax')
            uncertainty = 1 - torch.max(pred, 2)[0]
            return pred, uncertainty
        else:
            alpha = evidence_trans(logits, self.etrans_func) + 1
            S = torch.sum(alpha, dim=2, keepdim=True)  # Dirichlet strength
            uncertainty = self.num_classes / torch.sum(alpha, dim=2, keepdim=False)
            pred = alpha/S
            return pred, uncertainty
        
    def ece_value(self, logits, span_label_ltoken, real_span_mask_ltoken):
        
        mask = torch.where(real_span_mask_ltoken>0, 1, 0).eq(0)
        mask = mask.cuda()
        span_label_ltoken = span_label_ltoken.cuda()
        if self.etrans_func=='softmax':
            pred = evidence_trans(logits, 'softmax')
        else:
            alpha = evidence_trans(logits, self.etrans_func) + 1
            S = torch.sum(alpha, dim=2, keepdim=True)  # Dirichlet strength
            _ , pred_cls = torch.max(alpha / S, 2, keepdim=False)
            pred = alpha/S
        
        pred_scores, pred_cls = torch.max(pred, -1, keepdim=False)

        pred_cls = torch.masked_select(pred_cls, ~mask)
        pred_scores = torch.masked_select(pred_scores, ~mask)
        tgt_cls = torch.masked_select(span_label_ltoken, ~mask)

        return pred_cls, pred_scores, tgt_cls

class Tagger_Evidence(nn.Module):
    """Evidential MSE Loss."""
    def __init__(self, args, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.etrans_func = args.etrans_func
        self.loss_type = args.loss
        self.annealing_start = args.annealing_start
        self.annealing_step = args.annealing_step
        self.use_span_weight = args.use_span_weight
        self.loss_func = CrossEntropyLoss()
        self.total_epoch = args.iteration
        self.eps = 1e-10
        self.with_kldiv = args.with_kl
        self.with_iw = args.with_iw
        self.with_uc = args.with_uc
        self.gpu = args.gpu

    def kl_divergence(self, alpha, beta):

        S_alpha = torch.sum(alpha, dim=2, keepdim=False)
        S_beta = torch.sum(beta, dim=2, keepdim=False)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=2, keepdim=False)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=2, keepdim=False) - torch.lgamma(S_beta)
        dg0 = torch.digamma(torch.sum(alpha, dim=2, keepdim=True))
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=2, keepdim=False) + lnB + lnB_uni

        return kl

    def mes_loss(self, logits, loadall, sequence_labels, input_mask, epoch):
        
        alpha = evidence_trans(logits, self.etrans_func) + 1
        _, annealing_coef = self.compute_annealing_coef(epoch)
        sequence_labels = sequence_labels.cuda()
        input_mask = input_mask.cuda()
        S = torch.sum(alpha, dim=2, keepdim=True) # Dirichlet strength
        E = alpha - 1
        m = alpha / S
        soft_output = get_tagger_one_hot(sequence_labels, self.num_classes, self.gpu)
        A = torch.sum((soft_output - m) ** 2, dim=2, keepdim=False)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=2, keepdim=False)
        alp = E * (1 - soft_output) + 1
        beta = torch.ones_like(alp)
        C = annealing_coef * self.kl_divergence(alp, beta)
        loss = (A + B) + C

        if self.use_span_weight: # when training we should multiply the span-weight
            span_weight = loadall[6]
            loss = loss*span_weight

        loss = torch.masked_select(loss, input_mask.bool())
        loss= (loss.sum())

        return loss, m

    def ce_loss(self, sequence_logits, sequence_labels, input_mask, epoch):
        if input_mask is not None:
            active_loss = input_mask.view(-1) == 1
            active_logits = sequence_logits.view(-1, self.num_classes)
            active_labels = torch.where(
                active_loss, sequence_labels.view(-1), torch.tensor(self.loss_func.ignore_index).type_as(sequence_labels)
            )
            loss = self.loss_func(active_logits, active_labels.cuda())
        else:
            loss = self.loss_func(sequence_logits.view(-1, self.num_classes), sequence_labels.view(-1))
        return loss, loss

    def edl_loss(self, logits, sequence_labels, input_mask, epoch):
        '''
        :param all_span_rep: shape: (bs, n_span, n_class)
        :param sequence_labels:
        :param input_mask:
        :return:
        '''
        self.batch, self.seqlen = sequence_labels.size()
        # predict = self.classifier(all_span_rep) # shape: (bs, n_span, n_class)
        annealing_coef_step, annealing_coef_exp = self.compute_annealing_coef(epoch)
        logits = logits.reshape(self.batch, -1, self.num_classes)
        sequence_labels = sequence_labels.cuda()
        input_mask = input_mask.cuda()

        soft_output = get_tagger_one_hot(sequence_labels, self.num_classes, 0, 1, input_mask, self.gpu) # one-hot label
        alpha = evidence_trans(logits, self.etrans_func) + 1
        evidence = alpha -1
        S = torch.sum(alpha, dim=2, keepdim=True)  # Dirichlet strength
        bu = evidence/S

        alp = evidence * (1-soft_output) + 1 # items of evidence to be punished
        beta = torch.ones_like(alp) # Uniform Distribution

        _ , pred_cls = torch.max(alpha / S, 2, keepdim=False)
        uncertainty = self.num_classes / torch.sum(alpha, dim=2, keepdim=False)
        acc_match = torch.eq(pred_cls, sequence_labels).float()

        lam = (1-bu) if self.with_iw == True else 1
        A = torch.sum(lam * soft_output * (torch.digamma(S) - torch.digamma(alpha)), dim=2, keepdim=False) 
        B = self.kl_divergence(alp, beta) * annealing_coef_exp if self.with_kldiv == True else 0
        C = (1 - acc_match) * torch.log(uncertainty + 1e-20) * annealing_coef_step if self.with_uc == True else 0

        loss = (A+B-C)

        loss = torch.masked_select(loss, input_mask.bool())
        loss= loss.mean()

        return loss, alpha/S

    def compute_annealing_coef(self, epoch):

        epoch_num, total_epoch = epoch, self.total_epoch
        # annealing coefficient
        annealing_coef_step = torch.min(torch.tensor(
            1.0, dtype=torch.float32), torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32))

        annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
        annealing_coef_exp = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * epoch_num)

        return annealing_coef_step, annealing_coef_exp

    def loss(self, logits, sequence_labels, input_mask, epoch):
        if self.loss_type=='edl':
            return(self.edl_loss(logits, sequence_labels, input_mask, epoch))
        elif self.loss_type=='edl_mse':
            return(self.mes_loss(logits, sequence_labels, input_mask, epoch))
        elif self.loss_type=='ce':
            return(self.ce_loss(logits, sequence_labels, input_mask, epoch))
    
    def pred(self, logits):
                
        if self.etrans_func=='softmax':
            pred = evidence_trans(logits, 'softmax')
            uncertainty = 1 - torch.max(pred, 2)[0]
            return pred, uncertainty
        else:
            alpha = evidence_trans(logits, self.etrans_func) + 1
            S = torch.sum(alpha, dim=-1, keepdim=True)  # Dirichlet strength
            uncertainty = self.num_classes / torch.sum(alpha, dim=-1, keepdim=False)
            
            pred = alpha/S
            
            return pred, uncertainty
    
    def ece_value(self, logits, sequence_label):
        
        mask = torch.where(sequence_label>0, 1, 0).eq(0)
        
        if self.etrans_func=='softmax':
            pred = evidence_trans(logits, 'softmax')
        else:
            alpha = evidence_trans(logits, self.etrans_func) + 1
            S = torch.sum(alpha, dim=2, keepdim=True)  # Dirichlet strength
            _ , pred_cls = torch.max(alpha / S, 2, keepdim=False)
            pred = alpha/S
            
        pred_scores, pred_cls = torch.max(pred, -1, keepdim=False)

        pred_cls = torch.masked_select(pred_cls, ~mask)
        pred_scores = torch.masked_select(pred_scores, ~mask)
        tgt_cls = torch.masked_select(sequence_label, ~mask)

        return pred_cls, pred_scores, tgt_cls