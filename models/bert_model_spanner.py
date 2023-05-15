# encoding: utf-8


import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel,RobertaModel

from models.classifier import MultiNonLinearClassifier, SingleLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F

class BertNER(BertPreTrainedModel):
    def __init__(self, config,args):

        super(BertNER, self).__init__(config)
        self.bert = BertModel(config)
        self.args = args
        if 'roberta' in self.args.bert_config_dir:
            self.bert = RobertaModel(config)
            print('use the roberta pre-trained model...')
            
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)

        self.start_outputs = self.start_outputs.cuda()
        self.end_outputs = self.end_outputs.cuda()
    

        self.hidden_size = config.hidden_size

        self.span_combination_mode = self.args.span_combination_mode
        self.max_span_width = args.max_spanLen
        self.n_class = args.n_class
        self.tokenLen_emb_dim = self.args.tokenLen_emb_dim # must set, when set a value to the max_span_width.

        self.start_outputs = self.start_outputs.cuda()
        self.end_outputs = self.end_outputs.cuda()


        print("self.max_span_width: ", self.max_span_width)
        print("self.tokenLen_emb_dim: ", self.tokenLen_emb_dim)

        self._endpoint_span_extractor = EndpointSpanExtractor(config.hidden_size,
                                                              combination=self.span_combination_mode,
                                                              num_width_embeddings=self.max_span_width,
                                                              span_width_embedding_dim=self.tokenLen_emb_dim,
                                                              bucket_widths=True)

        self.linear = nn.Linear(10, 1)
        self.score_func = nn.Softmax(dim=-1)
        self.linear = self.linear.cuda()
        self.score_func = self.score_func.cuda()

        # import span-length embedding
        self.spanLen_emb_dim =args.spanLen_emb_dim
        self.morph_emb_dim = args.morph_emb_dim
        input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim
        if self.args.use_spanLen and not self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.spanLen_emb_dim
        elif not self.args.use_spanLen and self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.morph_emb_dim
        elif  self.args.use_spanLen and self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.spanLen_emb_dim + self.morph_emb_dim

        self.span_embedding = MultiNonLinearClassifier(input_dim, self.n_class,
                                                       config.model_dropout)
        self.span_embedding = self.span_embedding.cuda()

        self.spanLen_embedding = nn.Embedding(args.max_spanLen+1, self.spanLen_emb_dim, padding_idx=0)
        self.morph_embedding = nn.Embedding(len(args.morph2idx_list) + 1, self.morph_emb_dim, padding_idx=0)
        self.spanLen_embedding = self.spanLen_embedding.cuda()
        self.morph_embedding = self.morph_embedding.cuda()

    def forward(self,loadall, all_span_lens, all_span_idxs_ltoken, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
            all_span_idxs: the span-idxs on token-level. (bs, n_span)
            pos_span_mask: 0 for negative span, 1 for the positive span. SHAPE: (bs, n_span)
            pad_span_mask: 1 for real span, 0 for padding SHAPE: (bs, n_span)
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        all_span_idxs_ltoken = all_span_idxs_ltoken.cuda()
        all_span_lens = all_span_lens.cuda()

        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        all_span_rep = self._endpoint_span_extractor(sequence_heatmap, all_span_idxs_ltoken.long()) # [batch, n_span, hidden]
        if not self.args.use_spanLen and not self.args.use_morph:
            evidence = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)

        elif self.args.use_spanLen and not self.args.use_morph:
            spanlen_rep = self.spanLen_embedding(all_span_lens) # (bs, n_span, len_dim)
            spanlen_rep = F.relu(spanlen_rep)
            all_span_rep = torch.cat((all_span_rep, spanlen_rep), dim=-1)
            evidence = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)

        elif not self.args.use_spanLen and self.args.use_morph:
            morph_idxs = loadall[3]
            morph_idxs = morph_idxs.cuda()
            span_morph_rep = self.morph_embedding(morph_idxs) #(bs, n_span, max_spanLen, dim)
            span_morph_rep = torch.sum(span_morph_rep, dim=2) #(bs, n_span, dim)

            all_span_rep = torch.cat((all_span_rep, span_morph_rep), dim=-1)
            evidence = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)

        elif self.args.use_spanLen and self.args.use_morph:

            morph_idxs = loadall[3]
            morph_idxs = morph_idxs.cuda()
            span_morph_rep = self.morph_embedding(morph_idxs) #(bs, n_span, max_spanLen, dim)
            span_morph_rep = torch.sum(span_morph_rep, dim=2) #(bs, n_span, dim)

            spanlen_rep = self.spanLen_embedding(all_span_lens)  # (bs, n_span, len_dim)
            spanlen_rep = F.relu(spanlen_rep)

            all_span_rep = torch.cat((all_span_rep, spanlen_rep, span_morph_rep), dim=-1)
            evidence = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)

        return evidence