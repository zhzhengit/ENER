# encoding: utf-8

import torch
from typing import List

def tagger_collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, attention_mask, wordpiece_label_idx_lst
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(3):
        # 0 -> tokens
        # 1 -> token_type_ids
        # 2 -> attention_mask
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # 3 -> sequence_label
    # -100 is ignore_index in the cross-entropy loss function.
    pad_output = torch.full([batch_size, max_length], -100, dtype=batch[0][3].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    # 4 -> is word_piece_label
    pad_output = torch.full([batch_size, max_length], -100, dtype=batch[0][4].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][4]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    return output


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))

    return output

def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
             tokens,type_ids,all_span_idxs_ltoken,morph_idxs, ...
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """

    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    max_num_span = max(x[3].shape[0] for x in batch)
    output = []

    for field_idx in range(2):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # begin{for the pad_all_span_idxs_ltoken... }
    pad_all_span_idxs_ltoken = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append((0,0))
        pad_all_span_idxs_ltoken.append(sma)
    pad_all_span_idxs_ltoken = torch.Tensor(pad_all_span_idxs_ltoken)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][2]
        pad_all_span_idxs_ltoken[sample_idx, : data.shape[0],:] = data
    output.append(pad_all_span_idxs_ltoken)
    # end{for the pad_all_span_idxs_ltoken... }


    # begin{for the morph feature... morph_idxs}
    pad_morph_len = len(batch[0][3][0])
    pad_morph = [0 for i in range(pad_morph_len)]
    pad_morph_idxs = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append(pad_morph)
        pad_morph_idxs.append(sma)
    pad_morph_idxs = torch.LongTensor(pad_morph_idxs)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_morph_idxs[sample_idx, : data.shape[0], :] = data
    output.append(pad_morph_idxs)
    # end{for the morph feature... morph_idxs}


    for field_idx in [4,5,6,7]:
        pad_output = torch.full([batch_size, max_num_span], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    words = []
    for sample_idx in range(batch_size):
        words.append(batch[sample_idx][8])
    output.append(words)


    all_span_word = []
    for sample_idx in range(batch_size):
        all_span_word.append(batch[sample_idx][9])
    output.append(all_span_word)

    all_span_idxs = []
    for sample_idx in range(batch_size):
        all_span_idxs.append(batch[sample_idx][10])
    output.append(all_span_idxs)


    return output
