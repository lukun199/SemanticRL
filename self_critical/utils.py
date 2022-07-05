import numpy as np
import torch
import torch.nn as nn
import tqdm

from .cider.pyciderevalcap.ciderD.ciderD import CiderD
from .bleu.bleu import Bleu


def _array_to_str(arr, sos_token, eos_token):
    out = ''
    for i in range(len(arr)):
        if arr[i] == eos_token:
            break
        out += str(arr[i]) + ' '
    out += str(eos_token)  # optional
    return out.strip()


def get_ciderd_scorer_europarl(split_captions, test_data_num, sos_token, eos_token):

    all_caps = np.concatenate((split_captions, test_data_num))
    print('====> get_ciderd_scorer begin, seeing {} sentences'.format(len(all_caps)))

    refs_idxs = []
    for caps in all_caps:
        ref_idxs = []
        ref_idxs.append(_array_to_str(caps, sos_token, eos_token))
        refs_idxs.append(ref_idxs)

    scorer = CiderD(refs_idxs)
    del refs_idxs
    del ref_idxs
    print('====> get_ciderd_scorer end')
    return scorer

def get_bleu_scorer_europarl(n=4):

    scorer = Bleu(n=n)
    print('====> get_bleu_scorer end')

    return scorer

def get_self_critical_reward_sc(sample_captions, fns, ground_truth,
                             sos_token, eos_token, scorer):
    # the first dim of fns are the same with samples. fns is a list.
    device = sample_captions.device
    batch_size = len(ground_truth)
    seq_per_img = len(fns) // batch_size
    sample_captions = sample_captions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    max_seq_len = sample_captions.shape[1]
    sample_result = []
    gts = {}
    # first multiple samples.
    for fn in fns:
        sample_result.append({'image_id': fn, 'caption': [_array_to_str(sample_captions[fn], sos_token, eos_token)]})
        caps = []
        caps.append(_array_to_str(ground_truth[fn//seq_per_img][:max_seq_len], sos_token, eos_token))
        gts[fn] = caps

    if isinstance(scorer, CiderD):
        _, scores = scorer.compute_score(gts, sample_result)  # [bs*5,1]
        scores = torch.from_numpy(scores).to(device).view(-1, seq_per_img)  # [bs,5]
        detailed_reward = None
    elif isinstance(scorer, Bleu):
        _, scores_mat = scorer.compute_score(gts, sample_result)
        scores_b1 = np.array(scores_mat[0]).mean()
        scores_b2 = np.array(scores_mat[1]).mean()
        scores_b3 = np.array(scores_mat[2]).mean()
        scores_b4 = np.array(scores_mat[3]).mean()
        detailed_reward = (scores_b1, scores_b2, scores_b3, scores_b4)
        scores = (np.array(scores_mat[0]) + np.array(scores_mat[3]))/2
        scores = torch.from_numpy(scores).to(device).view(-1, seq_per_img)  # [bs,5]

    scores.requires_grad = False
    baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1) # [bs,5]
    scores = scores - baseline # [bs,5]
    scores = scores.view(-1, 1)  # [bs*5, 1]
    return scores, baseline.mean(), detailed_reward


def get_self_critical_reward_newsc_TXRL(sample_captions, fns, ground_truth,
                                        sos_token, eos_token, scorer):
    # the first dim of fns are the same with samples. fns is a list.
    device = sample_captions.device
    batch_size = len(ground_truth)
    seq_per_img = len(fns) // batch_size
    sample_captions = sample_captions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    max_seq_len = sample_captions.shape[1]
    sample_result = []
    gts = {}
    # first multiple samples.
    for fn in fns:
        sample_result.append({'image_id': fn, 'caption': [_array_to_str(sample_captions[fn], sos_token, eos_token)]})
        caps = []
        caps.append(_array_to_str(ground_truth[fn // seq_per_img][:max_seq_len], sos_token, eos_token))
        gts[fn] = caps

    _, scores = scorer.compute_score(gts, sample_result)  # [bs*5,1]
    scores = torch.from_numpy(scores).to(device).view(-1, seq_per_img)  # [bs,5]
    scores.requires_grad = False
    if seq_per_img > 1:
        baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)  # [bs,5]
        scores = scores - baseline  # [bs,5]
    scores = scores.view(-1, 1)  # [bs*5, 1]

    if seq_per_img > 1:
        return scores, baseline.mean()
    else:
        return scores, scores.mean()