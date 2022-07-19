from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json, os, argparse



def proc_and_to_list(input):
    res = []
    tmp = ''
    for x in input:
        if x != 0 and x!= 2:
            tmp+=(' '+str(x))
    res.append(tmp[1:])
    return res


def cal_fun():
    """specify the path of json file, then calculate the metrics."""

    save_path = './Evaluation/EvalResults/{}'.format(args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for x in range(args.epoch_start,args.epoch_end):
        if os.path.exists('{}/ckpt{}.json'.format(args.path, x)) \
                and not os.path.exists(save_path + '/scores_ckpt{}.json'.format(x)):
            score_dict = {}
            score_bleu_dict, score_cider_dict, score_rouge_dict = {}, {}, {}

            with open('{}/ckpt{}.json'.format(args.path, x), 'rb') as file:
                tmp = json.load(file)
                gts, res = {idx: proc_and_to_list(cont) for idx, cont in enumerate(tmp['gt_strs'])}, \
                           {idx: proc_and_to_list(cont) for idx, cont in enumerate(tmp['result_strs'])}
            score_bleu, score_cider, score_rouge = main(gts, res)
            score_bleu_dict[x] = score_bleu
            score_cider_dict[x] = score_cider
            score_rouge_dict[x] = score_rouge

            score_dict['bleu'] = score_bleu_dict
            score_dict['cider'] = score_cider_dict
            score_dict['rouge'] = score_rouge_dict

            json.dump(score_dict, open(save_path + '/scores_ckpt{}.json'.format(x), 'w'))
        else:
            print('-----[*]-----skipping ckpt at ', x)


def bleu(gts, res):
    scorer = Bleu(n=4)
    score, scores = scorer.compute_score(gts, res)
    print('belu = %s' % score)
    return score

def cider(gts, res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)
    return score

def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)
    return score

def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)
    return score

def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)
    return score

def main(gts, res):
    score_bleu = bleu(gts, res)
    score_cider = cider(gts, res)
    #meteor(gts, res)
    score_rouge = rouge(gts, res)
    #spice(gts, res)
    return score_bleu, score_cider, score_rouge

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='')
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--epoch_start", type=int, default=198)
    parser.add_argument("--epoch_end", type=int, default=200)
    args = parser.parse_args()
    cal_fun()