"""
This work is created by KunLu. Copyright reserved.
lukun199@gmail.com
19th Feb., 2021

# Inference.py
"""
import os, platform, json, time, argparse, random, sys
sys.path.append('./')
import torch
from math import log
from data_loader import Dataset_sentence_test, collate_func
from model import LSTMEncoder, LSTMDecoder, Embeds
from utils import Normlize_tx, Channel, smaple_n_times
import torch.utils.data as data
from functools import reduce



def onetime_sample_max(ckpt_this, input_data, encoder, decoder, normlize_layer, channel, len_batch, save_dir, eval_times=1):
    result_strs = []
    gt_strs = []
    to_save_dict = {}
    minib = 2048
    input_data_list = input_data.split(minib)
    len_batch_list = len_batch.split(minib)
    out_chunk = []
    gt_chunk = []
    for idxx, train_sent in enumerate(input_data_list):
        train_sent = train_sent[:,1:]
        train_sent, train_len = collate_func((zip(train_sent, len_batch_list[idxx]-1)))  # len minus 1 for LSTM

        with torch.no_grad():
            output, _ = encoder(train_sent, train_len)
            output = normlize_layer.apply(output)
            output = smaple_n_times(eval_times, output)
            if args.channel == 'gaussian':
                output = channel.awgn(output, _snr=_snr)
            elif args.channel == 'fading':
                output = channel.phase_invariant_fading(output, _snr=_snr)
            elif args.channel == 'vary_gaussian':
                output = channel.awgn(output, _snr=_snr + random.uniform(-10, 10))
            else:
                raise NotImplementedError
            
            output = decoder.sample_max_batch(output, x_mask=None)

        out_chunk.append(output.cpu().numpy().tolist())
        gt_chunk.append(smaple_n_times(eval_times, train_sent).cpu().numpy().tolist())

        if idxx % 4 ==0: print('processing {}'.format(idxx*minib))

    for xx in range(len(out_chunk)):
        for x in range(len(out_chunk[xx])):
            gt_strs.append(gt_chunk[xx][x])
            result_strs.append(out_chunk[xx][x])

    to_save_dict['gt_strs'] = gt_strs
    to_save_dict['result_strs'] = result_strs

    json.dump(to_save_dict, open(save_dir + '/ckpt{}.json'.format(ckpt_this), 'w'))




if __name__ == "__main__":

    _snr = 10
    _iscomplex = True
    channel_dim = 256

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./ckpt_AWGN_RL')
    parser.add_argument("--name", type=str, default='Test_AWGN_RL')
    parser.add_argument("--data_root", type=str, default='H:\MASTER')
    parser.add_argument("--epoch_start", type=int, default=198)
    parser.add_argument("--epoch_end", type=int, default=199)
    parser.add_argument("--channel", type=str, default='gaussian')
    args = parser.parse_args()

    data_path = os.path.join(args.data_root, 'Europarl')
    data_train = Dataset_sentence_test(_path=data_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
    embeds_shared = Embeds(vocab_size=data_train.get_dict_len(), num_hidden=128).to(device)
    encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
    decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size=data_train.get_dict_len()).to(
        device)

    encoder = encoder.eval()
    decoder = decoder.eval()
    embeds_shared = embeds_shared.eval()

    normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
    channel = Channel(_iscomplex=_iscomplex)


    #for rl training
    ckpt_dir = args.path
    save_dir = './Evaluation/InferenceResutls/{}'.format(args.name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    all_data = torch.tensor(data_train.data_num).to(device)
    len_batch = torch.tensor(list(map(lambda s: sum(s != 0), all_data))).to(device)

    ss = time.time()

    for idx in range(args.epoch_start, args.epoch_end):

        if idx%3 ==0 and not os.path.exists(save_dir + '/ckpt{}.json'.format(idx)):

            ckpt_this = idx

            print('processing epoch {}'.format(ckpt_this))
            model_path = '_epoch{}.pth'.format(ckpt_this)

            encoder.load_state_dict(torch.load(ckpt_dir + '/encoder' + model_path))##,  map_location='cpu'))
            decoder.load_state_dict(torch.load(ckpt_dir + '/decoder' + model_path))#,  map_location='cpu'))
            embeds_shared.load_state_dict(torch.load(ckpt_dir + '/embeds_shared' + model_path))#,  map_location='cpu'))

            onetime_sample_max(ckpt_this, all_data, encoder, decoder, normlize_layer, channel, len_batch, save_dir)

            print('total time cost at {}: '.format(ckpt_this), time.time()-ss)
        else:
            print('skipping ckpt', idx)

    print('done_all!')

