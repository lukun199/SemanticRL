"""
This work is created by KunLu. Copyright reserved.
lukun199@gmail.com
19th Feb., 2021

# Inference.py
"""
import os, platform, json, time, pickle, sys, argparse
import torch
from math import log
sys.path.append('./')
from data_loader import Dataset_sentence_test, collate_func
from model import LSTMEncoder, LSTMDecoder, Embeds
from utils import Normlize_tx, Channel, smaple_n_times


_snr = 10
_iscomplex = True
channel_dim = 256


device = torch.device("cpu:0")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=24064, num_hidden=128).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 24064).to(device)

encoder = encoder.eval()
decoder = decoder.eval()
embeds_shared = embeds_shared.eval()


normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)


def do_test(input_data, encoder, decoder, normlize_layer, channel, len_batch):

    with torch.no_grad():
        output, _ = encoder(input_data, len_batch)
        output = normlize_layer.apply(output)
        output = channel.awgn(output, _snr=_snr)
        output = decoder.sample_max_batch(output, None)

    return output

SemanticRL_example = ['this is a typical sentence used to check the performance',
                        'this is a typical unk used to check the performance',
                      'this is exactly a long sentence with complex structure which might be a challenge for both',
                       'i have just brought a yellow banana',
                      'a man is holding a giant elephant on his hand',
                      ]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pathCE", type=str, default='./ckpt_AWGN_CE_Stage2')
    parser.add_argument("--ckpt_pathRL", type=str, default='./ckpt_AWGN_RL_SemanticRLv1')  # or './ckpt_AWGN_RL'
    args = parser.parse_args()

    dict_train = pickle.load(open('./train_dict.pkl', 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}
    
    for input_str in SemanticRL_example:

        input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
        input_len = len(input_vector)
        input_vector = torch.tensor(input_vector)
    
        for ckpt_dir in [args.ckpt_pathCE, args.ckpt_pathRL]:
            model_name = os.path.basename(ckpt_dir)
    
            encoder.load_state_dict(torch.load(ckpt_dir + '/encoder_epoch198.pth', map_location='cpu'))
            decoder.load_state_dict(torch.load(ckpt_dir + '/decoder_epoch198.pth', map_location='cpu'))
            embeds_shared.load_state_dict(torch.load(ckpt_dir + '/embeds_shared_epoch198.pth',  map_location='cpu'))
    
            for _ in range(5):
                output = do_test(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                        len_batch=torch.tensor(input_len).view(-1, ))
                output = output.cpu().numpy()[0]
                res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'
                print('result of {}:            {}'.format(model_name, res))
            print('--------------------------------------------------')