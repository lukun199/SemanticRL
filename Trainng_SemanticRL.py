"""
lukun199@gmail.com
18th Feb., 2021

# train.py
"""
import os, argparse, yaml, random
import platform
import re

import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
from data_loader import Dataset_sentence, collate_func
from model import get_model
from utils import Normlize_tx, Channel, Crit, clip_gradient, copyStage1ckpts, smaple_n_times, GaussianPolicy
from self_critical.utils import get_ciderd_scorer_europarl, get_bleu_scorer_europarl, get_self_critical_reward_sc, get_self_critical_reward_newsc_TXRL

parser = argparse.ArgumentParser()
parser.add_argument("--snr", type=int, default=10)
parser.add_argument("--multiple_sample", type=int, default=5)  # number of parallel samples
parser.add_argument("--scheduled_sampling_start", type=int, default=18)  # when to start scheduled sampling
parser.add_argument("--iscomplex", type=float, default=1)
parser.add_argument("--channel_type", type=str, default='awgn')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--channel_dim", type=int, default=256)
parser.add_argument("--ckpt_resume", type=int, default=-1)
parser.add_argument("--lr_milestones", type=str, default='160') # update the learning rate
parser.add_argument("--max_epoch", type=int, default=202)
parser.add_argument("--seeds", type=int, default=7)  # not used in the paper. but we can assign it
parser.add_argument("--init_learning_rate", type=float, default=0.0001)
parser.add_argument("--save_model_path", type=str, default="./ckpt_RL/")
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--RL_training", type=float, default=1)
parser.add_argument("--reward_type", type=str, default='CIDEr')  # BLEU, CIDEr
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--backbone", type=str, default="LSTM")  # 'Transformer' 'LSTM'
parser.add_argument("--training_config", type=str, default="")
parser.add_argument("--teacher_forcing", type=float, default=0)
parser.add_argument("--SemanticRL_JSCC", type=float, default=1)  # set 0 for SemanticRL-SCSIU
parser.add_argument("--accumulate_grad", type=float, default=20)  # for variant SCSIU
args = parser.parse_args()
if args.training_config:
    f = yaml.safe_load(open(args.training_config, 'r'))
    for kk, vv in f.items():
        setattr(args, kk, vv)
setattr(args, 'init_learning_rate', float(args.init_learning_rate))
assert args.init_learning_rate<1, 'please check if your learning rate < 1'


# seeds
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
init_seeds(args.seeds)

print('\n[*]---------------args init done')
print('args preview:')
for k in args.__dict__:  print(k + ": " + str(args.__dict__[k]))
print('[*]---------------starting preparing dataset and network. Preparing the dataset may take a few minutes')

os.makedirs(args.save_model_path, exist_ok=True)
device = torch.device(args.device)
opt_decoder = 1
train_loader_params = {'batch_size': args.batch_size,
                       'shuffle': True, 'num_workers':4,  # set to 0 in Windows system
                       'collate_fn': lambda x: collate_func(x),
                       'drop_last': True}  # setting as False should also be OK
data_train = Dataset_sentence(_path = args.dataset_path, use_sos=args.backbone=='Transformer')
train_data_loader = data.DataLoader(data_train,**train_loader_params)


embeds_shared = get_model('Embeds')(vocab_size = data_train.get_dict_len(), num_hidden=128).to(device)
encoder = get_model(args.backbone+'Encoder')(channel_dim=args.channel_dim, embedds=embeds_shared).to(device)
decoder = get_model(args.backbone+'Decoder')(channel_dim=args.channel_dim, embedds=embeds_shared,
                             vocab_size=data_train.get_dict_len()).to(device)
normlize_layer = Normlize_tx(_iscomplex=args.iscomplex)
channel = Channel(_iscomplex=args.iscomplex)
policy = GaussianPolicy()

# print #params             a+b-c since embeds_shared is contained in both TX and RX
nums_model = sum(x.numel() for x in encoder.parameters() if x.requires_grad is True) + \
             sum(x.numel() for x in encoder.parameters() if x.requires_grad is True) - \
             sum(x.numel() for x in embeds_shared.parameters() if x.requires_grad is True)
print("Model {} have {} paramerters in total".format(args.backbone, nums_model))

# load saved ckpt
if args.ckpt_resume>0:
    copyStage1ckpts('./ckpt_{}_CE'.format('AWGN' if args.channel_type=='awgn' else 'FIF'), args.save_model_path)
    embeds_shared.load_state_dict(
        torch.load(args.save_model_path + 'resume_from_ce_embeds_shared_epoch{}.pth'.format(args.ckpt_resume - 1)))
    encoder.load_state_dict(torch.load(args.save_model_path + 'resume_from_ce_encoder_epoch{}.pth'.format(args.ckpt_resume-1)))
    decoder.load_state_dict(torch.load(args.save_model_path + 'resume_from_ce_decoder_epoch{}.pth'.format(args.ckpt_resume-1)))
    print('[*]---------------loaded ckpt at' + args.save_model_path)


# multigpu training with ddp brings a performance degradation. this is perhaps caused by the random number in the channel.
# so we will not provide the ddp version

_params = list(list(embeds_shared.parameters()) + list(decoder.parameters()) + list(encoder.parameters()))
if args.SemanticRL_JSCC:
    optimizer = torch.optim.Adam(_params, lr=args.init_learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[int(x) for x in args.lr_milestones.split(' ')],
                                           gamma=0.5)
else:  # SemanticRL-SCSIU
    # we assign the shared embedding to decoder. this is not a must.
    optimizer_encoder = torch.optim.Adam(list(set(list(encoder.parameters())) - set(list(embeds_shared.parameters()))),
                                         lr=args.init_learning_rate)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args.init_learning_rate)
    scheduler_encoder = optim.lr_scheduler.MultiStepLR(optimizer_encoder,
                                                       milestones=[int(x) for x in args.lr_milestones.split(' ')],
                                                       gamma=0.5)
    scheduler_decoder = optim.lr_scheduler.MultiStepLR(optimizer_decoder,
                                                       milestones=[int(x) for x in args.lr_milestones.split(' ')],
                                                       gamma=0.5)


# loss function
crit = Crit()
print('[*]---------------network config done.')

# reward config for RL training
if args.RL_training:
    reward_scorer = get_ciderd_scorer_europarl(data_train.data_num.numpy(), np.array(data_train.test_data_num),
                                               sos_token=1, eos_token=2) if args.reward_type == 'CIDEr' \
                    else get_bleu_scorer_europarl()


def train(encoder, decoder, device, train_loader, optimizer, epoch):

    # set model as training mode
    embeds_shared.train()
    encoder.train()
    decoder.train()

    print('--------------------epoch: %d' % epoch)

    # scheduled sampling
    frac = (epoch - args.scheduled_sampling_start) // 7  # hyper params. you can set to others
    ss_prob = min(0.05 * frac, 0.25)
    if ss_prob<0: ss_prob=0  # avoid ss_prob<0

    # training loops
    for batch_idx, (train_sents, len_batch) in enumerate(train_loader):
        # Note, for LSTM we feed m=[w_1, w_2, w_3, ...]
        # for Transformer we feed m=[<SOS>, w_1, w_2, w_3, ...]

        # distribute data to device
        train_sents = train_sents.to(device)
        optimizer.zero_grad()
        output, src_mask = encoder(train_sents, len_batch)  # for both LSTM and Transformer
        output = normlize_layer.apply(output)
        output = getattr(channel, args.channel_type)(output, _snr=args.snr)
        if not args.RL_training:
            output = decoder.forward_ce(output, train_sents, src_mask, ss_prob)
            loss = crit('ce', output, train_sents if 'LSTM' in args.backbone else train_sents[:,1:],
                        len_batch if 'LSTM' in args.backbone else [x-1 for x in len_batch])  # remove sos.
        else:
            if args.teacher_forcing>0:
                sample_captions, sample_logprobs, seq_masks = decoder.forward_rl_ssprob(output, train_sents,
                                 sample_max=False, multiple_sample=args.multiple_sample,  x_mask=src_mask)

            else:
                # decoder.sample_max_batch(output, src_mask) # for inference only
                sample_captions, sample_logprobs, seq_masks = decoder.forward_rl(output, sample_max=False,
                                                     multiple_sample=args.multiple_sample, x_mask=src_mask)
            fns = list(range(sample_logprobs.size()[0]))
            advantage, reward_mean, detailed_reward = get_self_critical_reward_sc(sample_captions,
                fns, train_sents if 'LSTM' in args.backbone else train_sents[:,1:], 1, 2, reward_scorer)
            loss = crit('rl', sample_logprobs, seq_masks, advantage)

        # When using Transformer backbone, it is recommended to set args.set teacher_forcing >0, and combine CE and RL loss to 
        # get the best performance. i.e., `loss = loss_ce*lambda + loss_rl` where lambda can be, say, 0.2.
        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()

        if batch_idx%500==0:
            if not args.RL_training:
                print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())
            else:
                print('[%4d / %4d]    ' % (batch_idx, epoch), ' advantage_mean =%.4f,   loss = %.4f,   train_reward = %.4f'
                      % (float(advantage.mean()), loss.item(), float(reward_mean)), detailed_reward)

    if epoch%3==0 or epoch==6: #== 0:
        torch.save(embeds_shared.state_dict(), os.path.join(args.save_model_path, 'embeds_shared_epoch{}.pth'.format(epoch)))
        torch.save(encoder.state_dict(), os.path.join(args.save_model_path, 'encoder_epoch{}.pth'.format(epoch)))
        torch.save(decoder.state_dict(), os.path.join(args.save_model_path, 'decoder_epoch{}.pth'.format(epoch)))
        print("Epoch {} model saved!".format(epoch + 1))


def train_TwoAgents(encoder, decoder, device, train_loader, optimizer_encoder, optimizer_decoder, epoch):
    global opt_decoder
    # if data_parallel: torch.cuda.synchronize()

    print('--------------------epoch: %d' % epoch)

    for batch_idx, (train_sents, len_batch) in enumerate(train_loader):
        # training decoder.
        if opt_decoder:
            print('training decoder.')
            decoder.train()
            with torch.no_grad():  # when training decoder, we fix the encoder and make it deterministic, i.e., std=0
                encoder.eval()
                train_sents = train_sents.to(device)  # with eos
                output_float, src_mask = encoder(train_sents, len_batch)
                output_float = normlize_layer.apply(output_float)
                output = getattr(channel, args.channel_type)(output_float, _snr=args.snr)

            output = smaple_n_times(args.multiple_sample, output)
            # decoder sample with softmax policy
            sample_captions, sample_logprobs, seq_masks = decoder.forward_rl(output, sample_max=False)

            fns = list(range(sample_logprobs.size()[0]))
            reward_decoder, cider_mean_decoder = get_self_critical_reward_newsc_TXRL(
                sample_captions, fns, train_sents, 1, 2, reward_scorer)
            loss_decoder = crit('rl', sample_logprobs, seq_masks, reward_decoder) / args.accumulate_grad

            loss_decoder.backward()
            if (batch_idx + 1) % args.accumulate_grad == 0:
                clip_gradient(optimizer_decoder, 0.1)
                optimizer_decoder.step()
                optimizer_decoder.zero_grad()
                opt_decoder = (opt_decoder + 1) % 2  # switch the training flag

        else:
            print('now we optimizer encoder.')
            # now we optimizer encoder.
            encoder.train()
            train_sents = train_sents.to(device)
            output_float_raw, src_mask = encoder(train_sents, len_batch)
            output_float = normlize_layer.apply(output_float_raw)
            output_float = smaple_n_times(args.multiple_sample, output_float)  # get advantage.
            #  when training encoder, we sample with Gaussian policy
            output_sampled, logprobs = policy.forward_sample(output_float, std=0.1)
            output_sampled = normlize_layer.apply(output_sampled)
            output = getattr(channel, args.channel_type)(output_sampled, _snr=args.snr)

            with torch.no_grad():
                # here we fix the decoder and make it deterministic, i.e., `sample_max=True`
                decoder.eval()
                sample_captions, sample_logprobs, seq_masks = decoder.forward_rl(output, sample_max=True, multiple_sample=1)
                fns = list(range(sample_logprobs.size()[0]))
                reward_encoder, cider_mean_encoder = get_self_critical_reward_newsc_TXRL(  # all is tensor type.
                    sample_captions, fns, train_sents, 1, 2, reward_scorer)
            loss_encoder = crit('tx_gaussian_sample', logprobs, reward_encoder) / args.accumulate_grad

            loss_encoder.backward()
            if (batch_idx + 1) % args.accumulate_grad == 0:
                clip_gradient(optimizer_encoder, 0.1)
                optimizer_encoder.step()
                optimizer_encoder.zero_grad()
                opt_decoder = (opt_decoder + 1) % 2

        if batch_idx % 100 == 0 and opt_decoder == 0:
            # for test only. In this case, both encoder and decoder are deterministic.
            # i.e., (encoder: std=0; decoder: argmax)
            with torch.no_grad():
                output_test = getattr(channel, args.channel_type)(output_float.view(
                                                train_sents.shape[0], args.multiple_sample, -1)[:, 0, :],
                                                _snr=args.snr)
                sample_captions, sample_logprobs, seq_masks = decoder.forward_rl(
                                                output_test, sample_max=True, multiple_sample=1)
                fns = list(range(sample_logprobs.size()[0]))
                _, cider_mean_test = get_self_critical_reward_newsc_TXRL(
                    sample_captions, fns, train_sents, 1, 2, reward_scorer)
            print('[%4d / %4d]    ' % (batch_idx, epoch), ' cider_mean_decoder =%.4f,   loss_decoder = %.4f,  '
                                                          'cider_mean_encoder =%.4f,   loss_encoder = %.4f,   now_cider = %.4f'
                  % (float(cider_mean_decoder), loss_decoder.item(), float(cider_mean_encoder),
                     loss_encoder.item(), float(cider_mean_test)))

    if epoch % 3 == 0:  # == 0:
        torch.save(embeds_shared.state_dict(), os.path.join(args.save_model_path, 'embeds_shared_epoch{}.pth'.format(epoch)))
        torch.save(encoder.state_dict(),
                   os.path.join(args.save_model_path, 'encoder_epoch{}.pth'.format(epoch)))
        torch.save(decoder.state_dict(),
                   os.path.join(args.save_model_path, 'decoder_epoch{}.pth'.format(epoch)))
        print("Epoch {} model saved!".format(epoch + 1))



# start training
print('[*]---------------Start training.')
for epoch in range(args.max_epoch):
    if epoch >= args.ckpt_resume-1:
        if args.SemanticRL_JSCC:
            train(encoder, decoder, device, train_data_loader, optimizer, epoch)
        else:
            train_TwoAgents(encoder, decoder, device, train_data_loader, optimizer_encoder, optimizer_decoder, epoch)
    else:
        print('skipping epoch ', epoch)
    if args.SemanticRL_JSCC:
        scheduler.step()
    else:  # variant SCSIU
        scheduler_encoder.step()
        scheduler_decoder.step()
    


