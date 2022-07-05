"""
lukun199@gmail.com
3rd Feb., 2021

# utils.py
"""
import os, sys, time, glob, shutil
import torch
from torch.distributions import Normal

# Communication Utils: Channel Model, error rate, etc.

class Normlize_tx:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def apply(self, _input):
        _dim = _input.shape[1]//2 if self._iscomplex else _input.shape[1]
        _norm = _dim**0.5 / torch.sqrt(torch.sum(_input**2, dim=1))
        return _input*_norm.view(-1,1)

class Channel:
    # returns the message when passed through a channel.
    # AGWN, Fading
    # Note that we need to make sure that the colle map will not change in this
    # step, thus we should not use *= and +=.
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex

    def ideal_channel(self, _input):
        return _input

    def awgn(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5  # for complex signals.
        _input = _input + torch.randn_like(_input) * _std
        #print(_std)
        return _input

    def awgn_physical_layer(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5
        _input = _input + torch.randn_like(_input) * _std
        #print(_std)
        return _input

    def fading(self, _input, _snr):
        # ref from DeepJSCC-f https://github.com/kurka/deepJSCC-feedback
        if self._iscomplex:
            _shape = _input.shape
            _dim = _shape[1]//2
            _std = (10**(-_snr/10.)/2)**0.5
            _mul = torch.abs(torch.randn(_shape[0], 2)/(2**0.5))  # should divide 2**0.5 here.
            _input_ = _input.clone()
            _input_[:,:_dim] *= _mul[:,0].view(-1,1)
            _input_[:,_dim:] *= _mul[:,1].view(-1,1)
            _input = _input_
        else:
            _std = (10**(-_snr/10.))**0.5
            _input = _input * torch.abs(torch.randn(_input.shape[0], 1)).to(_input)
        _input = _input + torch.randn_like(_input) * _std
        #print(_std)
        return _input

    def phase_invariant_fading(self, _input, _snr):
        # ref from DeepJSCC-f
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5
        if self._iscomplex:
            _mul = (torch.randn(_input.shape[0], 1)**2/2. + torch.randn(_input.shape[0], 1)**2/2.)**0.5
        else:
            _mul = (torch.randn(_input.shape[0], 1)**2 + torch.randn(_input.shape[0], 1)**2) ** 0.5
        _input = _input * _mul.to(_input)
        _input = _input +  torch.randn_like(_input) * _std
        #print(_std)
        return _input

# Other Utils:

class Crit:

    def __call__(self, mode, *args):
        return getattr(self, '_' + mode)(*args)

    def _ce(self, pred, target, lengths):
        mask = pred.new_zeros(len(lengths), target.size(1))
        for i, l in enumerate(lengths): # length can be bigger than the true len
            mask[i, :l] = 1

        loss = - pred.gather(2, target.unsqueeze(2)).squeeze(2) * mask   # log counted.
        loss = torch.sum(loss) / torch.sum(mask)  # ln(vocab_dim=24064) is around 10
        return loss

    def _rl(self, seq_logprobs, seq_masks, reward):
        # seq_logprobs[bs, T] reward:[bs, 1] or [bs, T]
        output = - seq_logprobs * seq_masks * reward
        output = torch.sum(output) / torch.sum(seq_masks)
        return output

    def _tx_gaussian_sample(self, log_samples, reward):
        return -(log_samples*reward).mean()

# SCSIU
class GaussianPolicy():

    def forward(self, x, std=0.1):
        return x + torch.randn_like(x) * std

    def forward_sample(self, mean, std=0.1):
        dist = Normal(mean, std)
        action = dist.sample()
        ln_prob = dist.log_prob(action)
        return action, ln_prob

# others

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def time_consum_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(func.__name__, 'is running')
        res = func(*args, **kwargs)
        print("time func takes", time.time() - start)
        return res
    return wrapper


def smaple_n_times(n, x):
        if n>1:
            x = x.unsqueeze(1) # Bx1x...
            x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
            x = x.reshape(x.shape[0]*n, *x.shape[2:])
        return x

def copyStage1ckpts(frompath, topath, strs='resume_from_ce_'):
    os.makedirs(topath, exist_ok=True)
    files = glob.glob(os.path.join(frompath, '*87.pth'))
    for file in files:
        shutil.copyfile(file, os.path.join(topath, strs+os.path.basename(file)))


if __name__ =='__main__':

    is_complex = False
    n = Normlize_tx(is_complex)
    x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [18., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
    y = n.apply(x)
    print(y)
    #for i in range(x.shape[1]//2):
    #    print(y[:,i], y[:,5+i])

    c = Channel(is_complex)
    # x = torch.ones(2,4)
    z = c.awgn(y,10)
    print(z)
