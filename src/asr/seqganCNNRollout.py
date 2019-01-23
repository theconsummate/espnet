import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import copy
import numpy as np

from asr_utils import clip_sequence, convert_ys_hat_prob_to_seq


# https://github.com/ZiJianZhao/SeqGAN-PyTorch
class Discriminator(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Highway architecture based on the pooled feature maps is added. Dropout is adopted.
    """

    def __init__(self, num_classes, vocab_size, embedding_dim, filter_sizes, num_filters, dropout_prob):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, embedding_dim)) for f_size, num_f in zip(filter_sizes, num_filters)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p = dropout_prob)
        self.fc = nn.Linear(sum(num_filters), num_classes)

    def forward(self, x):
        """
        Inputs: x
            - x: (batch_size, seq_len)
        Outputs: out
            - out: (batch_size, num_classes)
        """
        emb = self.embed(x).unsqueeze(1) # batch_size, 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        out = torch.cat(pools, 1)  # batch_size * sum(num_filters)
        highway = self.highway(out)
        transform = torch.sigmoid(highway)
        out = transform * F.relu(highway) + (1. - transform) * out # sets C = 1 - T
        out = F.log_softmax(self.fc(self.dropout(out)), dim=1) # batch * num_classes
        return out


class DiscriminatorEncoder(nn.Module):
    """
    A CNN for text classification.
    Uses a convolutional, max-pooling and softmax layer.
    Highway architecture based on the pooled feature maps is added. Dropout is adopted.
    """

    def __init__(self, num_classes, embedding_dim, filter_sizes, num_filters, dropout_prob):
        super(DiscriminatorEncoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, embedding_dim)) for f_size, num_f in zip(filter_sizes, num_filters)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p = dropout_prob)
        self.fc = nn.Linear(sum(num_filters), num_classes)

    def forward(self, x):
        """
        Inputs: x
            - x: (batch_size, seq_len, projection)
        Outputs: out
            - out: (batch_size, num_classes)
        """
        emb =  x.unsqueeze(1) # batch_size, 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        out = torch.cat(pools, 1)  # batch_size * sum(num_filters)
        highway = self.highway(out)
        transform = torch.sigmoid(highway)
        out = transform * F.relu(highway) + (1. - transform) * out # sets C = 1 - T
        out = F.log_softmax(self.fc(self.dropout(out)), dim=1) # batch * num_classes
        return out

class Rewards(object):
    """ Rollout Policy """

    def __init__(self, model, update_rate, eos):
        self.ori_model = model
        self.own_model = copy.copy(model)
        self.update_rate = update_rate
        self.eos = eos
        self.own_model.eval()

    def get_rollout_reward(self, xs_pad, ilens, ys_pad, num, discriminator):
        """
        Implements a rollout policy gradient reward
        https://arxiv.org/pdf/1609.05473.pdf
        """
        rewards = []
        # batch_size = ys_pad.size(0)
        # add one because the output of the generator has an extra <eos> tag at the end.
        _, _, _, _, ys_hat, ys_true = self.own_model(xs_pad, ilens, ys_pad)
        # convert to probabilities
        ys_hat = F.softmax(ys_hat, dim = 2)
        # action = c.sample()
        seq_len = ys_true.size(1)
        zero = torch.zeros(ys_true.size(), dtype=torch.long)
        if ys_hat.is_cuda:
            zero = zero.cuda()
        for i in range(num):
            for l in range(1, seq_len):
                # create a distribution object to draw samples from
                c = Categorical(ys_hat[:, 0:l])
                # just take the first l tokens
                # samples = torch.cat((ys_hat[:, 0:l], zero[:,l:]), 1)
                samples = torch.cat((c.sample(), zero[:,l:]), 1)
                if ys_hat.is_cuda:
                    samples = samples.cuda()
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(ys_true)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def get_rollout_reward_encoder(self, hs_pad, hs_pad_noise, num, discriminator):
        """
        Implements a rollout policy gradient reward
        https://arxiv.org/pdf/1609.05473.pdf
        hs_pad and hs_pad_noise: batch_size, seq_len
        hs_pad_noise: batch_size, seq_len,projection
        """
        # greedy_sampling
        rewards = []
        seq_len = hs_pad.size(1)
        zero = torch.zeros(hs_pad.size(), dtype=torch.long)
        if hs_pad_noise.is_cuda:
            zero = zero.cuda()
        for i in range(num):
            for l in range(1, seq_len):
                # just take the first l tokens
                samples = torch.cat((hs_pad_noise[:, 0:l], zero[:,l:]), 1)
                if hs_pad.is_cuda:
                    samples = samples.cuda()
                pred = discriminator(samples)
                if hs_pad_noise.is_cuda:
                    samples = samples.cuda()
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(hs_pad)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards


    def get_cer_reward(self, ys_hat, ys_true, num):
        """
        Implements Self-Critical Sequence Training (SCST)
        eq 7 in https://arxiv.org/pdf/1712.07101.pdf
        reward = 1 - cer
        """
        rewards = []
        #rewards = torch.zeros(ys_pad.size(0))
        # batch_size = ys_pad.size(0)
        # add one because the output of the generator has an extra <eos> tag at the end.
        #_, _, _, _, ys_hat, ys_true = self.ori_model(xs_pad, ilens, ys_pad)
        # convert to probabilities
        ys_hat = F.softmax(ys_hat, dim = 2)
        # action = c.sample()
        # seq_len = ys_true.size(1)
        # zero = torch.zeros(ys_true.size(), dtype=torch.long)
        #if ys_hat.is_cuda:
        #    rewards = rewards.cuda()
        greedy_cer = torch.sum(convert_ys_hat_prob_to_seq(ys_hat, self.eos) != ys_true, 1).float()
        c = Categorical(ys_hat)
        for i in range(num):
            sample = c.sample()
            samples = clip_sequence(sample, self.eos)
            # if ys_hat.is_cuda:
            #     samples = samples.cuda()
            # get cer for the samples.
            sample_cer = torch.sum(samples != ys_true, 1).float()
            # reward = -((1-sample_cer) - (1-greedy_cer))* p(y_sample/x)
            rewards.append(-c.log_prob(sample).permute(1,0) * ( greedy_cer - sample_cer))

        loss = torch.stack(rewards).sum() /(1.0 * num * ys_true.size(1))  # batch_size * seq_len
        return Variable(loss, requires_grad = True)

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if 'embed' in name:
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]

class PGLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward())
    for adversial training of Generator
    """

    def __init__(self):
        super(PGLoss, self).__init__()

    def forward(self, pred, target, reward):
        """
        Inputs: pred, target, reward
            - pred: (batch_size, seq_len),
            - target : (batch_size, seq_len),
            - reward : (batch_size, ), reward of each whole sentence
        """
        one_hot = torch.zeros(pred.size(), dtype=torch.uint8)
        if pred.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view(-1, 1), 1)
        loss = torch.masked_select(pred, one_hot)
        loss = loss * reward.contiguous().view(-1)
        loss = -torch.sum(loss)
        return Variable(loss, requires_grad = True)
