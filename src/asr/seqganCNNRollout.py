import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
import numpy as np

from asr_utils import clip_sequence


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
        transform = F.sigmoid(highway)
        out = transform * F.relu(highway) + (1. - transform) * out # sets C = 1 - T
        out = F.log_softmax(self.fc(self.dropout(out)), dim=1) # batch * num_classes
        return out


class Rollout(object):
    """ Rollout Policy """

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, xs_pad, ilens, ys_pad, num, discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discrimanator model
        """
        rewards = []
        # batch_size = ys_pad.size(0)
        # add one because the output of the generator has an extra <eos> tag at the end.
        seq_len = ys_pad.size(1) + 1
        for i in range(num):
            for l in range(1, seq_len):
                _, _, _, _, ys_hat, ys_true = self.own_model(xs_pad, ilens, ys_pad)
                ys_hat = clip_sequence(ys_hat, ys_true)

                pred = discriminator(ys_hat)
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
