import torch
import torch.autograd as autograd
import torch.nn as nn

from asr_utils import parse_hypothesis


# taken from https://github.com/suragnair/seqGAN
class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

    def forward(self, input, hidden):
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target

def get_espnet_output(batch, e2e, max_length):
    negatives = []
    for token, feat_str in zip(batch[0], batch[1]):
        feat = kaldi_io_py.read_mat(feat_str)
        false_hyp = e2e.recognize(feat, args, train_args.char_list, False)
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(false_hyp[0], char_list)
        # skip the last eos tag
        rec_tokenid = rec_tokenid.split()[:-1]
        rec_tokenid += [-1] * (max_length - len(rec_tokenid))
        negatives.append(rec_tokenid)
    return negatives

def convert_dict_to_list(jsonf, max_length):
    # shape [tokenid, feats]
    # both tokenid and feats are arrays
    tokenids, feats
    for name in jsonf.keys():
        tokenid = jsonf[name]["output"][0]["tokenid"].split()
        tokenid += [-1] * (max_length - len(tokenid))
        tokenids.append(tokenids)
        feats.append(jsonf[name]['input'][0]['feat'])
    return [tokenids, feats]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def train_discriminator(dis, dis_opt, e2e, train_json, batch_size, epochs, max_length, gpu):
    train_data = convert_dict_to_list(train_json, max_length)
    for batch in batch_iter(train_data, batch_size, epochs ):
        positives = batch[0]
        negatives = get_espnet_output(batch, e2e, max_length)

        dis_inp, dis_target = prepare_discriminator_data(positives, negatives, gpu)
        dis_opt.zero_grad()
        out = discriminator.batchClassify(dis_inp)
        loss_fn = nn.BCELoss()
        loss = loss_fn(out, dis_target)
        loss.backward()
        dis_opt.step()
