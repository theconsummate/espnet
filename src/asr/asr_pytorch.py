#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os

# chainer related
import chainer

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

# torch related
import torch

# espnet related
from asr_utils import adadelta_eps_decay
from asr_utils import add_results_to_json
from asr_utils import CompareValueTrigger
from asr_utils import get_model_conf
from asr_utils import load_inputs_and_targets
from asr_utils import make_batchset
from asr_utils import PlotAttentionReport
from asr_utils import restore_snapshot
from asr_utils import torch_load
from asr_utils import torch_resume
from asr_utils import torch_save
from asr_utils import torch_snapshot
from e2e_asr_th import E2E
from e2e_asr_th import Loss
from e2e_asr_th import pad_list

from e2e_asr_th import Reporter

from asr_utils import torch_load_without_dis

# for kaldi io
import kaldi_io_py

# rnnlm
import extlm_pytorch
import lm_pytorch

# matplotlib related
import matplotlib
import numpy as np
matplotlib.use('Agg')

# seqgan related
from seqganCNNRollout import Discriminator, Rewards, PGLoss
from asr_utils import clip_sequence

REPORT_INTERVAL = 100

VOCAB_SIZE = 52

class CustomEvaluator(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    x = self.converter(batch, self.device)
                    self.model(*x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomDiscriminatorEvaluator(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, e2e, dis, iterator, target, converter, device, eos):
        super(CustomDiscriminatorEvaluator, self).__init__(iterator, target)
        self.dis = dis
        self.converter = converter
        self.device = device
        self.eos = eos
        self.target = target
        self.e2e = e2e

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.dis.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    xs_pad, ilens, ys_pad = self.converter(batch, self.device)
                    if ys_pad.size()[1] < 20:
                        # skip iteration as sequence is too small for conv net
                        continue

                    # compute the negatives form e2e
                    _, _, _, _, ys_hat, ys_true = self.e2e(xs_pad, ilens, ys_pad)
                    ys_hat = clip_sequence(ys_hat, ys_true)

                    inp = torch.cat((ys_true, ys_hat), 0)
                    # .type(torch.LongTensor)
                    target = torch.ones(ys_true.size()[0] + ys_hat.size()[0]).type(torch.LongTensor)
                    target[ys_true.size()[0]:] = 0
                    # target[:ys_true.size()[0]] = 0.9

                    inp = inp.to(self.device)
                    target = target.to(self.device)

                    out = self.dis(inp)
                    loss_fn = torch.nn.NLLLoss()
                    loss = loss_fn(out, target)
                    pred = out.data.max(1)[1]
                    acc = pred.eq(target.data).cpu().sum().item()/float(target.size()[0])
                    self.target.report_dis(float(loss), acc)
                    print("discriminator loss: " + str(float(loss)) + ", accuracy: " + str(acc))
                summary.add(observation)
        self.dis.train()

        return summary.compute_mean()

class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu, rewards, dis, pg_loss, reporter):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.rewards = rewards
        self.dis = dis
        self.pg_loss = pg_loss
        self.reporter = reporter

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        xs_pad, ilens, ys_pad = self.converter(batch, self.device)

        # Compute the loss at this time step and accumulate it
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.rewards:
            if ys_pad.size()[1] < 20:
            # skip iteration as sequence is too small for conv net in discriminator
                return
            # this is during adversarial training
            rewards = torch.tensor(self.rewards.get_rollout_reward(xs_pad, ilens, ys_pad, 16, self.dis))
            rewards = rewards.to(self.device)
            loss_ctc, loss_att, acc, _, ys_hat, ys_true = self.model.predictor(xs_pad, ilens, ys_pad)
            # ys_hat = clip_sequence(ys_hat, ys_true)
            # convert ys_hat from batch_size x seq_len x vocab_size to batch_size*seq_len x vocab_size
            # convert ys_true from batch_size x seq_len to batch_size*seq_len
            loss = self.pg_loss(ys_hat.contiguous().view(-1, VOCAB_SIZE), ys_true.data.contiguous().view((-1,)), rewards)
            self.reporter.report(float(loss_ctc), float(loss_att), acc, float(loss))
            loss.backward()
        else:
            if self.ngpu > 1:
                loss = 1. / self.ngpu * self.model(xs_pad, ilens, ys_pad)
                loss.backward(loss.new_ones(self.ngpu))  # Backprop
            else:
                loss = self.model(xs_pad, ilens, ys_pad)
                loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()


class CustomDiscriminatorUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, e2e, discriminator, grad_clip_threshold, train_iter,
                 optimizer, converter, dis_reporter, device, ngpu):
        super(CustomDiscriminatorUpdater, self).__init__(train_iter, optimizer)
        self.e2e = e2e
        self.model = discriminator
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.dis_reporter = dis_reporter

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        # logging.warning("discriminator training loop.")
        batch = train_iter.next()
        xs_pad, ilens, ys_pad = self.converter(batch, self.device)
        if ys_pad.size()[1] < 20:
        # skip iteration as sequence is too small for conv net
            return

        # Compute the loss at this time step and accumulate it
        # optimizer.zero_grad()  # Clear the parameter gradients
        # compute the negatives form e2e
        _, _, _, _, ys_hat, ys_true = self.e2e(xs_pad, ilens, ys_pad)
        ys_hat = clip_sequence(ys_hat, ys_true)

        inp = torch.cat((ys_true, ys_hat), 0)
        # .type(torch.LongTensor)
        target = torch.ones(ys_true.size()[0] + ys_hat.size()[0]).type(torch.LongTensor)
        target[ys_true.size()[0]:] = 0
        # target[:ys_true.size()[0]] = 0.9

        inp = inp.to(self.device)
        target = target.to(self.device)

        optimizer.zero_grad()
        out = self.model(inp)
        loss_fn = torch.nn.NLLLoss()
        loss = loss_fn(out, target)
        pred = out.data.max(1)[1]
        acc_dis = pred.eq(target.data).cpu().sum().item()/float(target.size()[0])
        # report the values
        self.dis_reporter.report_dis(float(loss), acc_dis)

        # backprop
        loss.backward()

        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()

class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsamping_factor=1):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1

    def transform(self, item):
        return load_inputs_and_targets(item)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # specify model architecture
    # dis = Discriminator(64, 64, 52)
    # discriminator parameters
    d_num_class = 2
    d_embed_dim = 64
    d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    d_dropout_prob = 0.2
    #
    dis = Discriminator(d_num_class, VOCAB_SIZE, d_embed_dim, d_filter_sizes, d_num_filters, d_dropout_prob)
    e2e = E2E(idim, odim, args)
    model = Loss(e2e, args.mtlalpha)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(e2e.subsample[0])

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.MultiprocessIterator(
        TransformDataset(train, converter.transform),
        batch_size=1, n_processes=1, n_prefetch=8)
    valid_iter = chainer.iterators.SerialIterator(
        TransformDataset(valid, converter.transform),
        batch_size=1, repeat=False, shuffle=False)

    def create_main_trainer(epochs, tag, rewards, pg_loss):
        # Set up a trainer
        updater = CustomUpdater(
            model, args.grad_clip, copy.copy(train_iter), optimizer, converter, device, args.ngpu, rewards, dis, pg_loss, reporter)
        trainer = training.Trainer(
            # updater, (args.epochs, 'epoch'), out=args.outdir)
            updater, (epochs, 'epoch'), out=args.outdir)

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(CustomEvaluator(model, copy.copy(valid_iter), reporter, converter, device))

        # Save attention weight each epoch
        if args.num_save_attention > 0 and args.mtlalpha != 1.0:
            data = sorted(list(valid_json.items())[:args.num_save_attention],
                        key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
            if hasattr(model, "module"):
                att_vis_fn = model.module.predictor.calculate_all_attentions
            else:
                att_vis_fn = model.predictor.calculate_all_attentions
            trainer.extend(PlotAttentionReport(
                att_vis_fn, data, args.outdir + "/att_ws",
                converter=converter, device=device), trigger=(1, 'epoch'))

        # Make a plot for training and validation values
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                            'main/loss_ctc', 'validation/main/loss_ctc',
                                            'main/loss_att', 'validation/main/loss_att'],
                                            'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                            'epoch', file_name='acc.png'))

        # Save best models
        trainer.extend(extensions.snapshot_object(model, 'model.loss.best.' + tag, savefun=torch_save),
                    trigger=training.triggers.MinValueTrigger('validation/main/loss'))
        if mtl_mode is not 'ctc':
            trainer.extend(extensions.snapshot_object(model, 'model.acc.best.' + tag, savefun=torch_save),
                        trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

        # save snapshot which contains model and optimizer states
        trainer.extend(torch_snapshot(filename=tag +'.snapshot.ep.{.updater.epoch}'), trigger=(1, 'epoch'))

        # epsilon decay in the optimizer
        if args.opt == 'adadelta':
            if args.criterion == 'acc' and mtl_mode is not 'ctc':
                trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best.' + tag, load_fn=torch_load),
                            trigger=CompareValueTrigger(
                                'validation/main/acc',
                                lambda best_value, current_value: best_value > current_value))
                trainer.extend(adadelta_eps_decay(args.eps_decay),
                            trigger=CompareValueTrigger(
                                'validation/main/acc',
                                lambda best_value, current_value: best_value > current_value))
            elif args.criterion == 'loss':
                trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best.' + tag, load_fn=torch_load),
                            trigger=CompareValueTrigger(
                                'validation/main/loss',
                                lambda best_value, current_value: best_value < current_value))
                trainer.extend(adadelta_eps_decay(args.eps_decay),
                            trigger=CompareValueTrigger(
                                'validation/main/loss',
                                lambda best_value, current_value: best_value < current_value))

        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
        report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                    'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                    'main/acc', 'validation/main/acc', 'elapsed_time']
        if args.opt == 'adadelta':
            trainer.extend(extensions.observe_value(
                'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
                trigger=(REPORT_INTERVAL, 'iteration'))
            report_keys.append('eps')
        trainer.extend(extensions.PrintReport(
            report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

        trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

        return trainer

    # Setup discriminator
    # Discriminator settings
    # discriminator is defined at the top
    dis = dis.to(device)
    # train discriminator
    # dis_optimizer = torch.optim.Adagrad(dis.parameters())
    dis_optimizer = torch.optim.SGD(dis.parameters(), lr = 0.01, momentum=0.9)
    dis_reporter = Reporter()

    # FIXME: TOO DIRTY HACK
    setattr(dis_optimizer, "target", dis_reporter)
    setattr(dis_optimizer, "serialize", lambda s: dis_reporter.serialize(s))

    def create_dis_trainer(epochs):
        dis_updater = CustomDiscriminatorUpdater(e2e, dis, args.grad_clip, copy.copy(train_iter), dis_optimizer, converter, dis_reporter, device, args.ngpu)
        dis_trainer = training.Trainer(
            dis_updater, (epochs, 'epoch'), out=args.outdir)
        # Evaluate the model with the test dataset for each epoch

        dis_trainer.extend(CustomDiscriminatorEvaluator(e2e, dis, copy.copy(valid_iter), dis_reporter, converter, device, odim -1))
        dis_trainer.extend(torch_snapshot(filename='dis.snapshot.ep.{.updater.epoch}'), trigger=(1, 'epoch'))
        # Save best models
        #dis_trainer.extend(extensions.snapshot_object(model, 'dis.loss.best', savefun=torch_save),
        #            trigger=training.triggers.MinValueTrigger('validation/main/loss_dis'))
        #dis_trainer.extend(extensions.snapshot_object(model, 'dis.acc.best', savefun=torch_save),
        #                trigger=training.triggers.MaxValueTrigger('validation/main/acc_dis'))

        # Write a log of evaluation statistics for each epoch
        dis_trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
        report_keys = ['epoch', 'iteration', 'main/loss_dis', 'validation/main/loss_dis', 'validation/main/acc_dis', 'elapsed_time']
        dis_trainer.extend(extensions.PrintReport(
            report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

        dis_trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))
        return dis_trainer


    trainer = create_main_trainer(args.epochs, "base", None, None)
    dis_pre_train_epochs = 10
    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        print('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)
    # Run the training
    trainer.run()

    # train discriminator
    print("training discriminator")
    print("setting e2e to eval mode")
    e2e.eval()
    dis.train()
    dis_trainer = create_dis_trainer(dis_pre_train_epochs)
    dis_snapshot_path = "/mount/arbeitsdaten/asr-2/mishradv/espnet/egs/wsj/asr1/exp/train_si284_pytorch_cnnseqgan/results/dis.snapshot.ep.10"
    # dis_snapshot_path = "/mount/arbeitsdaten/asr-2/mishradv/espnet/egs/tedlium/asr1/exp/train_trim_pytorch_seqgan_esppretrain15_dispretrain22_advratio5/results/dis.snapshot.ep.22"
    torch_resume(dis_snapshot_path, dis_trainer)
    dis_trainer.run()


    # run adversarial training with policy gradient
    ADV_TRAIN_EPOCHS = 5
    # e2e.use_pgloss = True
    # e2e.train()
    print("starting adversarial training")
    rewards = Rewards(e2e, 0.8)
    pg_loss = PGLoss()
    for epoch in range(ADV_TRAIN_EPOCHS):
        # TRAIN GENERATOR
        # train generator with policy gradient
        print("training generator with pg loss")
        trainer = create_main_trainer(1, "pgloss" + str(epoch), rewards, pg_loss)
        dis_trainer = create_dis_trainer(8)

        e2e.train()
        dis.eval()
        trainer.run()
        if epoch == (ADV_TRAIN_EPOCHS - 1):
            # no need to train the discriminator at the last loop, break
            break

        # TRAIN DISCRIMINATOR
        print('Adversarial Training Discriminator')
        e2e.eval()
        dis.train()
        dis_trainer.run()

        print("Updating rewards model")
        # update roll-out model
        rewards.update_params()



def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    torch_load_without_dis(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
            nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
