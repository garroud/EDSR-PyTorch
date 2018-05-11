import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'QMSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)

    #Q based MSE loss for the task
    @classmethod
    def QMSELoss(self, Q, sr, hr):
        # print Q[1,0,10:15,10:15]
        # print sr[1,0,10:15,10:15]
        # print (Q.shape)
        Q_ = torch.cat((Q,Q,Q),1).cuda()
        Q_ = Q_.view(-1, Q.shape[2],Q.shape[3])
        # print sr[0,0,10:20,10:20]
        sr = sr - hr
        sr = sr.view(sr.shape[0]*sr.shape[1],-1,1)
        sr_t = torch.transpose(sr,1,2)
        q_loss = torch.sum(torch.squeeze(torch.bmm(torch.bmm(sr_t,Q_), sr))) * 0.5
        # torch.save(Q.squeeze(), 'test_tensor.pt')

        #soft solution
        Q = torch.bmm(Q[:,0,:,:], Q[:,0,:, :]) + 1e-4 *torch.eye(Q.shape[2],requires_grad=True).cuda().repeat(Q.shape[0],1,1)
        # Q = torch.bmm(Q,Q)
        # print Q[10,10:15,10:15]
        for i in range(Q.shape[0]):
            # time.sleep(5)
            ##how to calculate a stable det
            # print Q.shape
            # np.linalg.cholesky(Q[i,:,:].cpu().detach().numpy())
            U= torch.potrf(Q[i,:,:])
            # e,_ = torch.symeig(Q[i,:,:])
            # print e
            # U, piv= torch.pstrf(Q[i,:,:].cpu())
            # print U[10:15,10:15]
            for j in range(Q.shape[0]):
                q_loss  -= torch.log(torch.abs(U[j,j])) * 0.5            # q_loss -= torch.sum(torch.log(torch.abs(e))) * 0.5
            # print torch.det(Q[i,:,:])
            # q_loss -= torch.det(Q[i,:,:]) * 0.25
        print q_loss / (hr.shape[0] * hr.shape[1] * hr.shape[2] * hr.shape[3])
        return q_loss / (hr.shape[0] * hr.shape[1] * hr.shape[2] * hr.shape[3])

    def forward(self, Q,sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] == 'QMSE':
                    loss = self.QMSELoss(Q,sr,hr)
                else:
                    loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
