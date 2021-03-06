from __future__ import print_function

import os
import sys
import torch
import glog as log
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_util import weights_init
from torch.nn.parameter import Parameter
# import torch.optim as optim


sys.path.append('%s/pytorch_structure2vec-master/s2v_lib'
                % os.path.dirname(os.path.realpath(__file__)))


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            pred = logits.data.max(1)[1]
            correct = pred.eq(y.data.view_as(pred))
            accu = (correct.sum().item()) / float(correct.size(0))
            return loss, accu, pred
        else:
            log.warning('[MLP] No label info received.')
            return None


class MaxRecallAtPrecision(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, with_dropout=False):
        super(MaxRecallAtPrecision, self).__init__()

        self.device = torch.device('cuda')
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 2)
        self.with_dropout = with_dropout
        weights_init(self)

        self.alpha = alpha
        self.alpha_term = alpha / (1 - alpha)
        self.lam = Parameter(torch.tensor([2.0], device=self.device,
                                          requires_grad=True))
        self.result_dict = {}
        log.info('Maximize recall @ fixed precision=%.2f' % self.alpha)

    def print_result_dict(self):
        TP = self.result_dict['true_pos']
        FP = self.result_dict['false_pos']
        NYP = self.result_dict['num_Y_pos']
        TPL = self.result_dict['tp_lower']
        FPU = self.result_dict['fp_upper']
        precision, recall = TP / (TP + FP + 1e-10), TP / (NYP + 1e-10)

        if self.training:
            # log.info(self.h1_weights.weight)
            log.info('lambda = %.5f' % self.lam.item())

        log.info('TP = %.1f(>=%.1f), FP = %.1f(<=%.1f), |Y+| = %.1f' %
                 (TP, TPL, FP, FPU, NYP))
        log.info('precision = %.5f, recall = %.5f' % (precision, recall))
        log.info('inequality = %.5f(<=0)' % self.result_dict['inequality'])
        # recall_lb, precision_lb = TPL / (NP + 1e-5), TPL / (TPL + FPU + 1e-5)
        # log.info('R LB = %.5f, P LB = %.5f' % (recall_lb, precision_lb))

    def forward(self, X, target):
        """
        logits = f(X), target = Y in {0, 1}
        """
        h1 = self.h1_weights(X)
        h1 = F.sigmoid(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, p=0.1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.softmax(logits, dim=1)

        target = target.to(torch.float32)
        y = 2 * target - 1  # y must in {-1, 1}
        L = 0.0

        # pred belong to {-1, 1}
        # pred = (logits[:, 1] - self.bias) * 2 - 1
        pred = (logits[:, 1] - logits[:, 0]) * 2 - 1

        hinge_loss = torch.max(1 - y * pred, torch.tensor(0.0).to(y.device))
        Lp = (hinge_loss * target).sum()
        Ln = (hinge_loss * (1 - target)).sum()
        # L = (1 + lam) * Lp + lam * alpha_term * Ln - lam * target.sum()
        L = Lp + self.lam * (self.alpha_term * Ln + Lp - target.sum())

        # # pred_cls and pred_cls_float belong to {0.0, 1.0}
        pred_cls_float = (logits[:, 1] > logits[:, 0]).to(torch.float32)
        true_pos = (target * pred_cls_float).sum().item()
        false_pos = ((1 - target) * pred_cls_float).sum().item()
        num_Y_pos = target.sum().item()  # NOT positive of predicition
        tp_lower = (num_Y_pos - Lp).item()
        fp_upper = Ln.item()
        inequality = self.alpha_term * Ln + Lp - self.lam * num_Y_pos

        keys = ['true_pos', 'false_pos', 'num_Y_pos', 'tp_lower', 'fp_upper',
                'inequality', ]
        values = [true_pos, false_pos, num_Y_pos, tp_lower, fp_upper,
                  inequality, ]
        for key, value in zip(keys, values):
            self.result_dict[key] = value

        pred_cls = (logits[:, 1] > logits[:, 0]).to(torch.int32)
        correct = pred_cls.eq(target.to(torch.int32).data.view_as(pred_cls))
        accu = (correct.sum().item()) / float(correct.size(0))

        if self.lam.requires_grad is True and self.training is True:
            L *= -1

        return L, accu, pred_cls
