from __future__ import print_function

import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import random
# from torch.nn.parameter import Parameter
# import torch.optim as optim
# from tqdm import tqdm
# import pdb

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib'
                % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


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

            num_classes = logits.size(1)
            correct = pred.eq(y.data.view_as(pred))
            all_pos = to_onehot(pred, num_classes).sum(dim=0)
            true_pos = to_onehot(pred[correct], num_classes).sum(dim=0)
            all_true = to_onehot(y, num_classes).sum(dim=0)

            precision = true_pos / all_pos
            recall = true_pos / all_true
            accu = (correct.sum().item()) / float(correct.size(0))
            return logits, loss, accu, precision, recall, pred
        else:
            return logits
