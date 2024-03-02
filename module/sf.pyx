import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *


class SoftMaxlayer(torch.nn.Module):
    def __init__(self, ignore_index):
        super(SoftMaxlayer, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, args):
        emit,target = args
        loss = 0
        for sen_score, answer in zip(emit,target):
            if self.ignore_index in answer:
                break
            maxScore, maxIndex = torch.max(sen_score,0)
            score = torch.exp((sen_score - maxScore).float())
            sum_all = score.sum()
            sum_true = score[answer].sum()
            loss += torch.log(sum_all) - torch.log(sum_true)
        return loss

