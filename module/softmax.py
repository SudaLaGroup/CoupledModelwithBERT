import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *


class SoftMaxlayer(torch.nn.Module):
    def __init__(self, ignore_index):
        super(SoftMaxlayer, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, emit, target):
        loss = 0
        for emit_batch, target_batch in zip(emit,target):
            for sen_score, answer in zip(emit_batch, target_batch):
                if self.ignore_index in answer:
                    break
                maxScore, maxIndex = torch.max(sen_score,0)
                score = torch.exp((sen_score - maxScore).float())
                sum_all = score.sum()
                sum_true = score[answer].sum()
                loss += torch.log(sum_all) - torch.log(sum_true)
        return loss

    def calScore(self, emit, target):
        maxScore, _ = torch.max(emit, dim=2)
        batch, sen_len, labels = emit.shape
        maxScore = maxScore.view(sen_len,-1).repeat(1, labels).view(batch, sen_len, labels)
        # maxScore = maxScore.view(batch, sen_len, 1) #.repeat(1, labels).view(batch, sen_len, labels)
        scores = torch.exp((emit - maxScore).float())
        # scores = torch.exp(emit.float())
        sum_all = scores.sum(2)
        #cat_matrix = torch.zeros((batch, sen_len, 1))
        cat_matrix = torch.zeros((batch, sen_len, 1)).cuda()
        scores = torch.cat((cat_matrix, scores), -1)
        sum_true = torch.gather(scores, 2, target + 1).sum(2)
        return torch.log(sum_all) - torch.log(sum_true)
    
    def calScore2(self, emit, target):
        batch, sen_len, labels = emit.shape
        scores = torch.exp(emit.float())
        sum_all = scores.sum(2)
        cat_matrix = torch.ones((batch, sen_len, 1)).cuda()
        scores = torch.cat((cat_matrix, scores), -1)
        sum_true = torch.gather(scores, 2, target + 1).sum(2)
        return torch.log(sum_all) - torch.log(sum_true)

