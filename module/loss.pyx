cimport cython
#include<iostream>
#include<cmath>
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "math.h":
    float exp(float theta)
    float log(float theta)

@cython.boundscheck(False)
class SoftMaxlayer(torch.nn.Module):

    def __init__(self, int ignore_index):
        super(SoftMaxlayer, self).__init__()
        self.ignore_index = ignore_index


    def forward(self, np.ndarray[float, ndim=2] score, np.ndarray[long, ndim=2] target):
        cdef float loss = 0.0
        cdef float maxScore = 0.0
        cdef float s = 0.0
        cdef float sum_all = 0.0
        cdef float sum_true = 0.0
        cdef float scs
        for scores, answer in zip(score, target):
            sum_all = 0.0
            sum_true = 0.0
            if self.ignore_index in answer:
                break
            maxScore = max(scores)
            for sc in scores:
                # s = exp(sc - maxScore)
                # scs.push_back(s)
                sum_all += exp(sc - maxScore)
            for ans in answer:
                sum_true += exp(scores[ans] - maxScore)

            loss += log(sum_all) - log(sum_true)

        return torch.tensor(loss, requires_grad=True)


