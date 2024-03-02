import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

cdef extern from "math.h":
    cpdef float logf(float x);
    cpdef float expf(float x);

ctypedef np.float32_t DTYPE_t
cdef int _ignore_index = -1
cdef DTYPE_t NEG_INF = -1e20

class SoftMaxlayer(object):

    def __init__(self, int ignore_index):
        _ignore_index = ignore_index

    @cython.boundscheck(False)
    def forward(self, np.ndarray[DTYPE_t, ndim=2] emit, np.ndarray[long, ndim=2] target):
        cdef DTYPE_t loss = 0.0
        cdef DTYPE_t tmp = 0.0
        cdef DTYPE_t maxScore = 0.0
        cdef DTYPE_t sum_all = 0.0
        cdef DTYPE_t sum_true = 0.0

     #   cdef np.ndarray[DTYPE_t, ndim=1] score
        cdef int N= emit[0].size

      #  score = np.array([NEG_INF] * N, dtype=np.float32)
        for sen_score, answer in zip(emit,target):
            if _ignore_index in answer:
                break
            maxScore = max(sen_score)
            for i in range(N):
                tmp = expf(sen_score[i] - maxScore)
                if i in answer:
                    sum_true += tmp
                #score[i] = tmp
                sum_all += tmp
            #for j in answer:
             #   sum_true += score[j]
            loss += logf(sum_all) - logf(sum_true)
        return loss

