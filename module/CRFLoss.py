import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *


class CRFlayer(torch.nn.Module):
    def __init__(self, labels_num):
        super(CRFlayer, self).__init__()
        self.labels_num = labels_num
        # (i,j)=score(tag[i]->tag[j])
        self.transitions = torch.nn.Parameter(torch.randn(labels_num, labels_num))
        # (i)=score(<BOS>->tag[i])
        self.strans = torch.nn.Parameter(torch.randn(labels_num))
        # (i)=score(tag[i]-><EOS>)
        self.etrans = torch.nn.Parameter(torch.randn(labels_num))
        self.reset_parameters()

    def reset_parameters(self):
        # self.transitions.data.zero_()
        # self.etrans.data.zero_()
        # self.strans.data.zero_()
        init.normal_(self.transitions.data, 0,1 / self.labels_num ** 0.5)
        init.normal_(self.strans.data, 0, 1 / self.labels_num ** 0.5)
        init.normal_(self.etrans.data, 0, 1 / self.labels_num ** 0.5)
        # bias = (6. / self.labels_num) ** 0.5
        # nn.init.uniform_(self.transitions, -bias, bias)
        # nn.init.uniform_(self.strans, -bias, bias)
        # nn.init.uniform_(self.etrans, -bias, bias)

    def get_logZ(self, emit, mask):
        '''
        emit: emission (unigram) scores of sentences in batch,[sen_len, batch_size, labels_num]
        mask: masks of sentences in batch,[sen_lens, batch_size]
        return: sum(logZ) in batch
        '''
        batch_size, sen_len, labels_num = emit.shape
        assert (labels_num==self.labels_num)

        alpha = emit[0][0]  # [labels_num]
        for j in range(batch_size):
            for i in range(1, sen_len):
                #print(j*sen_len+i, flush=True)
                # trans_i = self.transitions  # [labels_num, labels_num]
                emit_i = emit[j][i].unsqueeze(0)  # [1, labels_num]
                scores = self.transitions.cpu() + emit_i + alpha.unsqueeze(1)  # [labels_num, labels_num]
                scores = torch.logsumexp(scores, dim=1)  # [labels_num, 1]

                mask_i = mask[j][i].expand_as(alpha)  # [labels_num]
                alpha[mask_i] = scores[mask_i]

        return torch.logsumexp(alpha, dim=0).sum()

    def score(self, emit, target, mask):
        '''
        author: zhangyu
        return: sum(score)
        '''
        sen_len, batch_size, labels_num = emit.shape
        assert (labels_num==self.labels_num)
        # _,_l,labels1 = target.shape
        # print("labels:", labels1, flush=True)
        # maskR = mask.view(-1, 1).contiguous().repeat(1,labels1).view(sen_len, batch_size, -1)
        scores = torch.zeros_like(target, dtype=torch.float)  #[sen_len, batch_size, labels_num]

        # 加上句间迁移分数
        scores[1:] += self.transitions[target[:-1], target[1:]]
        # 加上发射分数
        scores += emit.gather(dim=2, index=target)
        # 通过掩码过滤分数
        #print(mask)
        #_,_l,labels1 = target.shape
        #print("labels:", labels1, flush=True)
        #maskR = mask.contiguous().view(-1, 1).repeat(0,labels1).view(sen_len, batch_size, -1)
        #print(mask, flush=True)
        score = scores.masked_select(mask).sum()

        # 获取序列最后的词性的索引
        ends = mask.sum(dim=0).view(1, -1) - 1
        # 加上句首迁移分数
        score += self.strans[target[0]].sum()
        # 加上句尾迁移分数
        score += self.etrans[target.contiguous().view(sen_len,-1).gather(dim=0, index=ends)].sum()
        return score

    def forward(self, emit, labels, mask):
        '''
        emit: emission (unigram) scores of sentences in batch, [batch_size, sen_len, labels_num]
        mask: masks of sentences in batch, [batch_size, sen_lens]
        labels: target of sentences, [batch_size, sen_lens]
        return: sum(logZ-score)/batch_size
        '''
        emit = emit.transpose(0, 1)
        labels = labels.t()
        mask = mask.t()
        logZ = self.get_logZ(emit, mask)
        scores = self.score(emit, labels, mask)
        # return logZ - scores
        return (logZ - scores) / emit.size()[1]
