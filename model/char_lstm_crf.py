import torch
import datetime
from multiprocessing import Pool
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *

from module import *
from module.Bert_E import Bert_Embedding

class Char_LSTM_CRF(torch.nn.Module):
    def __init__(self, n_char, char_dim, char_hidden, n_bichar, bichar_dim,
                 n_layers, word_hidden, n_target, n_target2, n_labels, label1_index, label2_index, bert_path, drop=0.5):
        super(Char_LSTM_CRF, self).__init__()

        self.embedding_dim = char_dim
        self.drop1 = torch.nn.Dropout(drop)

        self.embedding_bert = Bert_Embedding(bert_path, 4, 768)
        self.embedding_char = torch.nn.Embedding(n_char, char_dim, padding_idx=0)
        self.embedding_bichar = torch.nn.Embedding(n_bichar, char_dim, padding_idx=0)

        # self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)

        '''
        print('n_target',n_target)
        print('n_target2',n_target2)
        print('n_labels',n_labels)
        exit()
        '''
        self.targetN1 = n_target
        self.targetN2 = n_target2
        self.label1_index = label1_index
        self.label2_index = label2_index
        # self.pruning_num = pruning_num
        # self.threshold = threshold

        if n_layers > 1:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim * 2 + 768,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=n_layers,
                dropout=0.2
            )
        else:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim + char_hidden,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
        # self.hidden = nn.Linear(word_hidden, word_hidden//2, bias=True)
        # self.normalization_layer = torch.nn.LayerNorm(elementwise_affine=False)
        self.out1 = torch.nn.Linear(word_hidden, n_target, bias=True) # n_target:num of single label1
        self.out2 = torch.nn.Linear(word_hidden, n_target2, bias=True) # n_target2: num of single label2
        self.out = torch.nn.Linear(word_hidden, n_labels, bias=True) # n_labels: num of coupled labels
        self.crf = CRFlayer(n_target * n_target2)
        # self.loss_func = torch.nn.CrossEntropyLoss(size_average=False, ignore_index=-1)
        self.loss_func = SoftMaxlayer(-1)
        self.reset_parameters()

    def load_pretrained_embedding(self, pre_embeddings, bichar_pre_embeddings):
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.embedding_char.weight = nn.Parameter(pre_embeddings)
        self.embedding_bichar.weight = nn.Parameter(bichar_pre_embeddings)

    def reset_parameters(self):
        init.xavier_uniform_(self.out1.weight)
        init.xavier_uniform_(self.out2.weight)
        init.xavier_uniform_(self.out.weight)
        # init.xavier_uniform_(self.hidden.weight)
        init.normal_(self.embedding_char.weight, 0, 1 / self.embedding_dim ** 0.5)
        init.normal_(self.embedding_bichar.weight, 0, 1 / self.embedding_dim ** 0.5)

    def forward(self, char_idxs, bichar_idxs, dataIndex, bert_outs):
        # mask = torch.arange(x.size()[1]) < lens.unsqueeze(-1)
        mask = char_idxs.gt(0)
        sen_lens = mask.sum(1)

        # char_vec = self.char_lstm.forward(char_idxs[mask])
        # char_vec = pad_sequence(torch.split(char_vec, sen_lens.tolist()), True, padding_value=0)

        char_vec = self.embedding_char(char_idxs)
        bichar_vec = self.embedding_bichar(bichar_idxs)
        feature = self.drop1(torch.cat((char_vec, bichar_vec, bert_outs), -1))

        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        feature = feature[sorted_idx]
        feature = pack_padded_sequence(feature, sorted_lens, batch_first=True)

        r_out, state = self.lstm_layer(feature, None)
        out, _ = pad_packed_sequence(r_out, batch_first=True, padding_value=0)
        out = out[reverse_idx]
        
        # normalization_layer = torch.nn.LayerNorm(out.size()[2], elementwise_affine=False)
        # out = normalization_layer(out)
        
        out1 = self.out1(out)
        out2 = self.out2(out)
        out_all = self.out(out)
        batch_size, length, tags = out1.size()
        '''
        print('out1.size:', out1.size())
        print('out2.size:', out2.size())
        print('label1_index',self.label1_index)
        print('len label1_index',len(self.label1_index))
        print()
        print('len label2_index',len(self.label2_index))
        exit()
        '''
        # label1_score, label1_index = torch.topk(out1, self.pruning_num, dim=-1, sorted=True)
        # label2_score, label2_index = torch.topk(out1, self.pruning_num, dim=-1, sorted=True)
        #out_all = out1[:, :, self.label1_index] + out2[:, :, self.label2_index]
        out_all += out1[:, :, self.label1_index] + out2[:, :, self.label2_index]
        
        # output1 = out1.view(-1, 1).repeat(1, self.targetN2).view(batch_size, length, -1)
        # output2 = out2.view(-1, 1).repeat(self.targetN1, 1).view(batch_size, length, -1)
        # output = out_all + output1 + output2
        '''
        output = out_all.view(batch_size, length, self.targetN1, self.targetN2) + out2.unsqueeze(2)
        output = output + out1.unsqueeze(3)
        output = output.view(batch_size, length, self.targetN1 * self.targetN2)
        '''
        return out_all

    def forward_batch(self, batch, dataIndex):
        char_idxs, bichar_idxs, label_idxs, target, subword_idxs, subword_masks, token_starts_masks = batch
        # import pdb
        # pdb.set_trace()
        bert_outs = self.embedding_bert(subword_idxs, subword_masks, token_starts_masks)
        mask = char_idxs.gt(0)
        out = self.forward(char_idxs, bichar_idxs, dataIndex, bert_outs)
        return mask, out, label_idxs, target
 
    def get_loss(self, emit, labels, mask):
        sen_lens = mask.sum(1)

        loss_calu = self.loss_func.calScore(emit, labels)
        loss = sum([sum(loss_calu[i][:sen_lens[i]]) for i in range(len(emit))])
                                                      
        #return torch.tensor(loss, requires_grad=True) / sen_lens.sum().float()
        return torch.tensor(loss, requires_grad=True), sen_lens.sum().float()

    def get_loss1(self, emit, labels, mask):
        logZ = self.crf.get_logZ(emit, mask)
        emit = emit.transpose(0, 1)
        labels = labels.transpose(0, 1)

        _,_l,labels_num = labels.shape
        sen_len, batch_size, _ll = emit.shape

        mask = mask.t()
        # logZ = self.crf.get_logZ(emit, mask)

        mask = mask.contiguous().view(-1, 1).repeat(1,labels_num).view(batch_size, sen_len, -1)
        mask = mask.transpose(0, 1)
        scores = self.crf.score(emit, labels, mask)

        return (logZ - scores) / emit.size()[1]


