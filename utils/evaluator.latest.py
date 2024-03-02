import torch
import torch.nn.functional as F
import numpy as np

class Metric(object):
    
    def is_continue_label(self, label, startlabel, distance):
        if distance == 0:
            return True
        if startlabel[0] == "S" or startlabel[0] == "s" or self.is_start_label(label) or label[2:] != startlabel[2:]:
            return False
        return True

    def is_start_label(self, label):
        return (label[0] == "B" or label[0] == "S" or label[0] == "b" or label[0] == "s")

    def eval(self, predict, target):
        preds, golds = [], []
        length = len(predict)
        idx = 0
        while idx < length:
            if self.is_start_label(predict[idx]):
                s = ""
                for idy in range(idx, length):
                    if not self.is_continue_label(predict[idy], predict[idx], idy - idx):
                        endpos = idy - 1
                        break
                    endpos = idy
                    s += predict[idy][0]
                ss = "[" + str(idx) + "," + str(endpos) + "]"
                preds.append(s + predict[idx][1:] + ss)
                idx = endpos
            else:
                print('######')
                golds.append(predict[idx])
                print('predict[idx]',predict[idx])
            idx += 1

        idx = 0
        while idx < length:
            if self.is_start_label(target[idx]):
                s = ""
                for idy in range(idx, length):
                    if not self.is_continue_label(target[idy], target[idx], idy - idx):
                        endpos = idy - 1
                        break
                    endpos = idy
                    s += target[idy][0]
                ss = "[" + str(idx) + "," + str(endpos) + "]"
                golds.append(s + target[idx][1:] + ss)
                idx = endpos
            else:
                golds.append(target[idx])
            idx += 1

        overall_count = len(golds)
        predict_count = len(preds)
        correct_count = 0
        for pred in preds:
            if pred in golds:
                correct_count += 1

        return overall_count, predict_count, correct_count

class Decoder(object):
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.illegal_bies = {'B_B', 'B_S', 'S_I', 'S_E', 'I_B', 'I_S', 'E_I', 'E_E'}
        self.continuous_bies = {'B_I', 'B_E', 'I_I', 'I_E'}
    
    # 判断分词联合词性的前后单侧标签是否合法
    def is_legal_label(self, label, pre_label):
        ws_label = pre_label[0] + '_' + label[0]
        if ws_label in self.illegal_bies:
            return False
        elif ws_label in self.continuous_bies:
            if pre_label[1:] != label[1:]:
                return False
            else:
                return True
        else:
            return True
    # 判断分词联合词性的前后耦合标签是否合法
    def is_legal_coupled_label(self, label, pre_label, source_tag, pre_source_tag, dataIndex):
        label1, label2 = label.split('@')
        pre_label1, pre_label2 = pre_label.split('@')
        #return (self.is_legal_label(label1, pre_label1) and self.is_legal_label(label2, pre_label2))
        if dataIndex == 1:
            return (self.is_legal_label(label1, pre_label1) and self.is_legal_label(label2, pre_label2) and label2 == source_tag and pre_label2 ==  pre_source_tag)
        else:
            return (self.is_legal_label(label1, pre_label1) and self.is_legal_label(label2, pre_label2) and label1 == source_tag and pre_label1 ==  pre_source_tag)
        #return (self.is_legal_label(label1, pre_label1) and self.is_legal_label(label2, pre_label2))
    
    # 在所有标签中找出所有当前耦合标签后的合法耦合标签集合
    '''def find_post_legal_labels(self, pre_label):
        legal_labels1 = set()
        legal_labels2 = set()
        legal_labels = set()
        pre_label1, pre_label2 = pre_label.split('@')
        for label1 in self.vocab._labels1:
            if self.is_legal_label(label1, pre_label1):
                legal_labels1.add(label1)
        for label2 in self.vocab._labels2:
            if self.is_legal_label(label2, pre_label2):
                legal_labels2.add(label2)
        for labels1 in legal_labels1:
            for labels2 in legal_labels2:
                legal_labels.add('@'.join((labels1, labels2)))
        legal_index = [self.vocab._label2id.get(label, -1) for label in legal_labels]
        return list(filter(lambda x : x!=-1, legal_index))'''
    
    def constrained_decoding(self, out, source_label, dataIndex):
        sen_len, labels_num = out.size()
        max_predicts = torch.max(out, 1)[1].tolist()
        sorted_index = torch.sort(out, 1, descending=True)[1].tolist()
        coupled_labels = self.vocab.id2label(max_predicts)
        # max_predicts1_str = [pred_str.split("@")[0] for pred_str in coupled_labels]
        # max_predicts2_str = [pred_str.split("@")[1] for pred_str in coupled_labels]
        
        constraint_predicts = []
        for index in sorted_index[0]:
            label = self.vocab.id2label(int(index))
            label1, label2 = label.split('@')
            if dataIndex == 1:
                if label1[0] in ['B','S'] and label2 == source_label[0]:
                    constraint_predicts.append(int(index))
                    coupled_labels[0] = self.vocab.id2label(int(index))
                    break
            else:
                if label2[0] in ['B','S'] and label1 == source_label[0]:
                    constraint_predicts.append(int(index))
                    coupled_labels[0] = self.vocab.id2label(int(index))
                    break

        
        #constraint_predicts = [int(max_predicts[0])]
        for i in range(0, len(coupled_labels) - 1):
            num = 0
            for index in sorted_index[i + 1]:
                post_coupled_label = self.vocab.id2label(int(index))
                post_source_tag = source_label[i+1]
                source_tag = source_label[i]
                if self.is_legal_coupled_label(post_coupled_label, coupled_labels[i], post_source_tag, source_tag, dataIndex):
                    constraint_predicts.append(int(index))
                    coupled_labels[i + 1] = post_coupled_label
                    break
                else:
                    if num == len(sorted_index[i+1])-1:
                        print('source_tag',source_tag)
                        print('post_source_tag',post_source_tag)
                        print('post_coupled_label',post_coupled_label)
                        print('coupled_label',coupled_labels[i])
                        #exit()
                num+=1
        return constraint_predicts

'''class Decoder(object):
    @staticmethod
    def viterbi(crf, emit_matrix):
        # viterbi for one sentence
        length = emit_matrix.size(0)
        max_score = torch.zeros_like(emit_matrix)
        paths = torch.zeros_like(emit_matrix, dtype=torch.long)

        max_score[0] = emit_matrix[0] + crf.strans
        for i in range(1, length):
            emit_scores = emit_matrix[i]
            scores = emit_scores + crf.transitions + max_score[i - 1].view(-1, 1).expand(-1, crf.labels_num)
            max_score[i], paths[i] = torch.max(scores, 0)

        max_score[-1] += crf.etrans
        prev = torch.argmax(max_score[-1])
        predict = [prev.item()]
        for i in range(length - 1, 0, -1):
            prev = paths[i][prev.item()]
            predict.insert(0, prev.item())
        return torch.tensor(predict)

    @staticmethod
    def viterbi_batch(crf, emits, masks):
        # viterbi for sentences in batch
        emits = emits.transpose(0, 1)
        masks = masks.t()
        sen_len, batch_size, labels_num = emits.shape

        lens = masks.sum(dim=0)  # [batch_size]
        scores = torch.zeros_like(emits)  # [sen_len, batch_size, labels_num]
        paths = torch.zeros_like(emits, dtype=torch.long) # [sen_len, batch_size, labels_num]

        scores[0] = crf.strans + emits[0]  # [batch_size, labels_num]
        for i in range(1, sen_len):
            trans_i = crf.transitions.unsqueeze(0)  # [1, labels_num, labels_num]
            emit_i = emits[i].unsqueeze(1)  # [batch_size, 1, labels_num]
            score = scores[i - 1].unsqueeze(2)  # [batch_size, labels_num, 1]
            score_i = trans_i + emit_i + score  # [batch_size, labels_num, labels_num]
            scores[i], paths[i] = torch.max(score_i, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(scores[length - 1, i] + crf.etrans)
            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            predicts.append(torch.tensor(predict).flip(0))

        return predicts'''

class Evaluator(object):
    def __init__(self, vocab, use_crf=True):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0
        self.vocab = vocab
        self.use_crf = use_crf

    def clear_num(self):
        self.pred_num = 0
        self.gold_num = 0 
        self.correct_num = 0

    def eval(self, network, data_loader, dataIndex, output_file):
    #def eval(self, network, data_loader, dataIndex):

        network.eval()
        total_loss = 0.0
        total_num = 0
        source_file = open('/data1/cgong/wspos/pku126/test_9','r')
        #source_file = open('/data1/cgong/wspos/trans/wspos.tran2.conll.name','r')
        source_tags = []
        chars = []
        s_tag = []
        char = []
        for line in source_file:
            if line != '\n':
                parts = line.split()
                char.append(parts[0])
                s_tag.append(parts[2])
            else:
                source_tags.append(s_tag)
                chars.append(char)
                s_tag = []
                char = []
        sentnum = 0 
        for batch in data_loader:
            batch_size = batch[0].size(0)
            total_num += batch_size
            # mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            # mask = word_idxs.gt(0)
            #mask, out, targets, Tags, bundled_labels = network.forward_batch(batch, dataIndex)
            mask, out, targets, Tags = network.forward_batch(batch, dataIndex)
            sen_lens = mask.sum(1)
            
           # print("mask:", mask)
           # print("sen_lens:", sen_lens)

            batch_loss, snum = network.get_loss(out, targets, mask)
            total_loss += batch_loss * batch_size

            # predicts = Decoder.viterbi_batch(network.crf, out, mask)
            # predicts = [torch.max(F.softmax(out_sen, dim = 1), 1)[1] for out_sen in out]
            
            # threshold = 0.9
            # predicts = []
            # num = 1
            # for i, out_sen in enumerate(out):
            #     # print(out_sen, flush=True)
            #     score, pre = torch.topk(F.softmax(out_sen[:sen_lens[i]], dim = 1), k=2, dim=-1, sorted=True)
            #     pre = pre.view(-1)
            #     # mask = score.gt(threshold)
            #     # pred = torch.masked_select(pre, mask)
            #     for p in self.vocab.id2label(pre.tolist()):
            #         f.write(str(p))
            #         if num % 2 == 0:
            #             f.write('\n')
            #         else:
            #             f.write(' ')
            #         num += 1
            predicts = []
            for i, out_sen in enumerate(out):
                # print(out_sen, flush=True)
                #print('i',i)
                #print('sentnum',sentnum)
                source_tag = source_tags[sentnum]
                char = chars[sentnum]
                #source_tag = source_tags[i]
                #print('i',i)
                pre = Decoder(self.vocab).constrained_decoding(F.softmax(out_sen[:sen_lens[i]], dim = 1), source_tag, dataIndex)
                pred = [p.split("@")[dataIndex-1] for p in self.vocab.id2label(pre)]
                predicts.append(pred)
                for pr,c in zip(pred,char):
                    output_file.write(c+' '+pr+'\n')
                output_file.write('\n')
                sentnum += 1
            # tags = torch.split(targets[mask], sen_lens.tolist())
            '''
            for predict_str, target in zip(predicts, Tags):  # zip()会自动取长度短的一边的list，所以不需要对Tags进行处理
                target_str = self.vocab.id2Tag(target.tolist(), dataIndex)           
                overall_count, predict_count, correct_count = Metric().eval(predict_str, target_str)
                # correct_num = sum(x == y for x,y in zip(predict, target))
                self.correct_num += correct_count
                self.pred_num += predict_count
                self.gold_num += overall_count
            '''
        exit()
        precision = self.correct_num/self.pred_num
        recall = self.correct_num/self.gold_num
        fmeasure = self.correct_num*2/(self.pred_num+self.gold_num)
        self.clear_num()
        return total_loss/total_num, precision, recall, fmeasure

