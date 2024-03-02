import collections
from itertools import chain

import torch
import torch.nn.init as init


class Vocab(object):
    def collect(self, corpus, min_freq=1):
        labels = sorted(set(chain(*corpus.label_seqs)))
        pos_tags = sorted(set(chain(*corpus.pos_seqs)))
        ws_tags = sorted(set(chain(*corpus.ws_seqs)))
        chars = list(chain(*corpus.char_seqs))
        bichars = list(chain(*corpus.bichar_seqs))
        
        # for bichar in bichars:
        #     if bichar not in corpus.bichar_dict:
        #         bichar='<UB>'
        
        chars_freq = collections.Counter(chars)
        chars = [c for c, f in chars_freq.items() if f > min_freq]
        bichars_freq = collections.Counter(bichars)
        bichars = [bc for bc, f in bichars_freq.items() if f> min_freq]
       
        return chars, bichars, labels, pos_tags, ws_tags

    def __init__(self, train_corpus, train2_corpus, labels_file, min_freq=1):
        chars, bichars, labels, pos_tags, ws_tags = self.collect(train_corpus, min_freq)
        chars2, bichars2, labels2, pos_tags2, ws_tags2 = self.collect(train2_corpus, min_freq)
        
        # with open('all_label', 'w', encoding='utf-8') as f:
        #     for label in labels:
        #         f.write(label)
        #         f.write(' ')
        #     f.write('\n')
        #     for label2 in labels2:
        #         f.write(label2)
        #         f.write(' ')
            

        # 读取裁剪后的联合标签集合
        pruning_lables = set()
        label1_index, label2_index = [], []
        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                label = line.strip()
                pruning_lables.add(label)
        pruning_lables = sorted(pruning_lables)
        for label in pruning_lables:
            label1, label2 = label.split("@")
            label1_index.append(labels.index(label1)) # 对应single tag 1中的index
            label2_index.append(labels2.index(label2)) # single tag 2中的index
            '''
            print('label1',label1)
            print('label2',label2)
            print('labels1',labels)
            print('labels2',labels2)
            exit()
            '''
        print(len(set(label1_index)), len(set(label2_index)))

        #  ensure the <PAD> index is 0
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self.label1_index = label1_index
        self.label2_index = label2_index

        self._bichars = [self.PAD] + bichars + bichars2 + [self.UNK]
        self._chars = [self.PAD] + chars + chars2 + [self.UNK]
        
        self._labels1 = labels
        self._labels2 = labels2

        self._posTags1 = pos_tags
        self._posTags2 = pos_tags2
        self._wsTags = ws_tags + ws_tags2

        # label = []
        # index = 0
        self._labels = list(pruning_lables)
        # for label1 in labels:
        #     for label2 in labels2:
        #         label.append(label1 + "@" + label2)
        # self._labels = label

        self._bichar2id = {bc: i for i, bc in enumerate(self._bichars)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}
        self._label12id = {l: i for i, l in enumerate(self._labels1)}
        self._label22id = {l: i for i, l in enumerate(self._labels2)}
        # self._labelws2id = {l: i for i, l in enumerate(label_ws)}
        # self._label2id2 = {l: i for i, l in enumerate(self._labels2)}

        self.num_bichars = len(self._bichars)
        self.num_chars = len(self._chars)
        self.num_labels1 = len(self._labels1)
        self.num_labels2 = len(self._labels2)
        self.num_labels = len(self._labels)

        self.UNK_bichar_index = self._bichar2id[self.UNK]
        self.UNK_char_index = self._char2id[self.UNK]
        self.PAD_bichar_index = self._bichar2id[self.PAD]
        self.PAD_char_index = self._char2id[self.PAD]

    def read_embedding(self, embedding_file, bichar_embedding_file, unk_in_pretrain=None):
        #  ensure the <PAD> index is 0
        with open(embedding_file, 'r') as f:
            lines = f.readlines()
        splits = [line.split() for line in lines]
        # read pretrained embedding file
        words, vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        if isinstance(unk_in_pretrain, str):
            assert unk_in_pretrain in words
            words = list(words)
            words[words.index(unk_in_pretrain)] = self.UNK

        pretrained = {w: torch.tensor(v) for w, v in zip(words, vectors)}
        out_train_chars = [w for w in words if w not in self._char2id]
        
        
        # read pretrained bichar embedding file
        with open(bichar_embedding_file, 'r') as f:
            lines = f.readlines()
        splits = [line.split() for line in lines]
        bichars, bichar_vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        if isinstance(unk_in_pretrain, str):
            assert unk_in_pretrain in bichars
            bichars = list(bichars)
            bichars[bichars.index(unk_in_pretrain)] = self.UNK
        
        bichar_pretrained = {w: torch.tensor(v) for w, v in zip(bichars, bichar_vectors)}
        out_train_bichars = [c for c in bichars if c not in self._bichar2id]
        

        # extend words and chars
        # ensure the <PAD> token at the first position
        self._chars =[self.PAD] + sorted(set(self._chars + out_train_chars) - {self.PAD})
        self._bichars =[self.PAD] + sorted(set(self._bichars + out_train_bichars) - {self.PAD})

        # update the words,chars dictionary
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self._bichar2id = {c: i for i, c in enumerate(self._bichars)}
        self.UNK_char_index = self._char2id[self.UNK]
        self.UNK_bichar_index = self._bichar2id[self.UNK]
        self.PAD_char_index = self._char2id[self.PAD]
        self.PAD_bichar_index = self._bichar2id[self.PAD]
        
        # update the numbers of words and chars
        self.num_chars = len(self._chars)
        self.num_bichars = len(self._bichars)

        # initial the extended embedding table
        embdim = len(vectors[0])
        extended_embed = torch.randn(self.num_chars, embdim)
        init.normal_(extended_embed, 0, 1 / embdim ** 0.5)
        
        bichar_extended_embed = torch.randn(self.num_bichars, embdim)
        init.normal_(bichar_extended_embed, 0, 1 / embdim ** 0.5)
        
        # the word in pretrained file use pretrained vector
        # the word not in pretrained file but in training data use random initialized vector
        for i, w in enumerate(self._chars):
            if w in pretrained:
                extended_embed[i] = pretrained[w]
        
        for i, w in enumerate(self._bichars):
            if w in bichar_pretrained:
                bichar_extended_embed[i] = bichar_pretrained[w]
        
        return extended_embed, bichar_extended_embed

    def bichar2id(self, bichar):
        assert (isinstance(bichar, str) or isinstance(bichar, list))
        if isinstance(bichar, str):
            return self._bichar2id.get(str, self.UNK_bichar_index)
        elif isinstance(bichar, list):
            return [self._bichar2id.get(w, self.UNK_bichar_index) for w in bichar]

    def label2id(self, dataIndex, label, labels2):
        assert (isinstance(label, str) or isinstance(label, list))
        if isinstance(label, str):
            return self._label2id.get(label, -1) # if label not in training data, index to 0 ?
        elif isinstance(label, list):
            if dataIndex == 1:    
                return [[self._label2id.get(l + "@" + l2, -1) for l2 in labels2] for l in label]
            elif dataIndex == 2:    
                return [[self._label2id.get(l2 + "@" + l, -1) for l2 in labels2] for l in label]

    def Tag2id(self, dataIndex, posTag):
        assert (isinstance(posTag, str) or isinstance(posTag, list))
        if isinstance(posTag, str):
            if dataIndex == 1:
                return self._label12id.get(posTag, 0) # if label not in training data, index to 0 ?
            else:
                return self._label22id.get(posTag, 0)
        elif isinstance(posTag, list):
            if dataIndex == 1:
                return [self._label12id.get(l, 0) for l in posTag]
            else:
                return [self._label22id.get(l, 0) for l in posTag]

    def char2id(self, char, max_len=200):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self.UNK_char_index)
        elif isinstance(char, list):
            return [self._char2id.get(c, self.UNK_char_index) for c in char]

    def id2label(self, id):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            assert (id <= self.num_labels)
            return self._labels[id]
        elif isinstance(id, list):
            return [self._labels[i] for i in id]

    def id2Tag(self, id, dataIndex):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            if dataIndex == 1:
                assert (id <= self.num_posTags)
                return self._labels1[id]
            else:
                assert (id <= self.num_posTags2)
                return self._labels2[id]
        elif isinstance(id, list):
            return [self._labels1[i] if dataIndex == 1 else self._labels2[i] for i in id]
