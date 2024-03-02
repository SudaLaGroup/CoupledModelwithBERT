# -*- coding: utf-8 -*-
# coding: utf-8
from .pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from .judge import *

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

class bert_Vocab(object):
    def __init__(self, bert_vocab_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_path, do_lower_case=False)

    def convert_tokens_to_ids(self, tokens):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor(token_ids, dtype=torch.long)
        mask = torch.ones(len(ids), dtype=torch.long)
        return ids, mask

    def subword_tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        token_start_idxs = torch.cumsum(torch.tensor([0] + subword_lengths[:-1]), dim=0)
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        tokens = ["[PAD]" if judge_ignore(t) else t for t in tokens]
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(len(subword_ids), dtype=torch.uint8)
        token_starts[token_start_idxs] = 1
        return subword_ids, mask, token_starts
    
    def tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        return subwords
