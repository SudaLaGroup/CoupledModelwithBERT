import argparse

import datetime
import torch
import torch.utils.data as Data

from config import config
from model import Char_LSTM_CRF
from utils import *
from module.bert_Vocab import *
from utils.data import TextDataset, batchify


def process_data(vocab, dataset, dataIndex, labels, bert_vocab, max_word_len=30):
    char_idxs, bichar_idxs, label_idxs, Tag_idxs = [], [], [], []
    subword_idxs, subword_masks, token_starts_masks= [], [], []
    for charseq, bicharseq, labelseq in zip(dataset.char_seqs, dataset.bichar_seqs, dataset.label_seqs):
        subword_ids, mask, token_starts = bert_vocab.subword_tokenize_to_ids(charseq)
        token_starts[[0, -1]] = 0
        subword_idxs.append(subword_ids)
        subword_masks.append(mask)
        token_starts_masks.append(token_starts)

        _char_idxs = vocab.char2id(charseq)
        _label_idxs = vocab.label2id(dataIndex, labelseq, labels)
        _target_idxs = vocab.Tag2id(dataIndex, labelseq)
        _bichar_idxs = vocab.bichar2id(bicharseq)

        char_idxs.append(torch.tensor(_char_idxs))
        bichar_idxs.append(torch.tensor(_bichar_idxs))
        label_idxs.append(torch.tensor(_label_idxs))
        Tag_idxs.append(torch.tensor(_target_idxs))

    return TextDataset((char_idxs, bichar_idxs, label_idxs, Tag_idxs,subword_idxs, subword_masks, token_starts_masks))


if __name__ == '__main__':
    # init config
    print("=========================start=========================", flush=True)
    t1 = datetime.datetime.now()
    model_name = 'char_lstm_crf'
    config = config[model_name]
    for name, value in vars(config).items():
        print('%s = %s' %(name, str(value)))
        
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu', type=int, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--pre_emb', action='store_true', help='choose if use pretrain embedding')
    parser.add_argument('--seed', type=int, default=17, help='random seed')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    args = parser.parse_args()
    print('setting:')
    print(args)
    print()

    # choose GPU and init seed
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print('using GPU device : %d' % args.gpu)
        print('GPU seed = %d' % torch.cuda.initial_seed())
        print('CPU seed = %d' % torch.initial_seed())
    else:
        use_cuda = False
        torch.set_num_threads(args.thread)
        torch.manual_seed(args.seed)
        print('CPU seed = %d' % torch.initial_seed())

    # read training , dev and test file
    print('loading three datasets...')
    train = Corpus(config.train_file, config.bichar_dict_filename)
    # train.reading_bichar_file()
    dev = Corpus(config.dev_file)
    test = Corpus(config.test_file)

    train2 = Corpus(config.train_file2, config.bichar_dict_filename)
    # train2.reading_bichar_file()
    dev2 = Corpus(config.dev_file2)
    test2 = Corpus(config.test_file2)
                       
    # collect all words, characters and labels in trainning data
    vocab = Vocab(train, train2, config.labels_file, min_freq=1)
    bert_vocab = bert_Vocab(config.bert_path)
    # choose if use pretrained word embedding
    if args.pre_emb and config.embedding_file !=None:
        print('loading pretrained embedding...')
        pre_embedding, bichar_pre_embedding = vocab.read_embedding(config.embedding_file, config.bichar_embedding_file, '<unk>')
    print('Chars : %d，BiCharacters : %d，labels : %d, labels1 : %d，labels2 : %d' %
          (vocab.num_chars, vocab.num_bichars, vocab.num_labels, vocab.num_labels1, vocab.num_labels2),flush=True)
    save_pkl(vocab, config.vocab_file)
    
    # process training data , change string to index
    print('processing datasets...', flush=True)
    train_data = process_data(vocab, train, 1, vocab._labels2,bert_vocab, max_word_len=20)
    # print('train_data: ', train_data[0])
    dev_data = process_data(vocab, dev, 1, vocab._labels2,bert_vocab, max_word_len=20)
    test_data = process_data(vocab, test, 1, vocab._labels2, bert_vocab,max_word_len=20)
    train_data2 = process_data(vocab, train2, 2, vocab._labels1,bert_vocab, max_word_len=20)
    dev_data2 = process_data(vocab, dev2, 2, vocab._labels1, bert_vocab,max_word_len=20)
    test_data2 = process_data(vocab, test2, 2, vocab._labels1,bert_vocab, max_word_len=20)

    train_loader = batchify(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )
    dev_loader = batchify(
        dataset=dev_data,
        batch_size=config.eval_batch,
        shuffle=False,
    )
    test_loader = batchify(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
    )

    train_loader2 = batchify(
        dataset=train_data2,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )
    dev_loader2 = batchify(
        dataset=dev_data2,
        batch_size=config.eval_batch,
        shuffle=False,
    )
    test_loader2 = batchify(
        dataset=test_data2,
        batch_size=config.eval_batch,
        shuffle=False,
    )

    # create neural network
    net = Char_LSTM_CRF(vocab.num_chars, 
                        config.char_dim, 
                        config.char_hidden, 
                        vocab.num_bichars,
                        config.word_dim, 
                        config.layers, 
                        config.word_hidden, 
                        vocab.num_labels1, 
                        vocab.num_labels2,
                        vocab.num_labels,
                        vocab.label1_index,
                        vocab.label2_index,
                        config.bert_path,
                        config.dropout
                        )
    if args.pre_emb:
        net.load_pretrained_embedding(pre_embedding, bichar_pre_embedding)
    # net = torch.load(config.net_file2)
    print(net)

    # if use GPU , move all needed tensors to CUDA
    if use_cuda:
        net.cuda() 
    
    # init evaluator
    evaluator = Evaluator(vocab)
    # init trainer
    trainer = Trainer(net, config)
    # start to train
    trainer.train((train_loader, dev_loader, test_loader), (train_loader2, dev_loader2, test_loader2), evaluator)
    t2 = datetime.datetime.now()
    print('total using time:',str(t2 - t1), flush=True)
