class Config(object):
    # train_file = '/data/jyshen/code/LSTM-POS/data/ctb5/train.wspos'
    # dev_file = '/data/jyshen/code/LSTM-POS/data/ctb5/dev.wspos'
    # test_file = '/data/jyshen/code/LSTM-POS/data/ctb5/test.wspos'

    #train_file = '../../wspos/ctb5/ctb5-train-remove-trans.conll'
    # train_file = '../ctb9ws/ctb9.train.ws'
    # dev_file = '../ctb9ws/ctb9.dev.ws'
    # test_file = '../ctb9ws/ctb9.test.ws'

    train_file = '/data3/chdou/data-mws-ner/ontonotes5-new/wspos/train.pos'
    dev_file = '/data3/chdou/data-mws-ner/ontonotes5-new/wspos/dev.pos'
    test_file = '/data3/chdou/data-mws-ner/ontonotes5-new/wspos/test.pos'
    #test_file = '/data1/cgong/wspos/trans/wspos.tran2.conll.name'
    #test_file = '../../wspos/pku126/test_9'

    # train_file2 = '../../data/wspos/pku126/pku126-train.conll'
    # dev_file2 = '../../data/wspos/pku126/pku126-dev.conll'
    # test_file2 = '../../data/wspos/pku126/pku126-test.conll'
    
    bichar_dict_filename = '../embedding/dict_bigram.txt'
    # train_file2 = '../../data/wspos/pkubig/pku.train.wspos'
    # dev_file2 = '../../data/wspos/pkubig/pku.dev.wspos'
    # test_file2 = '../../data/wspos/pkubig/pku.test.wspos'
    
    train_file2 = '/data3/chdou/coupled-model/msr/msr-train.conll'
    dev_file2 = '/data3/chdou/coupled-model/msr/msr-dev.conll'
    test_file2 = '/data3/chdou/coupled-model/msr/msr-test.conll'

    labels_file = '/data3/chdou/coupled-model/pruning_new/ontonotes5msr.coupledlable'

    embedding_file = '../embedding/giga.chars.100.txt'
    bichar_embedding_file = '../embedding/giga.bichars.100.txt'
    output_file = '9'
    bert_path = 'bert-base-chinese'


class Char_LSTM_CRF_Config(Config):
    model = 'Char_LSTM_CRF'
    net_file1 = './save-bert-pretrain/ontonotes5-new&msr/char_lstm_crf_ontonotes5_2prun.pt'
    net_file2 = './save-bert-pretrain/ontonotes5-new&msr/char_lstm_crf_msr_2prun.pt'
    vocab_file = './save-bert-pretrain/ontonotes5-new&msr/vocab_ontonotes5_msr_2prun.pkl'

    word_hidden = 300
    char_hidden = 200
    layers = 2
    dropout = 0.55
    char_dim = 100
    word_dim = 100

    optimizer = 'adam'
    epoch = 1000
    gpu = 5
    lr = 0.001
    batch_size = 2000
    #batch_size2 = 2000
    eval_batch = 300
    tread_num = 4
    decay = 0.05
    patience = 50
    shuffle = True
    threshold = 0.95
    pruning_num = 2

config = {
    'char_lstm_crf' : Char_LSTM_CRF_Config,
}
