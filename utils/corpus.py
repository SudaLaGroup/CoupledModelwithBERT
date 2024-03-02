class Corpus(object):
    def __init__(self, filename=None, filename2=None):
        self.filename = filename
        self.bichar_dict_filename = filename2
        self.bichar_dict = set()
        self.sentence_num = 0
        self.char_num = 0
        self.char_seqs = []
        self.bichar_seqs = []
        self.label_seqs = []
        self.pos_seqs = []
        self.ws_seqs = []
        chars = []
        bichars = []
        sequence = []
        pos_tags = []
        ws_tags = []
        sen_lens = 0
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    if sen_lens <= 500:
                        self.char_seqs.append(chars)
                        self.bichar_seqs.append(bichars)
                        self.label_seqs.append(sequence)
                        self.pos_seqs.append(pos_tags)
                        self.ws_seqs.append(ws_tags)
                        self.sentence_num += 1
                    chars = []
                    bichars = []
                    sequence = []
                    pos_tags = []
                    ws_tags = []
                    sen_lens = 0
                else:
                    sen_lens += 1
                    conll = line.split()         
                    chars.append(conll[0])
                    bichars.append(conll[1])
                    sequence.append(conll[2])
                    tags = conll[2].split("-")
                    pos_tags.append(tags[1])
                    ws_tags.append(tags[0])
                    self.char_num += 1
        print('%s : sentences:%d，words:%d' % (filename, self.sentence_num, self.char_num))
        
    def reading_bichar_file(self):
        with open(self.bichar_dict_filename, 'r', encoding='utf-8') as f2:
            lines = f2.readlines()
            for line in lines:
                self.bichar_dict.add(line.strip())
        print("%s : bichar dictionary length:%d" % (self.bichar_dict_filename, len(self.bichar_dict)))
