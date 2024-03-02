import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class hdf5_reader(nn.Module):
    def __init__(self):
        self.filename = ""
        self.data = None
        self.f = None

    def read_from_file(self, filename):
        self.filename = filename
        print("Loading elmo hdf5 from {}".format(self.filename))
        self.f = h5py.File(filename, 'r')
        print("Load total {} file".format(len(self.f.keys())))

    def forward(self, x, max_length, sentences_lengths):
        output = []
        for idx in range(len(x)):
            sen = ' '.join(x[idx])  # get the correspoding sentence
            sentence_length = len(x[idx])
            sen_len = int(sentences_lengths[idx])
            assert sentence_length == sen_len
            embeddings = list([self.f[sen]])
            embeddings = torch.autograd.Variable(torch.from_numpy(np.array(embeddings)).type(torch.FloatTensor),
                                                 requires_grad=False)
            embeddings = torch.squeeze(embeddings, 0)
            pad = (0, 0, 0, max_length - embeddings.size()[1])
            embeddings = F.pad(embeddings, pad)
            output.append(embeddings)
        output = torch.stack(output, dim=0)
        return output

