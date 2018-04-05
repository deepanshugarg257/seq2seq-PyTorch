import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, use_embedding=False, train_embedding=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # Size of embedding vector

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1] # Size of embedding vector

        self.embedding.weight.requires_grad = train_embedding

        self.gru = nn.GRU(self.input_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
