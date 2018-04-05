import time
import math

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Helper(object):
    def __init__(self):
        self.EOS_token = 1
        self.use_cuda = torch.cuda.is_available()

    def as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_slice(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.as_minutes(s), self.as_minutes(rs))

    def show_plot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    def indexes_from_sentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def variable_from_sentence(self, lang, sentence):
        indexes = self.indexes_from_sentence(lang, sentence)
        indexes.append(self.EOS_token)
        result = Variable(torch.LongTensor(indexes)).view(-1, 1)
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def variables_from_pair(self, input_lang, output_lang, pair):
        input_variable = self.variable_from_sentence(input_lang, pair[0])
        target_variable = self.variable_from_sentence(output_lang, pair[1])
        return (input_variable, target_variable)
