import time
import random

import torch
import torch.nn as nn
from torch import optim

from dataPreprocess import DataPreprocess
from embeddingGoogle import GetEmbedding
from encoderRNN import EncoderRNN
from decoderRNN import DecoderRNN
from trainNetwork import TrainNetwork
from helper import Helper

def trainIters(model, input_lang, output_lang, pairs, n_iters=750,
               learning_rate=0.01, print_every=50, plot_every=1):
    
    helpFn = Helper()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                  lr=learning_rate)
    decoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                  lr=learning_rate)
    training_pairs = [helpFn.variables_from_pair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = model.train(input_variable, target_variable,
                           encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (helpFn.time_slice(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    helpFn.show_plot(plot_losses)
    del helpFn

def evaluate(train_network, input_lang, sentence):
    helpFn = Helper()
    input_variable = helpFn.variable_from_sentence(input_lang, sentence)
    output_words, attentions = train_network.evaluate(input_variable, sentence)
    del helpFn
    return output_words, attentions

def evaluateRandomly(train_network, input_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(train_network, input_lang, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()

    data_preprocess = DataPreprocess()
    input_lang, output_lang, pairs = data_preprocess.prepare_data('eng', 'hin', True)
    print(random.choice(pairs))

    # embedding_src = GetEmbedding(input_lang.word2index, input_lang.word2count, "../Embeddings/GoogleNews/")
    embedding_dest = GetEmbedding(output_lang.word2index, input_lang.word2count, "../Embeddings/GoogleNews/")

    hidden_size = 256
    # encoder = EncoderRNN(hidden_size, torch.from_numpy(embedding_src.embedding_matrix).type(torch.FloatTensor),
    #                      use_embedding=True, train_embedding=False)
    decoder = DecoderRNN(hidden_size, data_preprocess.max_length,
                         torch.from_numpy(embedding_dest.embedding_matrix).type(torch.FloatTensor),
                         use_embedding=True, train_embedding=False, dropout_p=0.1)

    encoder = EncoderRNN(hidden_size, (len(input_lang.word2index) + 1, 300))
    # decoder = DecoderRNN(hidden_size, data_preprocess.max_length, (len(output_lang.word2index) + 1, 300))

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    print("Training Network.")
    train_network = TrainNetwork(encoder, decoder, output_lang, data_preprocess.max_length)
    trainIters(train_network, input_lang, output_lang, pairs)

    evaluateRandomly(train_network, input_lang, pairs)
