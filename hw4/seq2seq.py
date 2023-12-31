#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open
import numpy as np

import matplotlib.pyplot as plt

# if you are running on the gradx/ugradx/ another cluster,
# you will need the following line
# if you run on a local machine, you can comment it out
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """This class handles the mapping between the words and their indicies"""

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding="utf-8").read().strip().split("\n")
    # Split every line into pairs
    pairs = [l.split("|||") for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """Creates the vocabs for each of the langues based on the training corpus."""
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info("%s (src) vocab size: %s", src_vocab.lang_code, src_vocab.n_words)
    logging.info("%s (tgt) vocab size: %s", tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################


def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence"""
    indexes = []
    indexes.append(SOS_index)
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair"""
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class LSTM(nn.Module):
    """
    Implements a standard LSTM block with input, output, and forget gates (no peepholes)
    """

    def __init__(self, input_size, hidden_size, reverse=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse = reverse  # to handle bidirectional RNNs

        self.input_weights = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_weights = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_weights = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_weights = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden, cell_state):
        combined = torch.cat((input, hidden), 1)
        # Calculate gates
        input_gate = torch.sigmoid(self.input_weights(combined))
        output_gate = torch.sigmoid(self.output_weights(combined))
        forget_gate = torch.sigmoid(self.forget_weights(combined))
        cell_gate = torch.tanh(self.cell_weights(combined))

        # Comptue next hidden, cell states. Ordering here is important
        cell_state = forget_gate * cell_state + input_gate * cell_gate
        hidden = output_gate * torch.tanh(cell_state)

        return hidden, cell_state

class EncoderRNN(nn.Module):
    """the class for the enoder RNN"""

    def __init__(self, input_size, hidden_size, dropout_p=0.1, ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.embedding = nn.Linear(input_size, hidden_size)
        self.right_lstm = LSTM(hidden_size, hidden_size)  # rnn -> rnn
        self.left_lstm = LSTM(hidden_size, hidden_size, reverse=True)  # rnn <- rnn
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        The point of the forward pass in seq2seq is to just compute the annotations

        This needs to processs SOS and EOS.
        E.g. <SOS> foo bar <EOS>.
        """

        # collect the -> hidden states
        forward_hidden_states = []
        forward_cell_states = []
        left_hidden = self.get_initial_hidden_state()
        left_cell_state = self.get_initial_cell_state()
        for elem in input:
            one_hot = torch.nn.functional.one_hot(
                elem, num_classes=self.input_size
            ).float()
            embedding = self.embedding(one_hot)
            embedding = self.dropout(embedding)
            left_hidden, left_cell_state = self.right_lstm(
                embedding, left_hidden, left_cell_state
            )
            forward_hidden_states.append(left_hidden)
            forward_cell_states.append(left_cell_state)

        # collect the <- hidden states
        backward_hidden_states = []
        backward_cell_states = []
        hidden = self.get_initial_hidden_state()
        cell_state = self.get_initial_cell_state()
        for elem in reversed(input):
            one_hot = torch.nn.functional.one_hot(
                elem, num_classes=self.input_size
            ).float()
            embedding = self.embedding(one_hot)
            embedding = self.dropout(embedding)
            hidden, cell_state = self.left_lstm(embedding, hidden, cell_state)
            backward_hidden_states.append(hidden)
            backward_cell_states.append(cell_state)

        # combine the hidden states into annotations
        hidden_states = []
        cell_states = []
        for i in range(len(input)):
            hidden_states.append(
                torch.cat((forward_hidden_states[i], backward_hidden_states[i]), 1)
            )
            cell_states.append(
                torch.cat((forward_cell_states[i], backward_cell_states[i]), 1)
            )

        return hidden_states, cell_states

    def get_initial_hidden_state(self):
        return torch.zeros(1, self.hidden_size, device=device)

    def get_initial_cell_state(self):
        return torch.zeros(1, self.hidden_size, device=device)

class Attention(nn.Module):
    """the class for the single layer attention network"""

    def __init__(self, hidden_size) -> None:
        super(Attention, self).__init__()
        # attention weights
        self.attn_layer = nn.Linear(4 * hidden_size, hidden_size)
        self.attn_output = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_state):
        input = torch.cat((hidden, encoder_state), 1)
        output = self.attn_layer(input)
        output = torch.tanh(output)
        output = self.attn_output(output)
        return output


class AttnDecoderRNN(nn.Module):
    """the class for the decoder"""

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.exp = torch.exp

        self.dropout = nn.Dropout(self.dropout_p)

        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.tanh
        self.embedding = nn.Linear(output_size, 2 * hidden_size)
        self.output_layer = nn.Linear(6 * hidden_size, output_size)
        self.lstm = LSTM(4 * hidden_size, 2 * hidden_size)
        self.attention = Attention(hidden_size)


    def forward(self, input, hidden, encoder_outputs, cell_state):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.

        * input is prev predicted word y_{i - 1}
        * hidden is s_{i-1} as it appears in the paper
        * encoder_outputs is h_1, h_2, \dots, h_n, the concatenated annotations.
        """
        # Compute attention weights
        one_hot = torch.nn.functional.one_hot(
            input.long(), num_classes=self.output_size
        ).float()
        embedding = self.embedding(one_hot)
        embedding = self.dropout(embedding)
        attn_weights = []
        denom = 0
        for encoder_state in encoder_outputs:
            denom += self.exp(self.attention.forward(hidden, encoder_state))

        for encoder_state in encoder_outputs:
            e_ij = self.attention.forward(hidden, encoder_state)
            # compatibility of the i-th translated word to the j-th source word
            a_ij = self.exp(e_ij) / denom
            attn_weights.append(a_ij) 

        # Compute context vector
        context = torch.zeros(1, 2 * self.hidden_size)
        for j in range(len(encoder_outputs)):
            a_ij = attn_weights[j]
            h_j = encoder_outputs[j]
            context += a_ij * h_j

        # Compute next_hidden state s_{i}
        next_hidden, cell_state = self.lstm(
            torch.cat((embedding, context), 1), hidden, cell_state
        )

        # Compute prediction based on y_{i-1}, s_{i-1}, c_i
        output = torch.cat((embedding, hidden, context), 1)
        output = self.output_layer(output)
        output = self.softmax(output)

        return output, next_hidden, attn_weights, cell_state

    def get_initial_hidden_state(self):
        return torch.zeros(1, 2 * self.hidden_size, device=device)

    def get_initial_cell_state(self):
        return torch.zeros(1, 2 * self.hidden_size, device=device)


######################################################################
def train_mini_batch(
    mini_batch,
    encoder,
    decoder,
    criterion,
    src_vocab,
    tgt_vocab,
    max_length=MAX_LENGTH,
):
    loss = 0
    for sentence in mini_batch:
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, sentence)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss += train(input_tensor, target_tensor, encoder, decoder, criterion)

    return loss


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    criterion,
    max_length=MAX_LENGTH,
):
    # make sure the encoder and decoder are in training mode so dropout is applied
    annotations, cell_states = encoder(input_tensor)

    # Initialize decoder values
    all_attn_weights = []
    decoder_input = torch.tensor([SOS_index], device=device)
    preds = []
    # Collect model predictions
    hidden = annotations[-1]
    cell_state = cell_states[-1]
    for _ in range(len(target_tensor) - 1):
        log_softmax, hidden, attn_weights, cell_state = decoder(
            decoder_input, hidden, annotations, cell_state
        )
        preds.append(log_softmax)
        all_attn_weights.append(attn_weights)
        decoder_input = torch.tensor([log_softmax.argmax().item()], device=device)


    # Update the weights in the encoder, decoder based on preds
    loss = 0
    for i in range(len(target_tensor) - 1):
        prediction = preds[i]
        target = target_tensor[i + 1] # shift one over due to SOS symbol
        loss += criterion(prediction, target)

    return loss


######################################################################


def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()
 
    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        encoder_outputs, cell_states = encoder(input_tensor)

        decoder_input = torch.tensor([SOS_index], device=device)
        decoded_words = []
        decoder_attentions = []

        decoder_hidden = encoder_outputs[-1]
        decoder_cell_state = cell_states[-1]
        for ind in range(max_length):
            (
                decoder_output,
                decoder_hidden,
                decoder_attention,
                decoder_cell_state,
            ) = decoder(
                decoder_input, decoder_hidden, encoder_outputs, decoder_cell_state
            )
            decoder_attentions.append(decoder_attention)
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(tgt_vocab.index2word[topi.item()])
            if topi.item() == EOS_index:
                break

            decoder_input = torch.tensor([topi.item()], device=device)

        return decoded_words, decoder_attentions


######################################################################

def draw_annotations(french, english, a, ind):
    b = []
    for i in a:
        elem = []
        for j in i:
            elem.append(j.item())
        b.append(elem)
    # Sample data
    data = b  # Replace this with your actual data
    print("These should be equal", len(english.split()), len(data))
    print("These should also be equal", len(french.split()), len(data[0]))

    # Custom labels for x and y axes
    y_labels = english.split() 
    x_labels =  ["sos"] + french.split() + ["eos"]

    # Create a figure and axis objects
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)

    # Create a heatmap
    im = ax.imshow(data, cmap='viridis', interpolation='nearest')

    # Set custom labels on the x and y axes
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.xaxis.tick_top()

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # Display colorbar
    ax.figure.colorbar(im, ax=ax)

    # Save the plot as a PNG file
    fig.savefig(f'annotations/{ind}.png')

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(
    encoder,
    decoder,
    pairs,
    src_vocab,
    tgt_vocab,
    max_num_sentences=None,
    max_length=MAX_LENGTH,
):
    output_sentences = []
    output_attentions = []
    for ind, pair in enumerate(pairs[:max_num_sentences]):
        print("=============")
        print("french:", pair[0])
        print("english:", pair[1])
        output_words, attentions = translate(
            encoder, decoder, pair[0], src_vocab, tgt_vocab
        )
        output_sentence = " ".join(output_words)
        output_sentences.append(output_sentence)
        output_attentions.append(attentions)
        # sys.exit(0)
        print("translation:", output_sentence)

        draw_annotations(pair[0], output_sentence, attentions, ind)

    return output_sentences, output_attentions


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#


def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, attentions = translate(
            encoder, decoder, pair[0], src_vocab, tgt_vocab
        )
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


######################################################################


def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """

    "*** YOUR CODE HERE ***"
    pass


def translate_and_show_attention(
    input_sentence, encoder1, decoder1, src_vocab, tgt_vocab
):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab
    )
    print("input =", input_sentence)
    print("output =", " ".join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return " ".join(strx.replace("@@ ", "").replace(EOS_token, "").strip().split())


######################################################################


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="hidden size of encoder/decoder, also word vector size",
    )
    ap.add_argument(
        "--n_samples",
        default=100000,
        type=int,
        help="total number of examples to train on",
    )
    ap.add_argument(
        "--n_epochs",
        default=1,
        type=int,
        help="total number of epochs",
    )
    ap.add_argument(
        "--print_every",
        default=5000,
        type=int,
        help="print loss info every this many training examples",
    )
    ap.add_argument(
        "--checkpoint_every",
        default=10000,
        type=int,
        help="write out checkpoint every this many training examples",
    )
    ap.add_argument(
        "--initial_learning_rate", default=0.001, type=int, help="initial learning rate"
    )
    ap.add_argument(
        "--src_lang", default="fr", help='Source (input) language code, e.g. "fr"'
    )
    ap.add_argument(
        "--tgt_lang", default="en", help='Source (input) language code, e.g. "en"'
    )
    ap.add_argument(
        "--train_file",
        default="data/fren.train.bpe",
        help="training file. each line should have a source sentence,"
        + 'followed by "|||", followed by a target sentence',
    )
    ap.add_argument(
        "--dev_file",
        default="data/fren.dev.bpe",
        help="dev file. each line should have a source sentence,"
        + 'followed by "|||", followed by a target sentence',
    )
    ap.add_argument(
        "--test_file",
        default="data/fren.test.bpe",
        help="test file. each line should have a source sentence,"
        + 'followed by "|||", followed by a target sentence'
        + " (for test, target is ignored)",
    )
    ap.add_argument(
        "--out_file", default="out.txt", help="output file for test translations"
    )
    ap.add_argument("--load_checkpoint", nargs=1, help="checkpoint file to start from")

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state["iter_num"]
        src_vocab = state["src_vocab"]
        tgt_vocab = state["tgt_vocab"]
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(
            args.src_lang, args.tgt_lang, args.train_file
        )
    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(
        device
    )

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state["enc_state"])
        decoder.load_state_dict(state["dec_state"])

    # read in datafiles
    n_samples = args.n_samples
    train_pairs = split_lines(args.train_file)[:n_samples]
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    # .parameters() returns generator
    encoder_optimizer = optim.Adam(list(encoder.parameters()), lr=args.initial_learning_rate)
    decoder_optimizer = optim.Adam(list(decoder.parameters()), lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        encoder_optimizer.load_state_dict(state["encoder_opt_state"])
        encoder_optimizer.load_state_dict(state["encoder_opt_state"])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    encoder.train()
    decoder.train()
    epochs = args.n_epochs
    batches = [train_pairs]

    for epoch in range(epochs):
        print(f"{epoch=}")
        iter_num = 0
        for maxi_batch in batches:
            # use max length in maxibatch?
            for mini_batch in maxi_batch:
                iter_num += len(mini_batch)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = train_mini_batch(
                    [mini_batch], encoder, decoder, criterion, src_vocab, tgt_vocab
                )
                print_loss_total += loss
                print("loss", loss)
                
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                if iter_num % args.checkpoint_every == -1:
                    state = {
                        "iter_num": iter_num,
                        "enc_state": encoder.state_dict(),
                        "dec_state": decoder.state_dict(),
                        "encoder_opt_state": encoder_optimizer.state_dict(),
                        "decoder_opt_state": decoder_optimizer.state_dict(),
                        "src_vocab": src_vocab,
                        "tgt_vocab": tgt_vocab,
                    }
                    filename = "state_%010d.pt" % iter_num
                    torch.save(state, filename)
                    logging.debug("wrote checkpoint to %s", filename)

                if iter_num % args.print_every == -1:
                    print_loss_avg = print_loss_total / args.print_every
                    print_loss_total = 0
                    logging.info(
                        "time since start:%s (epoch: %d, iter:%d iter/n_samples:%d%%) loss_avg:%.4f",
                        time.time() - start,
                        epoch,
                        iter_num,
                        iter_num / args.n_samples * 100,
                        print_loss_avg,
                    )
                    # translate from the dev set
                    translate_random_sentence(
                        encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2
                    )
                    translated_sentences, attns = translate_sentences(
                        encoder, decoder, dev_pairs, src_vocab, tgt_vocab
                    )
                    encoder.train()
                    decoder.train()

                    references = [
                        [
                            clean(pair[1]).split(),
                        ]
                        for pair in dev_pairs[: len(translated_sentences)]
                    ]
                    candidates = [clean(sent).split() for sent in translated_sentences]
                    dev_bleu = corpus_bleu(references, candidates)
                    logging.info("Dev BLEU score: %.2f", dev_bleu)

    # while iter_num < args.n_iters:
    #     iter_num += 1
    #     training_pair = tensors_from_pair(
    #         src_vocab, tgt_vocab, random.choice(train_pairs)
    #     )
    #     input_tensor = training_pair[0]
    #     target_tensor = training_pair[1]
    #     loss = train(
    #         input_tensor, target_tensor, encoder, decoder, optimizer, criterion
    #     )

    # translate test set and write to file
    # TODO make this into a flag
    # translated_sentences = translate_sentences(
    #     encoder, decoder, test_pairs, src_vocab, tgt_vocab
    # )
    # with open(args.out_file, "wt", encoding="utf-8") as outf:
    #     for sent in translated_sentences:
    #         outf.write(clean(sent) + "\n")

    # Visualizing Attention
    translated_sentences, attns = translate_sentences(
        encoder, decoder, train_pairs, src_vocab, tgt_vocab
    )

    
    # translate_and_show_attention(
    #     "on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab
    # )
    # translate_and_show_attention(
    #     "j en suis contente .", encoder, decoder, src_vocab, tgt_vocab
    # )
    # translate_and_show_attention(
    #     "vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab
    # )
    # translate_and_show_attention(
    #     "c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab
    # )


if __name__ == "__main__":
    main()
