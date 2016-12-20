# -*- coding: utf-8 -*-
import numpy as np
import argparse
import re

from codecs import open
from os.path import join
from os import listdir

import Util.pipeline as pip


# SENTENCES_TRAIN = 'flat_train.txt'
SENTENCES_TRAIN = 'flat_train_root.txt'
SENTENCES_DEV = 'flat_dev_root.txt'
SENTENCES_TEST = 'flat_test_root.txt'

TREATED_CASE_SENSITIVE = 'lexicon.CS.txt'
TREATED_CASE_INSENSITIVE = 'lexicon.CI.txt'
TREATED_UNKNOWN = 'lexicon.unk.txt'
TREATED_STAT = 'lexicon.stat.txt'


def load_stat(vectors_path):
    with open(join(vectors_path, TREATED_STAT), 'r', 'utf-8') as f:
        dim = int(f.readline().split(' ')[-1])
        CS = int(f.readline().split(' ')[-1])
        CI = int(f.readline().split(' ')[-1])
        unk = int(f.readline().split(' ')[-1])

    return dim, CS + CI + unk


def load_treated_vectors(vectors_path, codec='utf-8'):
    """load embeddings of words treated for the given vocab"""

    print "Loading word vectors from:", vectors_path
    dim, vocab_size = load_stat(vectors_path)

    vectors = np.zeros((vocab_size, dim), float)
    # unknown = np.zeros((vocab_size), bool)

    lexicon = {}

    def add_word(word_vec):
        word, vec = word_vec
        vectors[len(lexicon)] = np.array(vec, dtype=float)
        lexicon[word] = len(lexicon)

    def split(line):
        parts = line.split(' ')
        return parts[0], np.array(parts[1:], dtype=float)

    print "Loading CS..."
    CS = join(vectors_path, TREATED_CASE_SENSITIVE)
    pip.FileReader(CS).map(split).foreach(add_word)

    print "Loading CI..."
    CI = join(vectors_path, TREATED_CASE_INSENSITIVE)
    pip.FileReader(CI).map(split).foreach(add_word)

    # print "Loading UNK..."
    # unk_map = {}
    # with open(join(vectors_path, TREATED_UNKNOWN), 'r', codec) as f:
    #     for line in f:
    #         parts = line.split('\n')
    #         unknown[lexicon[parts[0]]] = True
    #         unk_map[lexicon[parts[0]]] = len(unk_map)

    # return vectors, unknown, unk_map
    return vectors, lexicon, dim


def default_vec(dim):
    return np.zeros(dim)


def read_dataset(filename, vectors, lexicon, dim, remove_duplicate=False):

    X, y = [], []

    all_sentences = set()

    with open(filename, 'r', 'utf-8') as input:
        for line in input:
            label, sentence = line.split('|', 1)

            if remove_duplicate:
                if sentence in all_sentences:
                    continue
                else:
                    all_sentences.add(sentence)

            words = sentence.split()
            x = np.zeros((len(words), dim))

            for i in range(len(words)):
                if words[i] in lexicon:
                    x[i] = vectors[lexicon[words[i]]]
                else:
                    x[i] = default_vec(dim)

            X.append(x)
            y.append(float(label))

    X, y = np.array(X), np.array(y)

    assert X.shape[0] == y.shape[0]

    print 'Loaded %i sentences from' % X.shape[0], filename
    return X, y


def read_vocab_from_corpus(corpus, *filenames):
    print 'reading vocabulary from corpus:', corpus

    vocab = pip.FilesReader([join(corpus, f) for f in filenames])\
        .plug_to(pip.VocabReader())

    # for filename in filenames:
    #     with open(join(corpus, filename), 'r', 'utf-8') as input:
    #         for line in input:
    #             label, sentence = line.split('|', 1)
    #             for word in sentence.split():
    #                 vocab.add(word)

    print 'found %i words.' % len(vocab)
    return vocab


def extract_vectors_for_corpus(vectors, corpus):
    vocab = read_vocab_from_corpus(corpus, SENTENCES_DEV, SENTENCES_TEST, SENTENCES_TRAIN)
    return extract_vectors_for_vocab(vectors, vocab)


def extract_vectors_for_vocab(vectors, vocab):

    from Doc2Vec.word2vec import Word2Vec

    def copen(filename):
        return open(join(vectors, filename), 'w', 'utf-8')

    filename = filter(lambda x: x.endswith('.sem'), listdir(vectors))[0]
    w2v = Word2Vec.load_word2vec_format(join(vectors, filename), binary=True, norm_only=False)

    def write_word(out, word, word2, vec=True):
        out.write(word)
        if vec:
            for x in w2v[word2]:
                out.write(' ')
                out.write(unicode(x))
        out.write('\n')

    count_CS, count_CI, count_unk = 0, 0, 0

    def try_method(word, f):
        word2 = f(word)
        if word2 in w2v.vocab:
            write_word(CI, word, word2)
        return word2 in w2v.vocab

    with copen(TREATED_CASE_SENSITIVE) as CS:
        with copen(TREATED_CASE_INSENSITIVE) as CI:
            with copen(TREATED_UNKNOWN) as unk:
                for word in vocab:
                    if word in w2v.vocab:
                        write_word(CS, word, word)
                        count_CS += 1
                    elif try_method(word, lambda w: w.lower()):
                        count_CI += 1
                    elif try_method(word, lambda w: w.upper()):
                        count_CI += 1
                    elif try_method(word, lambda w: w.capitalize()):
                        count_CI += 1
                    elif try_method(word, lambda w: re.sub(r'[0-9]', '0', w)):
                        count_CI += 1
                    elif try_method(word, lambda w: w.replace('-', '_')):
                        count_CI += 1
                    else:
                        write_word(unk, word, word, vec=False)
                        count_unk += 1

    with copen(TREATED_STAT) as stat:
        stat.write('dim: %i\n' % w2v.layer1_size)
        stat.write('CS : %i\n' % count_CS)
        stat.write('CI : %i\n' % count_CI)
        stat.write('unk: %i\n' % count_unk)

def main(corpus_path, vectors_path, n_cat, deep=False, load=None, out=None, lr=0.1):

    vectors, lexicon, dim = load_treated_vectors(vectors_path)

    def read(filename):
        return read_dataset(filename, vectors, lexicon, dim)

    X_train, y_train = read(join(corpus_path, SENTENCES_TRAIN))
    X_dev, y_dev = read(join(corpus_path, SENTENCES_DEV))
    X_test, y_test = read(join(corpus_path, SENTENCES_TEST))

    cat = np.vectorize(lambda x: min(n_cat - 1, int(n_cat * x)))
    y_train = cat(y_train)
    y_dev = cat(y_dev)
    y_test = cat(y_test)

    if deep:
        from DCNN_with_W2V_transf import CNN
    else:
        from DCNN import CNN

    cnn = CNN(vector_size=dim, num_labels=n_cat)
    if load is not None:
        cnn.load_model(load)

    # print 'Results before training:'
    # print 'Validation Accuracy : ', cnn.accuracy(X_dev, y_dev), '\n'
    cnn.fit(
        X_train, y_train, X_dev, y_dev,
        information_freq=1,
        results_file=out,
        method='adadelta',
        lr=lr,
        reg=0.001,
        epoch=500
        )

    test_acc = cnn.accuracy(X_test, y_test)
    print 'Accurracy on test:', test_acc
    return cnn


if __name__ == '__main__':

    CORPUS = 'data/lm'
    VECTORS = 'data/wtf_lm'
    # CORPUS = 'data/rotten'
    # VECTORS = 'data/mikolov_rotten'

    parser = argparse.ArgumentParser('Trains a sentimental CNN')
    parser.add_argum ent('--corpus', default=CORPUS, help='chooses corpus path')
    parser.add_argument('--vectors', default=VECTORS, help='chooses the embedding vectors')
    parser.add_argument('--cat', default=5, help='choose number of classes')
    parser.add_argument('--deep', action='store_true')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--load', default=None, help='load a model from file')
    parser.add_argument('--prepare', action='store_true')

    args = parser.parse_args()

    fit_parser = argparse.ArgumentParser('Trains a sentimental CNN')
    fit_parser.add_argument('--out', default=None, dest='results_file')
    # fit_parser.add_argument('--save_at', default='test_results')
    fit_parser.add_argument('--method', default='adadelta')
    fit_parser.add_argument('--lr', default=0.1, type=float)
    fit_parser.add_argument('--information_freq', default=5, type=int)
    fit_parser.add_argument('--train_acc', action='store_true')
    fit_parser.add_argument('--epoch', default=500, type=int)

    fit_args, _ = fit_parser.parse_known_args()
    print 'main args:', args
    print 'fit args:', args

    if args.prepare:
        vocab = read_vocab_from_corpus(args.corpus, *['lm_q%i_tokenized.txt' % i for i in range(1, 7)])
        extract_vectors_for_vocab(args.vectors, vocab)
        # extract_vectors_for_corpus(args.vectors, args.corpus)
    else:
        main(args.corpus, args.vectors, args.cat, args.deep, args.load, **vars(fit_args))
