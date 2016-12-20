import numpy as np
import argparse

from os.path import join

import Util.pipeline as pip
from DCNN.train_sentiment import load_stat

TREATED_CASE_SENSITIVE = 'lexicon.CS.txt'
TREATED_CASE_INSENSITIVE = 'lexicon.CI.txt'
TREATED_UNKNOWN = 'lexicon.unk.txt'
TREATED_STAT = 'lexicon.stat.txt'


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

    print "Loading UNK..."
    unk = join(vectors_path, TREATED_UNKNOWN)
    unk = set(pip.FileReader(unk))

    return vectors, lexicon, dim, unk


def idf_base(input):
    idf = {}
    doc_count = np.zeros(1)

    def treat(line):
        for w in set(line.split()):
            idf[w] = idf[w] + 1 if w in idf else 1
        doc_count[0] += 1

    input.foreach(treat)

    for w, f in idf.iteritems():
        idf[w] = np.log(doc_count / idf[w])

    return idf


def continuous_idf(input, vectors, lexicon, dim, unk, threshold=0.8):
    idf = np.zeros(vectors.shape[0])
    doc_count = 0

    for doc in input:
        doc = doc.split()
        for word in set(doc):
            if word not in unk:
                vec = vectors[lexicon[word]]
                idf += (vectors.dot(vec) > threshold)
        doc_count += 1
        print '\r%i' % doc_count,

    idf_dict = {}
    for word, index in lexicon.iteritems():
        idf_dict[word] = np.log(doc_count / idf[index])

    return idf_dict


def tf_idf_vec(doc, idf, vectors, lexicon, dim, unk, log=False):
    v = np.zeros(dim)
    tf = {}
    doc = doc.split()

    for word in doc:
        tf[word] = tf[word] + 1 if word in tf else 1

    if log:
        for w, f in tf.iteritems():
            tf[w] = 0 if tf[w] < 1 else np.log(tf[w]) + 1

    for word in doc:
        if word not in unk:
            v += (tf[word] * idf[word]) * vectors[lexicon[word]]

    return v / np.linalg.norm(v)


if __name__ == '__main__':

    CORPUS = 'data/lm/lm_q4_tokenized.txt'
    VECTORS = 'data/wtf_lm'
    # CORPUS = 'data/rotten'
    # VECTORS = 'data/mikolov_rotten'

    parser = argparse.ArgumentParser('Trains a sentimental CNN')
    parser.add_argument('--corpus', default=CORPUS, help='chooses corpus path')
    parser.add_argument('--vectors', default=VECTORS, help='chooses the embedding vectors')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--log', action='store_true')

    args = parser.parse_args()

    vectors, lexicon, dim, unk = load_treated_vectors(args.vectors)
    if args.continuous:
        print 'computing continuous idf...'
        idf = continuous_idf(pip.FileReader(args.corpus), vectors, lexicon, dim, unk)
    else:
        print 'computing classic idf...'
        idf = idf_base(pip.FileReader(args.corpus))

    def write_arr(arr, out):
        out.write(unicode(arr[0]))
        for x in arr[1:]:
            out.write(' ')
            out.write(unicode(x))

    print 'computing tf-idf...'
    pip.FileReader(args.corpus)\
        .map(lambda doc: tf_idf_vec(doc, idf, vectors, lexicon, dim, unk))\
        .plug_to(pip.FileWriter(args.corpus.replace('.txt', '.docsem'), write=write_arr))
