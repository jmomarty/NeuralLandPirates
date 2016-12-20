__author__ = 'jmm'

import codecs
import random
import numpy as np
import sys
import gensim
from itertools import izip

def load_word2vec(data_file):

    word2vec = gensim.models.Word2Vec.load_word2vec_format(data_file, binary=True)

    return word2vec

def create_training_data(proportion, corpus, targets, word2vec, word2vec_size):

    X, y, z = [], [], []
    counter = 0
    f = codecs.open(corpus, 'r', encoding="utf-8")
    g = codecs.open(targets,'r', encoding="utf-8")
    for line1, line2 in izip(f,g):
        if random.random() > float(proportion):
            words = line1.split(u" ")
            target = line2.replace(u'[',u'').replace(u']',u'').split(u' > ')
            sentence_matrix = np.zeros((len(words), word2vec_size))
            for i in range(len(words)):
                try:
                    sentence_matrix[i] = word2vec[words[i]].reshape(word2vec_size)
                except:
                    sentence_matrix[i] = np.zeros((word2vec_size,))
            X.append(sentence_matrix)
            counter += 1
            if counter % 1000 ==0:
                print "counter : ", counter
            z.append(target[0])
            y.append(target[1])

    X, y, z = np.array(X), np.array(y), np.array(z)
    training_size = int(1e6*(1-float(proportion)))
    file_name = u"training_arrays_" + unicode(word2vec_size) + u'_'+  unicode(training_size)
    np.savez(file_name, X_train= X, y_train= y, z_train=z)

if __name__ == '__main__':

    source = sys.argv[1:]
    corpus = source[0]
    targets = source[1]
    word2vec_file = source[2]
    word2vec_size = source[3]
    proportion = source[4]

    print u"Loading Word2Vec... (this might take a few minutes)\n"
    word2vec = load_word2vec(word2vec_file)
    print u"Word2Vec loaded!\n"

    print u"Creating Training Arrays...\n"
    create_training_data(proportion, corpus, targets, word2vec, int(word2vec_size))
    print u"Training Arrays created!\n"
