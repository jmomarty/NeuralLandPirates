__author__ = 'jmm'

import cPickle
import codecs
import random
import numpy as np
import sys
import gensim

def load_word2vec(data_file):

    word2vec = gensim.models.Word2Vec.load_word2vec_format(data_file, binary=True)

    return word2vec

def create_training_data(proportion, corpus, word2vec, word2vec_size):

    X, y, z = [], [], []
    counter = 0
    with codecs.open(corpus, 'r', encoding="utf-8") as f:
        for line in f:
            if random.random() > float(proportion):
                temp = line.split(u"\t")
                if len(temp) < 3:
                    continue
                else:
                    temp2 = temp[0].replace(u"[", u"").replace(u"]", u"").split(u" > ")
                    if temp2[0] != u'divers':
                        try:
                            words = temp[-2].replace(u"[verbatim : ", u"").replace(u"]", u"")
                            words = words.split(u" ")
                            sentence_matrix = np.zeros((len(words), word2vec_size))
                            for i in range(len(words)):
                                if words[i] != u'':
                                    try:
                                        sentence_matrix[i]=word2vec[words[i]].reshape(word2vec_size)
                                    except:
                                        pass
                            X.append(sentence_matrix)
                            counter += 1
                            if counter % 100==0:
                                print "counter : ", counter
                            try:
                                y.append(temp2[1])
                                z.append(temp2[0])
                            except:
                                y.append(temp2[0])
                                z.append(temp2[0])
                        except:
                            continue
    X, y, z = np.array(X), np.array(y), np.array(z)
    print u"\n"
    training_size = int(1e6*(1-float(proportion)))
    file_name = u"training_arrays_" + unicode(word2vec_size) + u'_'+  unicode(training_size)
    np.savez(file_name, X_train= X, y_train= y, z_train=z)

if __name__ == '__main__':

    source = sys.argv[1:]
    corpus = source[0]
    word2vec_file = source[1]
    word2vec_size = source[2]
    proportion = source[3]

    print u"Loading Word2Vec... (this might take a few minutes)\n"
    word2vec = load_word2vec(word2vec_file)
    print u"Word2Vec loaded!\n"

    print u"Creating Training Arrays...\n"
    create_training_data(proportion, corpus, word2vec, int(word2vec_size))
    print u"Training Arrays created!\n"
