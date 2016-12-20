import logging
import sys
import os
import argparse

import numpy as np
import pylab as plt

from codecs import open
from matplotlib.widgets import Cursor, Button

from Doc2Vec.word2vec import Word2Vec
from Doc2Vec.doc2vec import Doc2Vec
from DCNN.train_sentiment import load_treated_vectors

from Util import pipeline as pip


# class read_files(object):
#     def __init__(self, *sources):
#         self.sources = sources

#     def __iter__(self):
#         for source in self.sources:
#             with open(source, 'r', 'utf-8') as input:
#                 for line in input:
#                     yield line.split('|', 1)[1].split()

def read_files(*sources):
    return pip.FilesReader(sources)


def w2v(input, vectors_path, neg_sampling=4):
    vectors, lexicon, dim = load_treated_vectors(vectors_path)
    w2v = Word2Vec(size=dim, negative=neg_sampling, hs=0, min_count=1)
    w2v.build_vocab(input)

    loaded = 0
    for i in range(len(w2v.index2word)):
        j = w2v.index2word[i]
        if j in lexicon:
            l = lexicon[j]
            w2v.semantic[i] = vectors[l]
            loaded += 1

    print 'loaded', loaded, 'words'
    return w2v


def loadDocVectors(file_path):
    with open(file_path, 'r', 'utf-8') as input:
        n, dim = input.readline().split()
        n, dim = int(n), int(dim)
        doc2vec = np.zeros((n, dim))

        for i, line in enumerate(input):
            parts = line.split(' ')
            doc2vec[i, :] = np.array(parts[1:]).astype(float)

    return doc2vec


def load_labels(file_path):
    l = []
    with open(file_path, 'r', 'utf-8') as input:
        for line in input:
            l.append(float(line.split('|', 1)[0]))

    return np.array(l)


# License: Creative Commons Zero (almost public domain) http://scpyce.org/cc0

"""
Example usage of matplotlibs widgets: Build a small 2d Data viewer.

Shows 2d-data as a pcolormesh, a click on the image shows the crossection
(x or y, depending on the mouse button) and draws a corresponding line in
the image, showing the location of the crossections. A reset button deletes all
crossections plots.

Works with matplotlib 1.0.1.
"""


plt.rcParams['font.size'] = 8


class viewer_2d(object):
    def __init__(self, z, verbatims=None, labels=None):
        """
        Shows a given array in a 2d-viewer.
        Input: z, an 2d array.
        x,y coordinters are optional.
        """

        self.z = z
        self.verbatims = verbatims
        self.labels = labels

        self.fig = plt.figure()

        # Doing some layout with subplots:
        self.fig.subplots_adjust(0.05, 0.05, 0.98, 0.98, 0.1)
        self.overview = plt.subplot2grid((8, 4), (0, 0), rowspan=7, colspan=4)

        self.draw()

        # Adding widgets, to not be gc'ed, they are put in a list:
        cursor = Cursor(self.overview, useblit=True, color='black', linewidth=2)
        but_ax = plt.subplot2grid((8, 4), (7, 0), colspan=1)
        reset_button = Button(but_ax, 'Reset')
        self._widgets = [cursor, reset_button]

        # connect events
        reset_button.on_clicked(self.clear_xy_subplots)
        self.fig.canvas.mpl_connect('button_press_event', self.click)

    def draw(self):
        self.overview.cla()
        if self.labels is not None:
            self.overview.scatter(self.z[:, 0], self.z[:, 1], 20, self.labels)
        else:
            self.overview.scatter(self.z[:, 0], self.z[:, 1], 20)
        plt.draw()

    def clear_xy_subplots(self, event):
        """Clears the subplots."""
        for j in [self.overview]:
            j.lines = []
            j.legend_ = None
        plt.draw()

    def closest(self, x, y):
        d = np.square(self.z[:, 0] - x) + np.square(self.z[:, 1] - y)
        return np.argmin(d)

    def click(self, event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """

        # Get nearest data
        if event.inaxes == self.overview:
            i = self.closest(event.xdata, event.ydata)
            self.overview.plot(self.z[i, 0], self.z[i, 1], 'ro')
            if self.verbatims is not None:
                print '*', self.verbatims[i]

            plt.draw()

if __name__ == '__main__':

    np.random.seed(1234)

    CORPUS = 'data/rotten/flat_dev_root.txt'
    # VECTORS = None
    VECTORS = 'data/mikolov_rotten'

    parser = argparse.ArgumentParser('Trains a sentimental CNN')
    parser.add_argument('--corpus', default=CORPUS, help='chooses corpus path')
    parser.add_argument('--vectors', default=VECTORS, help='chooses the embedding vectors')
    parser.add_argument('--docvectors', default=None)
    parser.add_argument('--points', default=None)

    parser.add_argument('--no_labels', action='store_true')
    parser.add_argument('--perplexity', default=20, type=float)

    parser.add_argument('--neural', action='store_true')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--iter', default=1000, type=int)

    args = parser.parse_args()
    corpus = args.corpus

    plt.ion()

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    if args.docvectors is None and args.points is None:
        if args.vectors is None:
            input_file = args.corpus
            model = Word2Vec(read_files(input_file), size=100, window=5, sg=True, negative=1, hs=False, min_count=1, workers=8)
            model.save(input_file + '.bin')
            model.save_word2vec_format(input_file + '.vec')
            model_path = input_file + '.vec'
        else:
            model_path = args.vectors
            model = w2v(read_files(args.corpus), args.vectors)

        model = Doc2Vec(read_files(corpus), w2v=model, hs=0, sg=5, negative=1)
        model.save_doc2vec_format(corpus + '.docvec')

        program = os.path.basename(sys.argv[0])
        logging.info("finished running %s" % program)
        doc2vec = loadDocVectors(corpus + '.docvec')
    else:
        if args.docvectors.endswith('.docsem'):
            doc2vec = pip.FileReader(args.docvectors).map(lambda x: np.array(x.split(), dtype=float)).to_list()
            doc2vec = np.array(doc2vec)
            doc2vec = doc2vec / np.max(np.abs(doc2vec))
        elif args.docvectors.endswith('.docvec'):
            doc2vec = loadDocVectors(corpus.replace('.txt', '.docvec'))
        else:
            raise Exception('Format not supported (expected .docvec or .docsem)')

    print doc2vec.shape
    print doc2vec[0, :10]
    print '...'
    print doc2vec[-2, :10]
    print np.max(doc2vec), np.min(doc2vec), np.mean(np.abs(doc2vec))

    if args.points is None:
        if args.neural:
            from CorpusViz.neural_tsne import tsne
            _, Y = tsne(doc2vec, perplexity=args.perplexity, epoch=args.iter)
            np.save(corpus.replace('.txt', '_neural_tsne'), Y)
        elif args.slow:
            from CorpusViz.slow_tsne import tsne
            Y = tsne(doc2vec, perplexity=args.perplexity, max_iter=args.iter)
            np.save(corpus.replace('.txt', '_tsne'), Y)
        else:
            from tsne import bh_sne as tsne
            Y = tsne(doc2vec, perplexity=args.perplexity)
    else:
        Y = np.load(args.points)

    if not args.no_labels:
        labels = load_labels(corpus)
        labels_5 = np.minimum(4, 5 * labels).astype(int)
        labels_3 = np.sign(labels_5 - 2) + 1
        plt.scatter(Y[:, 0], Y[:, 1], 20, labels_5)
        plt.show()
        plt.scatter(Y[:, 0], Y[:, 1], 20, labels_3)
        plt.show()
    else:
        if args.corpus:
            verbatims = list(pip.FileReader(corpus))
            viewer_2d(Y, verbatims=verbatims)
        else:
            viewer_2d(Y)
        plt.show()
