import os
import argparse

import numpy as np
import pylab as plt

from codecs import open
REAL = np.float32

def split(line, sep=' '):
    parts = map(lambda x: x.replace(',', '.'), line.split(sep))

    (w, v) = parts[0], np.array(parts[1:], dtype=float)
    # norm_v = np.sum(v * v)
    return w, v

# class read_files(object):
#     def __init__(self, *sources):
#         self.sources = sources

#     def __iter__(self):
#         for source in self.sources:
#             with open(source, 'r', 'utf-8') as input:
#                 for line in input:
#                     yield line.split('|', 1)[1].split()

def load_txt(filename, start=0, end=-1, sep='\t', prefix=""):
    print 'loading vectors from text file %s[%i:%i], with prefix "%s"' % (filename, start, end, prefix)
    c = 0
    end = end if end >= 0 else 1000000
    vectors = None
    words = []

    for line in open(filename, 'r', 'utf-8'):
        if start <= c < end:
            w, v = split(line, sep)
            words.append(prefix + w)
            if vectors is None:
                vectors = np.zeros((end - start, len(v)))
            vectors[c - start] = v
        if c >= end:
            break
        c += 1
    return words, vectors


def load_bin(filename, start=0, end=-1, prefix="", normalize=True):
    print 'loading vectors from bin file %s[%i:%i], with prefix "%s"' % (filename, start, end, prefix)

    with open(filename, 'rb') as fin:
        header = to_unicode(fin.readline(), 'utf-8')
        vocab_size, vec_size = map(int, header.split())  # throws for invalid file format
        if end < 0:
            end = vocab_size

        binary_len = np.dtype(REAL).itemsize * vec_size
        words = []
        vectors = np.zeros((end - start, vec_size))
        off = 0
        c = 0

        for line_no in xrange(vocab_size):
            # mixed text and binary: read text first, then binary
            word = []
            while True:
                ch = fin.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                    word.append(ch)
            v = np.fromstring(fin.read(binary_len), dtype=REAL)

            if start <= c < end:
                word = to_unicode(b''.join(word))
                words.append(word)
                if normalize:
                    v = v / np.linalg.norm(v)
                vectors[off] = v
                off += 1
            if c >= end:
                break
            c += 1
        return words, vectors

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    else:
        return unicode(text, encoding, errors=errors)

def load_files(filenames, start=0, end=100, sep='\t'):
    for f in filenames:
        f, f_start, f_end = extract_start_end(f, start, end)
        if f.endswith('.txt'):
            yield load_txt(f, f_start, f_end, sep)
        else:
            yield load_bin(f, f_start, f_end)


def extract_start_end(filename, start, end):
    if filename.endswith(']'):
        x = filename.rfind('[')
        y = filename.rfind(':')
        start = int(filename[x + 1:y])
        end = int(filename[y + 1:-1])
        filename = filename[:x]
    return filename, start, end


def plot(w, Y, pointsize=20, color='blue', fontsize=10):
    xs, ys = Y[:, 0], Y[:, 1]
    plt.scatter(xs, ys, s=pointsize, c=color)

    for label, x, y in zip(w, xs, ys):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(-20, 20),
            textcoords='offset points',
            fontsize=fontsize
        )


if __name__ == '__main__':

    np.random.seed(1234)

    # CORPUS = 'data/rotten/flat_dev_root.txt'
    # VECTORS = None
    # single_file = False

    parser = argparse.ArgumentParser('Visualize words')
    parser.add_argument(dest='files', nargs='*')
    parser.add_argument('--load', default=None, type=str)

    parser.add_argument('--no_labels', action='store_true')
    parser.add_argument('--perplexity', '-p', default=30, type=float)

    parser.add_argument('--iter', default=1000, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=100, type=int)
    parser.add_argument('--lr', default=500, type=float)
    parser.add_argument('--reg', default=0, type=float)
    parser.add_argument('--rot', default=0, type=float)

    parser.add_argument('--tab', action='store_true')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--fontsize', default=8, type=int)
    parser.add_argument('--pointsize', default=20, type=int)

    args = parser.parse_args()

    files = args.files
    args.end = args.end if args.end > 0 else 1000000

    sep = '\t' if args.tab else ' '

    plt.ion()

    colors = ['b', 'r', 'g', 'y', 'cyan', 'grey', 'pink']
    vectors = []
    words = []
    count = args.end - args.start
    i = 0
    # tsne_path = args.vectors.replace('.txt', '.tsne.npy')
    dot = files[0].rfind('.')
    dot = dot if dot >= 0 else len(files[0])
    tsne_path = files[0][:dot] + '.%i-%i.tsne' % (args.start, args.end)

    w2vs = list(load_files(files, start=args.start, end=args.end))
    for _, v in w2vs:
        v -= np.mean(v, axis=0, keepdims=True)
    X = np.concatenate(map(lambda w2v: w2v[1], w2vs))

    if args.load is None:
        if args.slow:
            from CorpusViz.slow_tsne import tsne
            Y = tsne(X, perplexity=args.perplexity, max_iter=args.iter, reg=args.reg, eta=args.lr)
            np.save(tsne_path + '.npy', Y)
        elif args.pca:
            from CorpusViz.slow_tsne import pca
            Y = pca(X, 2)
        else:
            from tsne import bh_sne as tsne
            Y = tsne(X, perplexity=args.perplexity)

        plt.clf()
    else:
        print('loading results from', args.load)
        Y = np.load(args.load)
        print('loading done')

    import matplotlib.patches as mpatches

    patches = [mpatches.Patch(color=c, label=l[:2]) for c, l in zip(colors, files)]
    plt.legend(handles=patches)

    if args.rot != 0:
        alpha = args.rot * 3.141 / 180
        cos = np.cos(alpha)
        sin = np.sin(alpha)
        rot = np.array([[cos, sin], [-sin, cos]])
        Y = np.dot(Y, rot)

    e = 0
    for i, (w, v) in enumerate(w2vs):
        s = e
        e += v.shape[0]
        plot(w, Y[s:e, :], pointsize=args.pointsize, color=colors[i], fontsize=args.fontsize)

    plt.pause(0.0000000001)
    plt.show()
