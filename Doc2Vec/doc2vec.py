#aba : doc2vec , basically this is just a modified version of gensim's wordvec
#the helper functions are all in matutils to remove dependency of gensim
from word2vec import Word2Vec
import logging
import os
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, take, random, dtype, get_include, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum

# logger = logging.getLogger("gensim.models.word2vec")
logger = logging.getLogger("doc2vec")

# This is where the dependency was removed
# from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange


# Modified Word2Vec code to work for sentences and docs
class Doc2Vec(utils.SaveLoad):
    def __init__(self, sentences, model_file=None, w2v=None, alpha=0.025, window=5, sample=0, seed=1,
        workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, iteration=1):
        self.sg = int(sg)
        self.table = None  # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.sample = sample
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.iteration = iteration

        if model_file is not None:
            self.w2v = Word2Vec.load(model_file)
        elif w2v is not None:
            self.w2v = w2v

        if sentences is not None:
            self.vocab = self.w2v.vocab
            self.layer1_size = self.w2v.layer1_size
            self.reset_sent_vec(sentences)
            for i in range(iteration):
                self.train_sent(sentences)

    def reset_sent_vec(self, sentences):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting vectors for sentences")
        random.seed(self.seed)
        self.sents_len = 0
        for sent in sentences:
            self.sents_len += 1
        self.sents = empty((self.sents_len, self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(self.sents_len):
            self.sents[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size

    def train_sent(self, sentences, total_words=None, word_count=0, sent_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        logger.info("training model with %i workers on %i sentences and %i features, "
                    "using 'skipgram'=%s 'hierarchical softmax'=%s 'subsample'=%s and 'negative sampling'=%s" %
                    (self.workers, self.sents_len, self.layer1_size, self.sg, self.hs, self.sample, self.negative))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        sent_count = [sent_count]
        total_words = total_words or sum(v.count for v in itervalues(self.vocab))
        total_sents = self.sents_len * self.iteration
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                    # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                if self.sg:
                    job_words = sum(self.train_sent_vec_sg(self.w2v, sent_no, sentence, alpha, work)
                                    for sent_no, sentence in job)
                else:
                    job_words = sum(self.train_sent_vec_cbow(self.w2v, sent_no, sentence, alpha, work, neu1)
                                    for sent_no, sentence in job)
                with lock:
                    word_count[0] += job_words
                    sent_count[0] += chunksize
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% sents, alpha %.05f, %.0f words/s" %
                                    (100.0 * sent_count[0] / total_sents, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for sent_no, sentence in enumerate(sentences):
                sampled = [self.vocab.get(word, None) for word in sentence]
                yield (sent_no, sampled)

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
                    (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]

    def train_sent_vec_cbow(self, model, sent_no, sentence, alpha, work=None, neu1=None):
        """
        Update CBOW model by training on a single sentence.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Word2Vec.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """
        sent_vec = self.sents[sent_no]
        if self.negative:
            # precompute negative labels
            labels = zeros(self.negative + 1)
            labels[0] = 1.

        for pos, word in enumerate(sentence):
            if word is None:
                continue  # OOV word in the input sentence => skip
            reduced_window = random.randint(self.window)  # `b` in the original word2vec code
            start = max(0, pos - self.window + reduced_window)
            window_pos = enumerate(sentence[start: pos + self.window + 1 - reduced_window], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.context[word2_indices], axis=0)  # 1 x layer1_size
            l1 += sent_vec
            if word2_indices and self.cbow_mean:
                l1 /= len(word2_indices)
            neu1e = zeros(l1.shape)

            if self.hs:
                l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
                fa = 1. / (1. + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
                ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                # model.syn1[word.point] += outer(ga, l1) # learn hidden -> output
                neu1e += dot(ga, l2a)  # save error

            if self.negative:
                # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                word_indices = [word.index]
                while len(word_indices) < self.negative + 1:
                    w = model.table[random.randint(model.table.shape[0])]
                    if w != word.index:
                        word_indices.append(w)
                l2b = model.semantic[word_indices]  # 2d matrix, k+1 x layer1_size
                fb = 1. / (1. + exp(-dot(l1, l2b.T)))  # propagate hidden -> output
                gb = (labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
                # model.semantic[word_indices] += outer(gb, l1) # learn hidden -> output
                neu1e += dot(gb, l2b)  # save error

            # model.context[word2_indices] += neu1e # learn input -> hidden, here for all words in the window separately
            self.sents[sent_no] += neu1e  # learn input -> hidden, here for all words in the window separately

        return len([word for word in sentence if word is not None])

    def train_sent_vec_sg(self, model, sent_no, sentence, alpha, work=None):
        """
        Update skip-gram model by training on a single sentence.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Word2Vec.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """
        if self.negative:
            # initialize negative labels
            labels = zeros(self.negative + 1)
            labels[0] = 1.0
            word_indices = zeros(self.negative + 1, dtype='int')

        treated = 0

        for word in sentence:
            # don't train on OOV words and on the `word` itself
            if word:
                # l1 = model.context[word.index]
                l1 = self.sents[sent_no]
                dL1 = zeros(l1.shape)

                if self.hs:
                    # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
                    fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                    ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                    # model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
                    dL1 += dot(ga, l2a)  # save error

                if self.negative:
                    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                    word_indices[0] = word.index
                    neg_sampling = 1
                    while neg_sampling < self.negative + 1:
                        w = model.table[random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices[neg_sampling] = w
                        neg_sampling += 1

                    l2b = model.semantic[word_indices]  # 2d matrix, k+1 x layer1_size
                    pred = 1. / (1. + exp(-dot(l2b, l1)))  # propagate hidden -> output
                    delta = (labels - pred) * alpha  # vector of error gradients multiplied by the learning rate
                    # model.semantic[word_indices] += outer(delta, l1) # learn hidden -> output
                    dL1 += dot(delta, l2b)  # save error

                # model.context[word.index] += dL1  # learn input -> hidden
                self.sents[sent_no] += dL1  # learn input -> hidden
                treated += 1

        return treated

    def save_doc2vec_format(self, fname):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        """
        logger.info("storing %sx%s projection weights into %s" % (self.sents_len, self.layer1_size, fname))
        assert (self.sents_len, self.layer1_size) == self.sents.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % self.sents.shape))
            # store in sorted order: most frequent words at the top
            for sent_no in xrange(self.sents_len):
                row = self.sents[sent_no]
                fout.write(utils.to_utf8("sent_%d %s\n" % (sent_no, ' '.join("%f" % val for val in row))))

    def similarity(self, sent1, sent2):
        """
        Compute cosine similarity between two sentences. sent1 and sent2 are
        the indexs in the train file.

        Example::

          >>> trained_model.similarity(0, 0)
          1.0

          >>> trained_model.similarity(1, 3)
          0.73

        """
        return dot(matutils.unitvec(self.sents[sent1]), matutils.unitvec(self.sents[sent2]))

    #TODO compute most_similar like in gensim


#Helper class, initialize it with a filename pointing to a file with one sentence per line it will generate a stream
class LineSentence(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, source):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in self.source:
                yield utils.to_unicode(line).split()
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in fin:
                    yield utils.to_unicode(line).split()
