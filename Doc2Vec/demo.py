#!/usr/bin/env python
# -*- coding: utf-8 -*-
# aba : just some tests to see that it is working

import logging
import sys
import os
from word2vec import Word2Vec
from doc2vec import Doc2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

input_file = 'test.txt'
model = Word2Vec(LineSentence(input_file), size=100, window=5, sg=0, min_count=5, workers=8)
model.save(input_file + '.model')
model.save_word2vec_format(input_file + '.vec')

#aba : initialize it with a already learned word vectors through model_file
sent_file = 'sent.txt'
model = Doc2Vec(LineSentence(sent_file), model_file=input_file + '.model')
model.save_doc2vec_format(sent_file + '.vec')

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)