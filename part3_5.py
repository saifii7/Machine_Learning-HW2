#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())


def word_filterer(trainfile):
    dict1 = {}
    for label, words in read_from(trainfile):
        for word in words:
            if word in dict1:
                dict1[word]+=1
            else:
                dict1[word]=1
    set_of_words = set(word for word, count in dict1.items() if count > 2)
    return set_of_words



def make_vector(words,set_of_words):
    v = svector()
    v['<bias>'] = 1  
    for word in words:
        if word in set_of_words:
            v[word] += 1
    return v

def test(devfile, model, set_of_words):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words,set_of_words))) <= 0
    return err/i  # i is |D| now


def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    best_model = svector()

    avg_model = svector()

    set_of_words = word_filterer(trainfile)

    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): 
            sent = make_vector(words, set_of_words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
            avg_model += model
        dev_err = test(devfile, avg_model,set_of_words)

        if(dev_err < best_err):
            best_err = dev_err
            best_model = avg_model


        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(best_model), time.time() - t))

if __name__ == "__main__":
     train(sys.argv[1], sys.argv[2], 10)

