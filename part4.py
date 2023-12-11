#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings('ignore')


def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def word_filterer(trainfile, min_count):
    word_counts = {}
    for label, words in read_from(trainfile):
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    return {word for word, count in word_counts.items() if count > min_count}

def make_vector(words, set_of_words):
    vector = np.zeros(len(set_of_words) + 1)  # +1 for bias term
    for word in words:
        if word in set_of_words:
            vector[list(set_of_words).index(word)] += 1
    return vector

def train(trainfile, devfile, epochs=5, min_count=1):
    t = time.time()
    set_of_words = word_filterer(trainfile, min_count)

    X_train = []
    y_train = []
    for label, words in read_from(trainfile):
        X_train.append(make_vector(words, set_of_words))
        y_train.append(label)

    X_dev = []
    y_dev = []
    for label, words in read_from(devfile):
        X_dev.append(make_vector(words, set_of_words))
        y_dev.append(label)

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_dev_np = np.array(X_dev)
    y_dev_np = np.array(y_dev)

    best_model = None
    best_dev_err = 1.0

    clf = LogisticRegression()
    clf.fit(X_train_np, y_train_np)

    dev_pred = clf.predict(X_dev_np)
    dev_err = 1.0 - accuracy_score(y_dev_np, dev_pred)

    if dev_err < best_dev_err:
        best_dev_err = dev_err
        best_model = clf

    print(f"Best dev error: {best_dev_err * 100:.1f}% in {time.time() - t:.1f} secs")

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], 10, min_count=2)

