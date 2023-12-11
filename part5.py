#!/usr/bin/env python3

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
    vector = np.zeros(len(set_of_words) + 1)  
    for word in words:
        if word in set_of_words:
            vector[list(set_of_words).index(word)] += 1
    return vector

def replace_question_mark(predictions, testfile):
    with open(testfile, 'r') as file:
        line = file.readlines()

    for i in range(len(line)):
        if line[i].startswith('?'):
            if predictions[i]==1:
                sign = '+'
            else:
                sign = '-'
            line[i] = sign + line[i][1:]
    
    with open("test.txt.predicted", 'w') as file:
        file.writelines(line)

def train(trainfile, testfile, min_count=1):
    t = time.time()
    set_of_words = word_filterer(trainfile, min_count)

    X_train = []
    y_train = []
    for label, words in read_from(trainfile):
        X_train.append(make_vector(words, set_of_words))
        y_train.append(label)
    X_test = []
    for label, words in read_from(testfile):
        X_test.append(make_vector(words, set_of_words))

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_np, y_train_np)

    y_pred = clf.predict(X_test_np)
    
    replace_question_mark(y_pred, testfile)
    print(f"Time : {time.time() - t:.1f} secs")

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], min_count=1)
