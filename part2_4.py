#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector
from collections import Counter

LEARNING_RATE = 0.1

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    v['<bias>'] = 1  
    for word in words:
        v[word] += 1
    return v

def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now

def top_features(model, n=20):
    feature_weights = model.items()
    sorted_weights = sorted(feature_weights, key=lambda x: x[1], reverse=True)

    top_positive_features = sorted_weights[:n]  
    top_negative_features = sorted_weights[-n:]  

    return top_positive_features, top_negative_features



def averaged_perceptron(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = 1.
    best_model = svector()
   
    avg_model = svector()

    model = svector()

    found_misclassified_positive = 0
    found_misclassified_negative = 0

    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
            avg_model += Counter(model)

        dev_err = test(devfile, avg_model)

        if(dev_err < best_err):
            best_err = dev_err
            best_model = avg_model

        #print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    #print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(avg_model), time.time() - t)

            if found_misclassified_positive >= 5 and found_misclassified_negative >= 5:
                break 

    get_top_misclassified(sys.argv[2], best_model,found_misclassified_positive, found_misclassified_negative)

def get_top_misclassified(devfile, best_model,found_misclassified_positive, found_misclassified_negative):
    predictions = []

    for i, (label, words) in enumerate(read_from(devfile), 1):
        sent = make_vector(words)
        prediction = best_model.dot(sent)
        predictions.append((i, prediction,label, words))

    predictions = sorted(predictions, key=lambda x: x[1])

    misclassified_positive = []
    misclassified_negative = []


    for i, pred_i, true_label, words in predictions:
        if true_label == 1 and found_misclassified_negative < 5 and pred_i < 0:
            misclassified_negative.append((i, pred_i, words))
            found_misclassified_negative += 1

    for i, pred_i, true_label, words in reversed(predictions):
        if true_label == -1 and found_misclassified_positive < 5 and pred_i > 0:
            misclassified_positive.append((i, pred_i, words))
            found_misclassified_positive += 1
       
        if found_misclassified_positive >= 5 and found_misclassified_negative >= 5:
            break


    print("5 negative examples misclassified as positive:")
    for i,pred_i, words in misclassified_positive:
        print(f"Example {i}, Prediction: {pred_i}, Words:{' '.join(words)}")

    print("\n5 positive examples misclassified as negative:")
    for i,pred_i, words in misclassified_negative:
        print(f"Example {i}, Prediction: {pred_i}, Words: {' '.join(words)}")



if __name__ == "__main__":
    averaged_perceptron(sys.argv[1], sys.argv[2], 10)
    
