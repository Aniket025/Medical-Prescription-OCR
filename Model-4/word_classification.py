# -*- coding: UTF-8 -*

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
import math
from collections import Counter
import unidecode
from abc import ABC, abstractmethod

# Import Widgets
from ipywidgets import Button, Text, HBox, VBox
from IPython.display import display, clear_output

# Import costume functions, corresponding to notebooks
from ocr import charSeg
from ocr.normalization import letterNorm, imageNorm
# from ocr import charSeg
# Helpers
from ocr.helpers import implt, resize, extendImg
from ocr.datahelpers import loadWordsData, idx2char
from ocr.tfhelpers import Graph
from ocr.viz import printProgressBar

LANG = 'en'

charClass_1 = Graph('models/char-clas/' + LANG + '/CharClassifier')
# charClass_2 = Graph('models/char-clas/' + LANG + '/Bi-RNN/model_2', 'prediction')
# charClass_3 = Graph('models/char-clas/' + LANG + '/Bi-RNN/model_1', 'prediction')

wordClass = Graph('models/word-clas/' + LANG + '/WordClassifier2', 'prediction_infer')
wordClass2 = Graph('models/word-clas/' + LANG + '/SeqRNN/Classifier3', 'word_prediction') # None
wordClass3 = Graph('models/word-clas/' + LANG + '/CTC/Classifier2', 'word_prediction')

images, labels = loadWordsData('data/test_words/' + LANG + '_raw', loadGaplines=False)

for i in range(len(images)):
    printProgressBar(i, len(images))
    images[i] = imageNorm(
        cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB),
        60,
        border=False,
        tilt=True,
        hystNorm=True)

if LANG == 'en':
    for i in range(len(labels)):
        labels[i] = unidecode.unidecode(labels[i])
print()
print('Number of chars:', sum(len(l) for l in labels))

# Load Words
WORDS = {}
with open('data/' + LANG + '_50k.txt') as f:
    for line in f:
        if LANG == 'en':
            WORDS[unidecode.unidecode(line.split(" ")[0])] = int(line.split(" ")[1])
        else:
            WORDS[line.split(" ")[0]] = int(line.split(" ")[1])
WORDS = Counter(WORDS)

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    if word in WORDS:
        return word
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."

    if LANG == 'cz':
        letters = 'aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž'
    else:
        letters = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

class Cycler(ABC):
    """ Abstract cycler class """
    def __init__(self, images, labels, charClass, stats="NO Stats Provided", slider=(60, 15), ctc=False, seq2seq=False, charRNN=False):
        self.images = images
        self.labels = labels
        self.charClass = charClass
        self.slider = slider
        self.totalChars = sum([len(l) for l in labels])
        self.ctc = ctc
        self.seq2seq = seq2seq
        self.charRNN = charRNN
        self.stats = stats

        self.evaluate()

    @abstractmethod
    def recogniseWord(self, img):
        pass

    def countCorrect(self, pred, label, lower=False):
        correct = 0
        for i in range(min(len(pred), len(label))):
            if ((not lower and pred[i] == label[i])
                 or (lower and pred[i] == label.lower()[i])):
                correct += 1

        return correct


    def evaluate(self):
        """ Evaluate accuracy of the word classification """
        print()
        print("STATS:", self.stats)
        print(self.labels[1], ':', self.recogniseWord(self.images[1]))
        start_time = time.time()
        correctLetters = 0
        correctWords = 0
        correctWordsCorrection = 0
        correctLettersCorrection = 0
        for i in range(len(self.images)):
            word = self.recogniseWord(self.images[i])
            correctLetters += self.countCorrect(word,
                                         self.labels[i])
            # Correction works only for lower letters
            correctLettersCorrection += self.countCorrect(correction(word.lower()),self.labels[i],lower=True)
            # Words accuracy
            if word == self.labels[i]:
                correctWords += 1
            if correction(word.lower()) == self.labels[i].lower():
                correctWordsCorrection += 1

        print("Correct/Total: %s / %s" % (correctLetters, self.totalChars))
        print("Letter Accuracy: %s %%" % round(correctLetters/self.totalChars * 100, 4))
        print("Letter Accuracy with Correction: %s %%" % round(correctLettersCorrection/self.totalChars * 100, 4))
        print("Word Accuracy: %s %%" % round(correctWords/len(self.images) * 100, 4))
        print("Word Accuracy with Correction: %s %%" % round(correctWordsCorrection/len(self.images) * 100, 4))
        print("--- %s seconds ---" % round(time.time() - start_time, 2))

class CharCycler(Cycler):
    """ Cycle through the words and recognise them """
    def recogniseWord(self, img):
        img = cv2.copyMakeBorder(img, 0, 0, 30, 30, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        gaps = charSeg.segmentation(img, RNN=True)

        chars = []
        for i in range(len(gaps)-1):
            char = img[:, gaps[i]:gaps[i+1]]
            # TODO None type error after treshold
            char, dim = letterNorm(char, is_thresh=True, dim=True)
            # TODO Test different values
            if dim[0] > 4 and dim[1] > 4:
                chars.append(char.flatten())

        chars = np.array(chars)
        word = ''
        if len(chars) != 0:
            if self.charRNN:
                pred = self.charClass.eval_feed({'inputs:0': [chars], 'length:0': [len(chars)], 'keep_prob:0': 1})[0]
            else:
                pred = self.charClass.run(chars)

            for c in pred:
                # word += CHARS[charIdx]
                word += idx2char(c)
        return word

# Class cycling through words

#WordCycler(images, labels, wordClass, stats='Seq2Seq', slider=(60, 2), seq2seq=True)
#WordCycler(images, labels, wordClass2, stats='Seq2Seq2CNN', slider=(60, 2))
#WordCycler(images, labels, wordClass3, stats='CTC', slider=(60, 2), ctc=True)
CharCycler(images, labels, charClass_1, stats='Bi-RNN and CNN', charRNN=False)
