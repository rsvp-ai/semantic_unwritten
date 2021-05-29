#!/usr/bin/env python3
import nltk
import os
import json


def sentence_split(line):
        sents = nltk.tokenize.sent_tokenize(line)
        rnt = [sent.split() for sent in sents]
        return rnt


