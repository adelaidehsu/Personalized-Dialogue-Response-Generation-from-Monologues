# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import random

from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

# Special vocabulary symbols
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\":;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
    #Very basic tokenizer: split the sentence into a list of tokens.
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def maybe_split(data_path):
    data_dir, file_name = data_path.split('/')
    train_path = os.path.join(data_dir, 'train_' + file_name)
    dev_path = os.path.join(data_dir, 'dev_' + file_name)
    if not (gfile.Exists(train_path)) and ( not gfile.Exists(dev_path) ):
        if not (gfile.Exists(data_path)):
            raise ValueError("Source file %s not found.", data_path)
        raise ValueError("Train file or development file not found.")
    return (train_path, dev_path)

def create_vocabulary(vocabulary_path, train_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d from %s" % (counter, data_path))
        halfmax = (int)(max_vocabulary_size/2) 
        tmp_list = sorted(vocab, key=vocab.get, reverse=True)[:halfmax]
        for key in tmp_list:
            vocab[key] += 1000

        print("Creating vocabulary %s from data %s" % (vocabulary_path, train_path))
        with gfile.GFile(train_path, mode="rb") as f:
            counter = 0
            for line in f:
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d from %s" % (counter, train_path))
    
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().encode('utf-8') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, target_voc_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, revoc = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                with gfile.GFile(target_voc_path, mode="w") as voc_file:
                    counter = 0
                    for line in data_file:
                        token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                tokenizer, normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                        voc_file.write(" ".join([revoc[tok].decode('utf-8') for tok in token_ids]) + "\n")
                        counter += 1
                        if counter % 100000 == 0:
                            print("tokenizing line %d" % counter)


def create_feature(feature_path, feature_size, feature, tokenizer=None, normalize_digits=True):
    voc_feature = []
    voc_feature.append({})
    with gfile.GFile(feature_path, mode="rb") as f:
        counter = 0
        for line in f:
            line = tf.compat.as_bytes(line)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for id, w in enumerate(tokens):
                word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                if word in voc_feature[id]:
                    continue
                else:
                    voc_feature[id][word] = len(voc_feature[id])
                    print(word)
            counter += 1
            if counter % 100000 == 0:
                print("processing line %d" % counter)

    with gfile.GFile(feature_path, mode="rb") as f:
        for line in f:
            line = tf.compat.as_bytes(line)
            tmp = [0 for _ in range(feature_size)]
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for id, w in enumerate(tokens):
                word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                number = voc_feature[id][word]
                tmp[number] = 1  
            feature.append(tmp)


def prepare_data(feature_path, feature_size, data_dir, data_path, train_path, vocabulary_size, tokenizer=None):
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    create_vocabulary(vocab_path, train_path, data_path, vocabulary_size, tokenizer, normalize_digits=False)

    # create feature list for the training corpus
    feature = []
    create_feature(feature_path, feature_size, feature)

    # Create token ids for the training data
    # Create token ids for the development data.
    train_ids_path = train_path + (".ids%d" % vocabulary_size)
    train_voc_path = train_path + (".voc%d" % vocabulary_size)
    data_to_token_ids(train_path, train_ids_path, train_voc_path, vocab_path, tokenizer, normalize_digits=False)
    
    data_ids_path = data_path + (".ids%d" % vocabulary_size)
    data_voc_path = data_path + (".voc%d" % vocabulary_size)
    data_to_token_ids(data_path, data_ids_path, data_voc_path, vocab_path, tokenizer, normalize_digits=False)
    
    return (feature, data_ids_path, train_ids_path, data_voc_path, train_voc_path)


def combine_corpus(input_file1, input_file2, vocab_path, output_file, limit):
    if not gfile.Exists(output_file):
        with open(input_file1, 'r') as f1:
            with open(input_file2, mode='r') as f2:
                with open(vocab_path, mode='r') as v:
                    with open(output_file, 'w') as f3:
                        for line in v.readlines():
                            f3.write(line)
                        for line in f1.readlines()+f2.readlines():
                            linelist = line.split()
                            if len(linelist) < limit:
                                linelist.insert(0, '_GO')
                                linelist.append('_EOS')
                                for i in range(30 - len(linelist)):
                                    linelist.append('_PAD')
                                output_line = ' '.join(linelist)
                                f3.write(output_line)
                                f3.write('\n')
                            else:
                                f3.write(line)

def training_data_grouping(train_path, feature_path, feature_size):
    feature = []
    group_data = {}
    id_feature_map, id_line_map = {}, {}
    with open(feature_path, 'r')as f:
        for line in f:
            if line.strip() not in feature:
                feature.append(line.strip())
            if len(feature)==feature_size:
                break
    #cluster data by speakers        
    for i, x in enumerate(feature):
            group_data[x] = []
            id_feature_map[i] = x
    with open(train_path, 'r') as train_file:
        idx = 0
        for line in train_file:
            id_line_map[idx] = line.strip()
            idx += 1
    with open(feature_path, 'r')as f:
        f_idx = 0
        for name in f:
            group_data[name.strip()].append(id_line_map[f_idx])
            f_idx+=1
    return group_data, id_feature_map