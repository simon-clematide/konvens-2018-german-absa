#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# adaptation of example code for the bilstm POS tagger from DyNet
# https://github.com/clab/dynet_tutorial_examples/blob/master/tutorial_bilstm_tagger.py
# Tested with DyNet 2.0.x

from __future__ import unicode_literals
from collections import Counter, defaultdict
from optparse import OptionParser
from itertools import count
import random
import codecs
import dynet as dy
import numpy as np
import json
import sys

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)



def seqlabel2labelset(seqseq):
    """Return set of labels for each sequence

    Needed for set-of-labels evaluation for Task C
    """
    result = []

    for seq in seqseq:
        labelset = set()
        labelset.update((t for w,t in seq))
        result.append(labelset)
    return result

def eval_labelset(gold, test, nulllabel='O'):
    """Evaluate test while ignoring null labels

    Needed for set-of-labels evaluation for Task C
    """

    tp = fp = fn = 0
    for i,gset in enumerate(gold):
        if nulllabel in gset:
            gset.remove(nulllabel)
        tset = test[i]
        if nulllabel in tset:
            tset.remove(nulllabel)
        tp += len(gset.intersection(tset))
        fp += len(tset.difference(gset))
        fn += len(gset.difference(tset))
    p = compute_precision(tp, fp)
    r = compute_recall(tp,fn)
    result = {
        'TP':tp,
        'FP':fp,
        'FN':fn,
        'P': p,
        'R':r,
        'F':compute_f1(p,r)
    }
    return result


def safe_div(n, dn):
    """
    Deal with zero division
    """
    return float(n) / dn if dn > 0 else float('nan')


def compute_f1(p, r):
    """
    Return F-Score from Precision and Recall
    """
    return safe_div(2 * p * r, (p + r))

def compute_recall(tp, fn):
    return safe_div(tp, tp+fn)

def compute_precision(tp, fp):
    return safe_div(tp, tp+fp)



class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def output_dataset(data, filename='data.tsv',meta={}):
    with codecs.open(filename,'w',encoding='utf-8') as f:
        for i,s in enumerate(data):
            for w,t in s:
                print >> f, "%d\t%s\t%s" %(i,w,t)

    with codecs.open(filename+'.json', 'w', encoding='utf-8') as f:
        json.dump(meta,f,encoding='utf-8')

def read(fname, dataset="TRAIN"):
    """
    Read a POS-tagged file where each line is of the form "word1/tag2 word2/tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    global CLASSDIST, NBR_OF_CLASSES
    if dataset == "TRAIN":
        CLASSDIST = {}
    tags = set()

    with codecs.open(fname,"r",encoding='utf-8') as fh:
        for line in fh:
            l = line.strip().split()
            tags.clear()
            tag_counter = Counter()
            sent = []
            has_dummy_label = False
            has_dummy_token = False
            for x in l:
                if "/" not in x:
                    print >> sys.stderr, 'FORMAT-ERROR', l
                    exit(1)
                w,t = x.rsplit("/",1)
                if w.startswith('http'):
                    w = '__§__'
                if t.startswith('D__'):

                    has_dummy_label = t[3:]
                    t = 'O'
                if w == '__D__':
                    has_dummy_token = True
                if dataset != "TRAIN" and not t in CLASSDIST:

                    print >> sys.stderr, '#WARNING:MAPPED-UNKNOWN-TAG-TO-O', t
                    t = 'O'
                tags.add(t)
                if dataset=="TRAIN":
                    if not t in CLASSDIST:
                        CLASSDIST[t] = 1
                    else:
                        CLASSDIST[t] += 1
                w = w.lower()
                sent.append((w,t))
                if t != 'O':
                    tag_counter[t] += 1

            if has_dummy_token:
                pass
            elif has_dummy_label:
                sent.append(('__D__',has_dummy_label))
                tags.add(has_dummy_label)
            else:
                dummy_label = tag_counter.most_common()[0][0] if tag_counter else 'O'

                sent.append(('__D__', dummy_label))
                tags.add(dummy_label)
            if SKIP_NON_RELEVANT and dataset == "TRAIN" and tags == DUMMYTAGSET:
                print >> sys.stderr, '#SKIPPING IRRELEVANT SENTENCE', l
                continue
            general_stats["SENTLEN_"+dataset][len(sent)] += 1

            yield sent
    if dataset == "TRAIN":
        NBR_OF_CLASSES = len(CLASSDIST)



def word_rep(w, cf_init, cb_init):
    """"Return word representation"""
    if wc[w] > 2:
        w_index = vw.w2i[w]
        return WORDS_LOOKUP[w_index]
    else:
        pad_char = vc.w2i["<*>"]
        char_ids = [pad_char] + [vc.w2i[c] for c in w if c in vc.w2i] + [pad_char]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        #char_embs = [dy.noise(we, 0.05) for we in char_embs]  # optional
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

def build_tagging_graph(words, mode="TRAIN"):
    dy.renew_cg()
    # parameters -> expressions
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()

    cf_init = cFwdRNN.initial_state()
    cb_init = cBwdRNN.initial_state()

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs = [word_rep(w, cf_init, cb_init) for w in words]
#    if mode == 'TRAIN':
#        wembs = [dy.noise(we,0.05) for we in wembs] # optional

    # feed word vectors into biLSTM
    fw_exps = f_init.transduce(wembs)
    bw_exps = b_init.transduce(reversed(wembs))

    # biLSTM states
    bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

    # feed each biLSTM state to an MLP
    exps = []
    for x in bi_exps:

        r_t = O*(dy.tanh(H * x))
        #r_t = O*(dy.rectify(H * x))  # SC RELU
        exps.append(r_t)

    return exps

def sent_loss(words, tags):
    vecs = build_tagging_graph(words)
    errs = []
    for v,t in zip(vecs,tags):
        tid = vt.w2i[t]
        err = dy.pickneglogsoftmax(v, tid)
        errs.append(err)
    return dy.esum(errs)

def tag_sent(words, mode):
    """Return tagged sentence"""
    vecs = build_tagging_graph(words,mode)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    #print >> sys.stderr, probs
    tags = []
    for prb in probs:
        tag = np.argmax(prb)
        tags.append(vt.i2w[tag])
    return zip(words, tags)


def tag_dataset(seqofseq, mode):
    """Return tagged sequence of sequences pairs (Token, Tag) """
    result = []
    for s in seqofseq:
        words = [w for w, t in s]
        system_sent = tag_sent(words,mode)
        result.append(system_sent)
    return result


def eval_dataset(systemset, goldset, label="DEV", options=None):
    """
    Return evaluation data dictionary and emit diagnostics
    """

    confusion = Counter()
    labelset_eval_dict = {'dataset': label }
    good_sent = bad_sent = good = bad = 0.0

    for i,g in enumerate(goldset):
        golds = [t for _, t in g]
        tags = [t for _, t in systemset[i]]
        if tags == golds:
            good_sent += 1
        else:
            bad_sent += 1
        for go, gu in zip(golds, tags):
            confusion[(go, gu)] += 1
            if go == gu:
                good += 1

            else:
                bad += 1


    print >> sys.stderr, 'WORD ACCURACY %sSET: %.4f' % (label,good / (good + bad))
    print >> sys.stderr, 'SENT ACCURACY %sSET: %.4f' % (label,good_sent / (good_sent + bad_sent))
    for ((gold, system), count) in confusion.most_common():
        if gold != system:
            print >> sys.stderr, '# CONFUSION-%s:\t%s\t%s\t%d' % (label,gold, system, count)
        else:
            print >> sys.stderr, '# NOCONFUSION-%s:\t%s\t%s\t%d' % (label,gold, system, count)
    system_sent_labelset = seqlabel2labelset(systemset)
    labelset_eval_dict = eval_labelset(dev_labelset_goldtags, system_sent_labelset)
    print >> sys.stderr, '%s-LABELSET EVAL PRECISION: %.4f' % (label,labelset_eval_dict['P'],)
    print >> sys.stderr, '%s-LABELSET EVAL RECALL: %.4f' % (label,labelset_eval_dict['R'],)
    print >> sys.stderr, '%s-LABELSET EVAL F1: %.4f' % (label,labelset_eval_dict['F'],)
    print >> sys.stderr, '%s-LABELSET EVAL TP: %d' % (label,labelset_eval_dict['TP'],)

    return labelset_eval_dict

def process(options,args):
    """Do the processing..."""
    # sorry for the ugly global variables... it's research code...
    global train, dev, train_labelset_goldtags , dev_labelset_goldtags, test1, test2
    global vw, vt, vc, UNK, nwords, ntags, nchars, model, trainer
    global WORDS_LOOKUP, CHARS_LOOKUP, p_t1, pH,pO, fwdRNN,bwdRNN,cFwdRNN,cBwdRNN

    global DEVSET_EVAL_INTERVAL, SKIP_NON_RELEVANT, PATIENCE, BEST_DEV_F1, CHARACTER_THRESHOLD, BEST_MODEL, DUMMYTAGSET
    global CHAR_EMBEDDING_SIZE, WORD_EMBEDDING_SIZE, HIDDEN_OUTPUT_SIZE, STOP_LABELSET_EVAL_F1, NBR_OF_CLASSES, CLASSDIST
    global general_stats, dev_confusion


    MAX_EPOCHS = options.max_epochs
    DEVSET_EVAL_INTERVAL = 5000
    SKIP_NON_RELEVANT = False

    PATIENCE = 5
    BEST_DEV_F1 = 0.0
    MINIMUM_DEV_F1_SCORE = options.minimum_dev_f1_score
    BEST_DEV_F1_SCORE = 0
    CHARACTER_THRESHOLD = 5
    BEST_MODEL = None
    DUMMYTAGSET = set(['O'])

    CHAR_EMBEDDING_SIZE = 20
    WORD_EMBEDDING_SIZE = 64  # must be even number
    HIDDEN_OUTPUT_SIZE = 64  # must be even number

    STOP_LABELSET_EVAL_F1 = 0.7100

    NBR_OF_CLASSES = 20

    CLASSDIST = {
        "Allgemein": 14066,
        "Zugfahrt": 3583,
        "Sonstige_Unregelm\u00e4ssigkeiten": 3361,
        "Atmosph\u00e4re": 2135,
        "Sicherheit": 1140,
        "Ticketkauf": 1005,
        "Service_und_Kundenbetreuung": 670,
        "DB_App_und_Website": 570,
        "Informationen": 491,
        "Connectivity": 441,
        "Auslastung_und_Platzangebot": 431,
        "Komfort_und_Ausstattung": 214,
        "Gastronomisches_Angebot": 131,
        "Barrierefreiheit": 103,
        "Image": 93,
        "Reisen_mit_Kindern": 68,
        "Design": 60,
        "Toiletten": 56,
        "Gep\u00e4ck": 16
    }

    # some general stats on
    general_stats = defaultdict(Counter)

    dev_confusion = Counter()

    # format of files: each line is "word1/tag2 word2/tag2 ..."
    train_file = options.train_file
    dev_file = options.dev_file
    test1_file = options.test1_file
    test2_file = options.test2_file




    train = list(read(train_file))
    print >> sys.stderr, '#TRAINING SET SEQUENCE SIZE', len(train)
    print >> sys.stderr, '#TRAINING: NUMBER OF CLASSES',len(CLASSDIST)

    train_labelset_goldtags = seqlabel2labelset(train)

    dev = list(read(dev_file, dataset="DEV"))
    print >> sys.stderr, '#DEV SET SEQUENCE SIZE', len(dev)

    dev_labelset_goldtags = seqlabel2labelset(dev)

    output_dataset(dev, filename='gold' + '__' + 'devset' + '.tsv', meta={})

    test1 = list(read(test1_file, dataset="TEST"))
    print >> sys.stderr, '#TEST1 SET SEQUENCE SIZE', len(test1)

    output_dataset(test1, filename='gold' + '__' + 'test1set' + '.tsv', meta={})

    test2 = list(read(test2_file, dataset="TEST"))
    print >> sys.stderr, '#TEST2 SET SEQUENCE SIZE', len(test2)
    global words, tags, wc
    output_dataset(test2, filename='gold' + '__' + 'test2set' + '.tsv', meta={})
    words = []
    tags = []
    chars_counter = Counter()
    wc = Counter()

    for sent in train:
        for w, p in sent:
            words.append(w)
            tags.append(p)
            chars_counter.update(w)
            wc[w] += 1
    words.append("_UNK_")
    words.append("__D__")  # Dummy words for sentences without an explicit cateogorie

    for c in chars_counter.keys():
        if chars_counter[c] < CHARACTER_THRESHOLD:
            del chars_counter[c]

    chars = set(chars_counter)
    chars.add("<*>")

    vw = Vocab.from_corpus([words])
    vt = Vocab.from_corpus([tags])
    vc = Vocab.from_corpus([chars])

    UNK = vw.w2i["_UNK_"]

    nwords = vw.size()
    ntags = vt.size()
    nchars = vc.size()
    print >> sys.stderr, '# NUMBER OF DIFFERENT WORDS', nwords
    print >> sys.stderr, '# NUMBER OF DIFFERENT TAGS', ntags, vt.w2i
    print >> sys.stderr, '# NUMBER OF DIFFERENT CHARACTERS', nchars, vc.w2i

    for statistics in general_stats:
        print >> sys.stderr, '#STATISTICS'
        for k, c in general_stats[statistics].most_common():
            print >> sys.stderr, "%s\t%s\t%d" % (statistics, k, c)  # DyNet Starts

    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    #trainer = dy.AdadeltaTrainer(model)

    WORDS_LOOKUP = model.add_lookup_parameters((nwords, WORD_EMBEDDING_SIZE))
    CHARS_LOOKUP = model.add_lookup_parameters((nchars, CHAR_EMBEDDING_SIZE))
    p_t1 = model.add_lookup_parameters((ntags, NBR_OF_CLASSES))

    # MLP on top of biLSTM outputs 2*HIDDEN_OUTPUT_SIZE -> HIDDEN_OUTPUT_SIZE -> ntags
    pH = model.add_parameters((HIDDEN_OUTPUT_SIZE, HIDDEN_OUTPUT_SIZE * 2))
    pO = model.add_parameters((ntags, HIDDEN_OUTPUT_SIZE))

    # word-level LSTMs
    fwdRNN = dy.CoupledLSTMBuilder(2, WORD_EMBEDDING_SIZE, HIDDEN_OUTPUT_SIZE, model)  # layers, in-dim, out-dim, model
    bwdRNN = dy.CoupledLSTMBuilder(2, WORD_EMBEDDING_SIZE, HIDDEN_OUTPUT_SIZE, model)

    # char-level LSTMs
    cFwdRNN = dy.CoupledLSTMBuilder(2, CHAR_EMBEDDING_SIZE, WORD_EMBEDDING_SIZE / 2, model)
    cBwdRNN = dy.CoupledLSTMBuilder(2, CHAR_EMBEDDING_SIZE, WORD_EMBEDDING_SIZE / 2, model)

    num_tagged = cum_loss = 0
    sample_iter_count = best_sample_iter_count = 0
    for ITER in xrange(MAX_EPOCHS):
        random.shuffle(train)
        for i,s in enumerate(train,1):
            sample_iter_count += 1
            best_sample_iter_count += 1
            if i > 0 and i % (DEVSET_EVAL_INTERVAL / 2) == 0:   # print status
                #trainer.status()
                print >> sys.stderr, 'AVERAGE LOSS: %.4f' % (cum_loss / num_tagged)
                cum_loss = num_tagged = 0

            if i % DEVSET_EVAL_INTERVAL == 0 or i == len(train)-1: # eval on dev
                dev_system = tag_dataset(dev,'DEV')
                dev_labelset_eval_dict = eval_dataset(dev_system, dev, 'DEV')
                dev_labelset_eval_dict['ITERATION'] = i
                if dev_labelset_eval_dict['F'] >  MINIMUM_DEV_F1_SCORE  and  dev_labelset_eval_dict['F'] > BEST_DEV_F1_SCORE:
                    BEST_DEV_F1_SCORE = dev_labelset_eval_dict['F']
                    PATIENCE = 5
                    output_dataset(dev_system, filename=options.model_identifier +'__'+'devset' + '.tsv',meta=dev_labelset_eval_dict)
                    system_test1 = tag_dataset(test1, 'TEST')
                    system_test2 = tag_dataset(test2, 'TEST')
                    output_dataset(system_test1, filename=options.model_identifier +'__'+'testset1_' + '.tsv', meta=dev_labelset_eval_dict)
                    output_dataset(system_test2, filename=options.model_identifier +'__'+'testset2_' + '.tsv', meta=dev_labelset_eval_dict)
                if BEST_DEV_F1_SCORE > 0.0:
                    if PATIENCE > 0:
                        PATIENCE -= 1
                    else:
                        exit(0)
            # train on sent
            words = [w for w,t in s]
            golds = [t for w,t in s]

            loss_exp =  sent_loss(words, golds)
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(golds)
            loss_exp.backward()
            trainer.update()
        print >> sys.stderr, "epoch %r finished" % ITER
        #trainer.update_epoch(1.0)


def main():
    """
    Invoke this module as a script
    """
    global options
    parser = OptionParser(
        usage = '%prog [OPTIONS] [ARGS...]',
        version='%prog 0.99', #
        description='Train model, evaluate on devset  and apply it to test data (early stopping)'
        )
    parser.add_option('-l', '--logfile', dest='logfilename',
                      help='write log to FILE', metavar='FILE')
    parser.add_option('-q', '--quiet',
                      action='store_true', dest='quiet', default=False,
                      help='do not print status messages to stderr')
    parser.add_option('-d', '--debug',
                      action='store_true', dest='debug', default=False,
                      help='print debug information')
    parser.add_option('-M', '--model_identifier',
                      action='store', dest='model_identifier', default='model',
                      help='where to store the results of the model')
    parser.add_option('-S', '--dynet-seed',type=int,default=None,
                      action='store', dest='dynet_seed',
                      help='dynet seed value')
    parser.add_option('', '--dynet-memory',
                      action='store', dest='dynet_memory', type=int,
                      help='dynet memory value MB')
    parser.add_option('-T', '--train_file',
                      action='store', dest='train_file', default=None,
                      help='Training file (penn format) %default')
    parser.add_option('-D', '--dev_file',
                      action='store', dest='dev_file', default=None,
                      help='development file (penn format) %default')
    parser.add_option('-E', '--test1_file',
                      action='store', dest='test1_file', default=None,
                      help='test1 file (penn format) %default')
    parser.add_option('-F', '--test2_file',
                      action='store', dest='test2_file', default=None,
                      help='test2 file (penn format) %default')

    parser.add_option('-f', '--minimum_dev_f1_score',
                      action='store', dest='minimum_dev_f1_score', default=0.7150, type=float,
                      help='minimal labelset dev F1 score %default')
    parser.add_option('-e', '--max_epochs',
                      action='store', dest='max_epochs', default=100,
                      help='maximum number of epochs %default')


    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, sys.argv
        print >> sys.stderr, "options=",options

    random.seed(a=options.dynet_seed)
    process(options=options,args=args)


if __name__ == '__main__':
    main()
