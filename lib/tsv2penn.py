#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
import os
import sys
import codecs
import collections
import json
import re
"""
Module for converting tsv output into penn sentence by line format for dynet bilstm tagger

"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

CLASSDIST = {
  "Allgemein":14066,
  "Zugfahrt":3583,
  "Sonstige_Unregelm\u00e4ssigkeiten":3361,
  "Atmosph\u00e4re":2135,
  "Sicherheit":1140,
  "Ticketkauf":1005,
  "Service_und_Kundenbetreuung":670,
  "DB_App_und_Website":570,
  "Informationen":491,
  "Connectivity":441,
  "Auslastung_und_Platzangebot":431,
  "Komfort_und_Ausstattung":214,
  "Gastronomisches_Angebot":131,
  "Barrierefreiheit":103,
  "Image":93,
  "Reisen_mit_Kindern":68,
  "Design":60,
  "Toiletten":56,
  "Gep\u00e4ck":16,
  "O":0
}
VALIDCLASS = set(CLASSDIST)
VALIDCLASS.update({c +p for c in CLASSDIST for p in [':neutral',':positive',':negative']})

def select_most_freq_cat(categories, options):
    """Return most frequent class if any or None"""
    mostfreq = (None,-1)
    for c in categories:
        if c not in VALIDCLASS:
            print >> sys.stderr,'# IGNORED-UNKNOWN-CAT',c
            continue
        if c  in CLASSDIST:
            count = CLASSDIST[c]
        else:
            if ":" in c and not options.polarity:
                _c = re.sub(r':(neutral|positive|negative)','',c)
            count = CLASSDIST[_c]

        if count > mostfreq[1]:
            mostfreq = (c,count)

    if mostfreq[0] == None:
        print >> sys.stderr, 'MOSTFREQ-WAS-NONE',categories
        return 'O'
    else:
        return mostfreq[0]


def get_best_category(categories, options):
    """

    """
    valid_categories = [c for c in categories if c in VALIDCLASS]
    if valid_categories == []:
        print >> sys.stderr, '#NO-VALID-CATEGORY-FOUND',categories
        return 'O'
    if len(valid_categories) == 1 :
        return valid_categories[0]
    else:
        category = select_most_freq_cat(valid_categories, options)
        print >> sys.stderr, 'CHOSEN-CATEGORY', category, 'FROM', categories
        return category



def print_segment(l):
    """Output penn style format: one sent per line, POS/TAG."""

    for (w, t) in l:
        print >> sys.stdout,"%s/%s" % (w, t),
    print

def process_freq(args, options):
    """

    """
    global CLASSDIST, VALIDCLASS
    catcounter = collections.Counter()
    with codecs.open(options.freq_file, 'r', encoding="utf-8") as f:
        # 3 columns: ID, Word, Category list
        for l in f:
            F = l.strip().split('\t')
            for c in F[2].split():
                if not options.polarity:
                    c = re.sub(r':(neutral|positive|negative)', '', c)
                if '#' in c:  # support for old format with full categories
                    maincat, subcat = c.split('#',2)
                    catcounter[maincat] += 1
                else:
                    catcounter[c] += 1
    print >> sys.stderr, catcounter
    CLASSDIST = dict(catcounter)
    VALIDCLASS = set(CLASSDIST)



def process(options=None, args=None):
    """Do the processing"""
    global CLASSDIST, VALIDCLASS
    if options.debug:
        print >>sys.stderr, options

    process_freq(args, options)


    linenb = -1
    segment = []
    characters = collections.Counter()
    with codecs.open(args[0], 'r', encoding="utf-8") as f:
        for l in f:
            F = l.strip().split('\t')
            if len(F) < 3:
                print >> sys.stderr, '#WARNING: Empty line', l
                continue
            current_linenb = int(F[0].split('-')[0])
            if not options.polarity:
                F[2] = re.sub(r':(neutral|positive|negative)', '', F[2])
            categories = F[2].split(' ')



            F[2] = get_best_category(categories, options)

            characters.update(F[1])
            if current_linenb != linenb:
                if segment != []:
                    print_segment(segment)
                segment = [(F[1],F[2])]
            else:
                segment.append((F[1],F[2]))
            linenb = current_linenb

    if segment != []:
        print_segment(segment)

    for k,c in characters.most_common():
        print >> sys.stderr, "%s\t%d" %(k,c)

def main():
    """
    Invoke this module as a script
    """
# global options
    parser = OptionParser(
        usage = '%prog [OPTIONS] TSVFILE',
        version='%prog 0.99', #
        description='Convert ABSA tokenized format into penn',
        epilog='Contact simon.clematide@uzh.ch'
        )
    parser.add_option('-l', '--logfile', dest='logfilename',
                      help='write log to FILE', metavar='FILE')
    parser.add_option('-q', '--quiet',
                      action='store_true', dest='quiet', default=False,
                      help='do not print status messages to stderr')
    parser.add_option('-d', '--debug',
                      action='store_true', dest='debug', default=False,
                      help='print debug information')
    parser.add_option('-t', '--tsv',
                      action='store', dest='tsv', default=False,
                      help='produce tsv data')
    parser.add_option('-P', '--polarity',
                      action='store_true', dest='polarity', default=False,
                      help='keep polarity labels on categories')
    parser.add_option('-F', '--freq_file',
                      action='store', dest='freq_file', default= 'data/train_v1.4.xml.tsv',
                      help='read file for collecting the frequency distribution (TRAIN) for selecting the best category (default=%default)')


    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, "options=",options


    process(options=options,args=args)


if __name__ == '__main__':
    main()
