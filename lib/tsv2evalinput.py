#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
import os
import sys
import codecs

"""
Module for creating input for the official ABSA evalution script

Creates TSV input

"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

def get_goldfile(filename):
    result = []
    with codecs.open(filename, 'r', encoding="utf-8") as f:
        for i, l in enumerate(f):
            d = l.rstrip().split('\t')
            result.append(d[0])
    print >> sys.stderr, '#INFO: GOLD SENTENCES READ IN: ', len(result)
    return result


def process(options=None,args=None):
    """
    Do the processing

    http://twitter.com/pouvraire/statuses/697803662270267392	screams @ deutsche bahn.	true	neutral	Allgemein:neutral
    URL                                                          TEXT                       RELE    SENTI    CAT:SENTI
    """
    goldlist = get_goldfile(options.goldfile)
    if options.debug:
        print >>sys.stderr, options
    currentid = None
    currentsent = []
    labels = list()
    i = -1
    for l in sys.stdin:
        F = l.rstrip().split('\t')
        if currentid is None:
            currentid = F[0]

        if currentid != F[0]:
            i += 1
            if not options.polarity:
                print >> sys.stdout, '%s\t%s\ttrue\tnegative\t%s\t' %(goldlist[i]," ".join(currentsent)," ".join((l+":negative " for l in labels)))
            else:
                print >> sys.stdout, '%s\t%s\ttrue\tnegative\t%s\t' % (
                goldlist[i], " ".join(currentsent), " ".join(labels))
            labels= list()
            currentsent = []
            currentid = F[0]

        else:

            currentsent.append(F[1])
            if F[2] != 'O':
                # Currently just count once!
                if F[2].startswith('Allgemein') and labels.count(F[2]) < 1:
                    labels.append(F[2])
                elif labels.count(F[2]) < 1:
                    labels.append(F[2])

    if currentsent != []:
        i += 1
        if not options.polarity:
            print >> sys.stdout, '%s\t%s\ttrue\tnegative\t%s\t' % (
                goldlist[i], " ".join(currentsent), " ".join((l + ":negative " for l in labels)))
        else:
            print >> sys.stdout, '%s\t%s\ttrue\tnegative\t%s\t' % (
                goldlist[i], " ".join(currentsent), " ".join(labels))


def main():
    """
    Invoke this module as a script
    """
# global options
    parser = OptionParser(
        usage = '%prog [OPTIONS] [ARGS...]',
        version='%prog 0.99'
        )
    parser.add_option('-l', '--logfile', dest='logfilename',
                      help='write log to FILE', metavar='FILE')
    parser.add_option('-q', '--quiet',
                      action='store_true', dest='quiet', default=False,
                      help='do not print status messages to stderr')
    parser.add_option('-d', '--debug',
                      action='store_true', dest='debug', default=False,
                      help='print debug information')
    parser.add_option('-m', '--mode',
                      action='store', dest='mode', default='tsv',
                      help='evaluation mode file (tsv)')
    parser.add_option('-g', '--goldfile',
                      action='store', dest='goldfile', default=None,
                      help='goldfile (tsv)')
    parser.add_option('-P', '--polarity',
                      action='store_true', dest='polarity', default=False,
                      help='label contain polarity %default')

    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, "options=",options

    if not options.goldfile:
        print >> sys.stderr, 'ERROR: goldstandard file needed'
        exit(1)
    process(options=options,args=args)


if __name__ == '__main__':
    main()
