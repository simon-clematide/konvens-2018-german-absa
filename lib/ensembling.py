#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division
from optparse import OptionParser
import os
import sys
import codecs
from collections import Counter
"""
Module for ensembling single model output with special treatment of 0 tags

Allows the specification of a percentage voting threshold with regard to non-0 tags.

Input is the (horizontal) tabseparated concatenation of N individual systems results (as can be produced
by a command as `paste *__testset1_.tsv`):
  - DOCID_SYS1 TAB TOKEN_SYS1 TAB CLASS_SYS1 TAB ... DOCID_SYSN TAB TOKEN_SYSN TAB CLASS_SYSN NL

"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

def voting(seq, options):
    """
    Return the ensemble decision given several modelspecific labelings
    """

    c = Counter(seq)

    if options.mode == 'majority':
        # majority class mode (ignoring any difference between O tags and non-O tags)
        return c.most_common()[0][0]
    elif options.mode.isdigit():
        # percentage mode
        percentage = float(options.mode)/100
        N = sum(c.values())
        result = c.most_common()[0][0]
        if result == 'O':
            for tag,count in c.most_common()[1:]:
                if count/N > percentage:
                    result = tag
                    if options.debug:
                        print >> sys.stderr, '#TAG-CHOSEN-WITH-PERCENTAGE',tag, count, N, count/N
                    break
        return result


def process(options=None,args=None):
    """
    Do the processing

    """
    if options.debug:
        print >>sys.stderr, options
    maxcol = None
    start = options.column
    stride = start + 1

    for l in sys.stdin:
        F = l.rstrip().split('\t')
        if maxcol is None:
            maxcol = len(F) + 1
            print >> sys.stderr, '#INFO-INPUT-COLUMN-COUNT', len(F)
        label = voting(F[start:maxcol:stride],options)
        print >> sys.stdout, "%s\t%s\t%s" % (F[0],F[1],label)









def main():
    """
    Invoke this module as a script
    """

    parser = OptionParser(
        usage = '%prog [OPTIONS] < INPUT',
        description = """Input is the (horizontal) tabseparated concatenation of N
        individual systems results (as can be produced
        by a command as `paste *__testset1_.tsv`):""",
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
    parser.add_option('-C', '--column',
                      action='store', dest='column', default=2,
                      help='columns (zero-based) containing the label (default=%default)')
    parser.add_option('-m', '--mode',
                      action='store', dest='mode', default='majority',
                      help='ensembling schema: majority vote, NN (percent); (default=%default)')

    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, "options=",options


    process(options=options,args=args)


if __name__ == '__main__':
    main()
