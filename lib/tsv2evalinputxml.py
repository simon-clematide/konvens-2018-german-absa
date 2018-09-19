#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
import os
import sys
import codecs
from collections import defaultdict
from lxml import etree
"""
Module for converting TSV system output into the official ABSA XML input format

We just take the gold XML, remove the gold information and input the system's output.
"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

def get_goldfile(filename):
    """Return DOM tree of file"""
    dom = etree.parse(filename)
    return dom

def get_offsetfile(filename):
    """
    Return list of offset information for all tokens
0-0-7	screams	O
0-8-9	@	O
0-10-18	deutsche	O
0-19-23	bahn	O
0-23-24	.	O
0-0-0	__D__	Allgemein:neutral
1-0-6	Studie	O
1-6-7	:	O
1-8-14	Messen	O
1-15-22	bringen	O
1-23-29	Berlin	O
1-30-39	Millionen	O
1-40-46	Studie	O
1-46-47	:	O
1-48-54	Messen	O
1-55-62	bringen	O
1-63-69	Berlin	O
1-70-79	Millionen	O
1-80-86	Berlin	O
1-87-88	(	O
1-88-94	dpa/bb	O
1-94-95	)	O
1-96-97	-	O
1-98-103	Jeder	O
1-104-108	Euro	O
1-109-121	Messe-Umsatz	O
1-122-128	bringt	O
1-129-133	5,10	O
1-134-138	Euro	O
1-139-150	zusätzliche	O
1-151-160	Kaufkraft	O
1-161-165	nach	O
1-166-172	Berlin	O
1-172-173	.	O
1-174-177	Das	O
1-178-181	ist	O
1-182-185	das	O
1-186-194	Ergebnis	O
1-195-200	einer	O
1-201-207	Studie	O
1-208-211	der	O
1-212-228	Investitionsbank	O
1-229-235	Berlin	O
1-236-240	über	O
1-241-256	wirtschaftliche	O
1-257-264	Effekte	O
1-265-268	der	O
1-269-275	Messen	O
1-276-278	in	O
1-279-285	Berlin	O
1-285-286	,	O
1-287-289	di	O
1-0-0	__D__	O

    """
    result = []
    with codecs.open(filename, 'r', encoding="utf-8") as f:
        for i, l in enumerate(f):
            d = l.rstrip().split('\t')
            d[0] = d[0].split('-')
            d[0] = [int(d[0][0]),int(d[0][1]),int(d[0][2])]
            result.append(d[0:2])
    print >> sys.stderr, '#INFO: GOLD SENTENCES READ IN: ', len(result)
    return result

def strip_annotations(dom):
    """
    """
    for o in dom.iterfind('//Opinion'):

        o.getparent().remove(o)
    return dom

def read_system(options):
    """
    read system output

    687	brand	O
687	in	O
687	einem	O
687	ice	O
687	wird	O
687	ein	O
687	falscher	O
687	__d__	O
688	re	O
688	:	O
688	geschwister&apos;liebe	O
688	-	O
688	geschwister&apos;kriese	O
688	geschwister&apos;streit	O
688	-	O

    """
    result = []
    for l in sys.stdin:
        d = l.rstrip().split('\t')
        d[0] = int(d[0])
        result.append(d)
    lasttag = 'O'
    intag = False
    for i,r in enumerate(result):
        currenttag = r[2]
        if currenttag == 'O':
            r.append('O')
            intag = False
        elif intag:
            if lasttag == currenttag:
                r.append('I')  # default
            else:
                r.append('B')
        else:
            r.append('B')
            intag = True

        lasttag = currenttag
    if result[-1][2] != 'O':
        result[-1][3] = 'E'
    return result



def docid2anno(offsets, system):
    annos = defaultdict(list)
    intag = False

    lastinset = None
    lastoffset = None
    lastcat = None
    lastcontent = None
    lastpolarity = None
    lastdoc = None
    currentdoc = None
    for i,s in enumerate(system):
        currentpolarity = 'neutral' # default
        currentcat = s[2]
        if ":" in currentcat:
            cc = currentcat.split(':')
            currentpolarity = cc[1]
            currentcat = cc[0]
        currentcontent = s[1]
        currentdoc = s[0]

        if s[3] == 'B':
            if intag:
                annos[lastdoc].append({'inset':lastinset,'offset':lastoffset,'cat':lastcat,'target':lastcontent,'polarity':lastpolarity})


            lastinset = offsets[i][0][1]
            lastoffset = offsets[i][0][2]
            lastcontent = currentcontent
            lastdoc = currentdoc
            lastcat = currentcat
            lastpolarity = currentpolarity
            intag = True
        elif s[3] == 'I':
            lastoffset = offsets[i][0][2]
            lastcontent += currentcontent
        else: # 'O'
            if intag:
                annos[lastdoc].append({'inset': lastinset, 'offset': lastoffset, 'cat': lastcat,'target':lastcontent,'polarity':lastpolarity})
                lastcat = currentcat
                lastpolarity = currentpolarity
                lastcontent = currentcontent
                lastdoc = currentdoc

            intag = False
    if lastcat != 'O':
        annos[currentdoc].append({'inset': lastinset, 'offset': lastoffset, 'cat': lastcat, 'target':lastcontent,'polarity':lastpolarity})
    return annos


def update_dom(dom, annos):
    """
     <Opinion category="Atmosphäre#Haupt" from="2" to="41" target="Studentinnen in der Bahn erörtern grade" polarity="neutral"/>
    """
    for i,doc in enumerate(dom.iter('Document')):
        if doc.findall('Opinions') == []:
            doc.insert(0,etree.fromstring('<Opinions/>'))
        ops = doc.find('Opinions')
        if i in annos:
            for anno in annos[i]:
                if anno['offset'] == 0:
                    anno['target'] = 'NULL'
                opinion = etree.fromstring('<Opinion category="{cat}" from="{inset}" to="{offset}" target="{target}" polarity="{polarity}" />'.format(**anno))
                ops.insert(0,opinion)

def process(options=None,args=None):
    """
    Do the processing

    http://twitter.com/pouvraire/statuses/697803662270267392	screams @ deutsche bahn.	true	neutral	Allgemein:neutral
    URL                                                          TEXT                       RELE    SENTI    CAT:SENTI
    """
    goldxml = get_goldfile(options.goldfile)

    goldxml = strip_annotations(goldxml)
    #print>> sys.stdout, etree.tostring(goldxml)
    offsets = get_offsetfile(options.offsetfile)
    system = read_system(options)
    annos = docid2anno(offsets, system)
    for i,a in enumerate(annos):
        print >> sys.stderr, i,a,annos[a]
    update_dom(goldxml,annos)
    print etree.tostring(goldxml)

def main():
    """
    Invoke this module as a script
    """

    parser = OptionParser(
        usage = '%prog [OPTIONS] [ARGS...] < TSVSYSTEMOUTPUT',
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
                      help='goldfile (xml)')
    parser.add_option('-o', '--offsetfile',
                      action='store', dest='offsetfile', default=None,
                      help='offsetfile (tsv) (default=%default)')

    parser.add_option('-P', '--polarity',
                      action='store_true', dest='polarity', default=False,
                      help='label contain polarity (default=%default)')

    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, "options=",options

    if not options.goldfile:
        print >> sys.stderr, 'ERROR: goldstandard file needed'
        exit(1)
    process(options=options,args=args)


if __name__ == '__main__':
    main()
