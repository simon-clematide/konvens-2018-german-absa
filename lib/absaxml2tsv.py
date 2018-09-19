#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
import os
import sys
import codecs
import lxml
from lxml.etree import tostring, Element

from lxml import etree

import re
from xmljson import BadgerFish
from collections import OrderedDict, defaultdict, Counter
bf = BadgerFish(dict_type=dict)
"""
Module for converting the ABSA XML data into a tokenized vertical format



"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


# Tokenizer pattern
pattern = r'''(?x)(?u)           # set flag (?x) to allow verbose regexps

     (?:z\.B\.|bzw\.|usw\.)  # known abbreviations ...
   | (?:\w\.)+               # abbreviations, e.g. U.S.A. or ordinals
   #| [#@,;]\W
   | (?:\.\.+|--+)         # ellipsis, ASCII long hyphens
   | \$?\d+(?:[.,:]\d+)*[%€]? # currency/percentages, $12.40, 82% 23€
   | (?:\w+(?:(?:\S|:/?/?)?\w+)?)+(?:['\S-]\w+)*         # words with optional internal hyphens or apostrophs
   | [„.,;?!'"»«[\]()]+        # punctuation
   | [#@]\w+
   | \S+                     # catch-all for non-layout characters
   '''

def split_span(s, offset=0, pattern=pattern):
    """
    Return split text with character offsets

    https://stackoverflow.com/questions/9518806/how-to-split-a-string-on-whitespace-and-retain-offsets-and-lengths-of-words
    """
    for match in re.finditer(pattern, s):
        span = match.span()
        yield match.group(0), span[0]+offset, span[1]+offset



def docjs2penn(d,docseq, options):
    """
    Return tab-separated verticalized format with CO labeling (category C and out O)

    - First column: DOCSEQNBR-FROM-TO: Sequential number of documents (starting from 0) and the offset From and To
    - Second column: word
    - Third column: label (either category or category:sentiment depending on the options)

        <Document id="http://www.facebook.com/2222310914/posts/10153796475265915?comment\_id=10153796834815915#2222310914\_10153796475265915\_10153796834815915">
        <Opinions>
            <Opinion category="Allgemein#Haupt" from="0" to="0" target="NULL" polarity="neutral"/>
        </Opinions>
        <relevance>true</relevance>
        <sentiment>neutral</sentiment>
        <text>Re: Sylt Ich fahre immer mit der Bahn auf die Insel und miete mir dort ein Auto</text>
    </Document>


{
  'Document':{
    'relevance':{
      '$':True
    },
    'Opinions':{
      'Opinion':{
        '@to':0,
        '@category':'Allgemein#Haupt',
        '@target':'NULL',
        '@from':0,
        '@polarity':'neutral'
      }
    },
    '@id':'http://www.facebook.com/2222310914/posts/10153796475265915?comment\\_id=10153796834815915#2222310914\\_10153796475265915\\_10153796834815915',
    'sentiment':{
      '$':'neutral'
    },
    'text':{
      '$':'Re: Sylt Ich fahre immer mit der Bahn auf die Insel und miete mir dort ein Auto'
    }
  }
}


   <Document id="http://news-koblenz.de">
        <Opinions>
            <Opinion category="Sonstige_Unregelmässigkeiten#Haupt" from="119" from2="0" to="134" to2="0" target="Einschränkungen" polarity="negative"/>
            <Opinion category="Sonstige_Unregelmässigkeiten#Haupt" from="335" from2="0" to="352" to2="0" target="ohne Verkehrshalt" polarity="negative"/>
            <Opinion category="Sonstige_Unregelmässigkeiten#Haupt" from="377" from2="0" to="392" to2="0" target="Einschränkungen" polarity="negative"/>
            <Opinion category="Sonstige_Unregelmässigkeiten#Haupt" from="842" from2="0" to="857" to2="0" target="Einschränkungen" polarity="negative"/>
            <Opinion category="Sonstige_Unregelmässigkeiten#Haupt" from="1011" from2="0" to="1026" to2="0" target="Einschränkungen" polarity="negative"/>
            <Opinion category="Sonstige_Unregelmässigkeiten#Haupt" from="1174" from2="0" to="1182" to2="0" target="gesperrt" polarity="negative"/>
        </Opinions>
        <relevance>true</relevance>
        <sentiment>negative</sentiment>
        <text>14.11.2016 – 13:53 Berlin-Mitte (ots) - Anlässlich des Besuches des amerikanischen Präsidenten Barack Obama wird es zu Einschränkungen im S-Bahn-Verkehr kommen. Von Mittwoch, 16. November 2016, ein Uhr bis Freitag, 18. November 2016, ca. 14 Uhr wird der S-Bahnhof Brandenburger Tor gesperrt. Züge der S-Bahn passieren den Bahnhof dann ohne Verkehrshalt. Bitte beachten Sie die Einschränkungen in Ihren Reisplanungen. Wir bitten um Ihr V erständnis. Rückfragen bitte an: Bundespolizeidirektion Berlin - Pressestelle - Schnellerstraße 139 A/ 140 12439 Berlin Telefon: 030 91144 4050 Mobil: 0171 7617149 Fax: 030 91144-4049 E-Mail: presse.berlin@polizei.bund.de http://www.bundespolizei.de Original-Content von: Bundespolizeidirektion Berlin, übermittelt durch news aktuell Der Beitrag BPOLD-B: Bundespolizei informiert über sicherheitsbedingte Einschränkungen erschien zuerst auf NEPOLI NEWS . 14.11.2016 – 13:53 Berlin-Mitte (ots) - Anlässlich des Besuches des amerikanischen Präsidenten Barack Obama wird es zu Einschränkungen im S-Bahn-Verkehr kommen. Von Mittwoch, 16. November 2016, ein Uhr bis Freitag, 18. November 2016, ca. 14 Uhr wird der S-Bahnhof Brandenburger Tor gesperrt. Züge der S-Bahn passieren den Bahnhof dann ohne Verkehrsha</text>
    </Document>

    """

    text = d['Document']['text']['$']
    tokenized = list(split_span(text)) # ('Re:', 0, 3),
    #print >> sys.stderr, " ".join(t for t,_,_ in tokenized)
    # lookup (From, To) => [Token, (cat1,pol1),..,(catn,poln)]
    tokenlookup = {(fr,to):[token] for (token,fr,to) in tokenized}
    #

    if 'Opinions' in d['Document'] and 'Opinion' in d['Document']['Opinions']:
        opinions = d['Document']['Opinions']['Opinion']
        if type(opinions) == dict:
            opinions = [opinions]

        for opinion in opinions:
            opfr = int(opinion['@from'])
            opto = int(opinion['@to'])
            category = opinion['@category'].split('#')[0]
            polarity = opinion['@polarity']
            target = unicode(opinion['@target'])
            if opto == 0:
                if not (0,0) in tokenlookup:
                    tokenlookup[(0,0)] = ['__D__',(category,polarity)]
                else:
                    tokenlookup[(0, 0)].append((category,polarity))
            else:

                if not " " in target:

                    if (opfr,opto) in tokenlookup:
                        tokenlookup[(opfr, opto)].append((category, polarity))
                    else:
                        subtokenfromto = get_subtoken(opfr,opto,tokenlookup)
                        if subtokenfromto:
                            print >> sys.stderr, 'SUBTOKEN-FOUND:', text[opfr:opto], "==>" ,text[subtokenfromto[0]:subtokenfromto[1]]
                            tokenlookup[subtokenfromto].append((category,polarity))
                        else:
                            print >> sys.stderr, '#WARNING:SUBTOKEN-NOT-FOUND--GIVING UP', text[opfr:opto]
                else: # multiword target?

                    tokenizedopinionspan = {(_fr,_to):_tkn for (_tkn,_fr,_to) in split_span(text[opfr:opto],offset=opfr)}
                    for (fr,to) in tokenizedopinionspan:
                        if (fr, to) in tokenlookup:
                            tokenlookup[(fr, to)].append((category, polarity))
                        else:
                            subtokenfromto = get_subtoken(fr, to, tokenlookup)
                            if subtokenfromto:
                                print >> sys.stderr, 'SUBTOKEN FOUND', text[fr:to], "==>" , text[subtokenfromto[0]:
                                subtokenfromto[1]]
                                tokenlookup[subtokenfromto].append((category, polarity))
                            else:
                                print >> sys.stderr, '#WARNING:SUBTOKEN-NOT-FOUND--GIVING UP', text[fr:to]
    for (token, fr, to) in tokenized:
        if (fr,to) in tokenlookup and len(tokenlookup[(fr,to)]) > 1:
            cats = " ".join(x[0]+':'+ x[1] for x in tokenlookup[(fr,to)][1:])
        else:
            cats = 'O'
        print >> sys.stdout,'%d-%d-%d\t%s\t%s' %(docseq,fr,to,token,cats)
    if (0,0) in tokenlookup:
        print >> sys.stdout, '%d-%d-%d\t%s\t%s' %(docseq,0,0,'__D__'," ".join(x[0]+':'+ x[1] for x in tokenlookup[(0,0)][1:]))
    else:
        print >> sys.stdout, '%d-%d-%d\t%s\t%s' % (docseq, 0, 0, '__D__', "O")


def get_subtoken(frm, to, tokenlookup):
    """Return start and end position from subtoken of frm-to-token"""
    for (f, t) in tokenlookup:
        if frm >= f and to <=t:
            return f, t

def process(options=None,args=None):
    """
    Do the processing
    """
    if options.debug:
        print >> sys.stderr, options
    dom = etree.parse(sys.stdin)
    for i,d in enumerate(dom.findall('//Document')):
        docjs = bf.data(d)
        docjs2penn(docjs,i,options)


def main():
    """
    Invoke this module as a script
    """

    parser = OptionParser(
        usage = '%prog [OPTIONS] [ARGS...] < XMLINPUT > TSVOUTPUT',
        version='%prog 0.99', #
        description='Convert XML from stdin to tokenized output',
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
    parser.add_option('-m', '--mode',
                      action='store', dest='mode', default='c',
                      help='operation mode: c, cs, r (default=%default)')

    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, "options=",options


    process(options=options,args=args)


if __name__ == '__main__':
    main()
