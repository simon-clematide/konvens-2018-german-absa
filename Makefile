# Makefile for experimentation on GermEval 2017 ABSA task

# global variables that can be overwritten
ENSEMBLINGTHRESHOLD?=33



TRAIN_FILE:=data/train_v1.4.xml.tsv
#
#data-txt-files:=$(wildcard data/germeval*txt)
data-xml-files:=$(wildcard data/*.xml)


data-cpenn-files:=$(data-xml-files:.xml=.xml.tsv)
data-cpenn-files+=$(data-xml-files:.xml=.cpenn)


data-cspenn-files+=$(data-xml-files:.xml=.cspenn)


# Intermediate tabseparated format with offsets

%.xml.tsv: %.xml
	python lib/absaxml2tsv.py < $< > $@ 2> $@.log

# 

%.cpenn: %.xml.tsv $(TRAIN_FILE)
	python lib/tsv2penn.py -F $(word 2,$+) $< > $@ 2> $@.log

%.cspenn: %.xml.tsv $(TRAIN_FILE)
	python lib/tsv2penn.py -P -F $(word 2,$+) $< > $@ 2> $@.log

cpenn: $(data-cpenn-files)
cspenn: $(data-cspenn-files)

test:
	python lib/tutorial_bilstm_tagger.py -T data/train_v1.4.cpenn -D data/dev_v1.4.cpenn -E data/test_TIMESTAMP1.cpenn -F data/test_TIMESTAMP2.cpenn



do-cpenn-experiment-%:
	mkdir -p cpenn.d && cd cpenn.d && \
	for i in $$(seq -w 1 $*); do \
		python ../lib/tutorial_bilstm_tagger.py --dynet-mem 2048  -M $${i} 2> experiment-$${i}.log -T ../data/train_v1.4.cpenn -D ../data/dev_v1.4.cpenn -E ../data/test_TIMESTAMP1.cpenn -F ../data/test_TIMESTAMP2.cpenn &  \
	done && wait

do-cpenn-experiment-eval:
	rm -f cpenn.d/00__*tsv
	cd cpenn.d &&  paste *__testset1_.tsv | python ../lib/ensembling.py -m $(ENSEMBLINGTHRESHOLD) | python ../lib/tsv2evalinput.py -g ../data/test_TIMESTAMP1.tsv > 00__testset1-evalin.tsv && \
	java -jar ../GermEval2017/EvaluationScript.jar category 00__testset1-evalin.tsv  ../data/test_TIMESTAMP1.tsv > test_TIMESTAMP1.eval.txt; cat test_TIMESTAMP1.eval.txt
	cd cpenn.d &&  paste *__testset2_.tsv | python ../lib/ensembling.py -m $(ENSEMBLINGTHRESHOLD) | python ../lib/tsv2evalinput.py -g ../data/test_TIMESTAMP2.tsv > 00__testset2-evalin.tsv && \
	java -jar ../GermEval2017/EvaluationScript.jar category 00__testset2-evalin.tsv  ../data/test_TIMESTAMP2.tsv > test_TIMESTAMP2.eval.txt; cat test_TIMESTAMP2.eval.txt



do-cspenn-experiment-%:
	mkdir -p cspenn.d && cd cspenn.d && \
	for i in $$(seq -w 1 $*); do \
		python ../lib/tutorial_bilstm_tagger.py --dynet-mem 2048  -M $${i}  -f 0.61 2> experiment-$${i}.log -T ../data/train_v1.4.cspenn -D ../data/dev_v1.4.cspenn -E ../data/test_TIMESTAMP1.cspenn -F ../data/test_TIMESTAMP2.cspenn &  \
	done && wait

do-cspenn-experiment-eval:
	rm -f cspenn.d/00__*tsv
	cd cspenn.d &&  paste *__testset1_.tsv | python ../lib/ensembling.py -m $(ENSEMBLINGTHRESHOLD) | python ../lib/tsv2evalinput.py -P -g ../data/test_TIMESTAMP1.tsv > 00__testset1-evalin.tsv && \
	java -jar ../GermEval2017/EvaluationScript.jar category 00__testset1-evalin.tsv  ../data/test_TIMESTAMP1.tsv > test_TIMESTAMP1.eval.txt; cat test_TIMESTAMP1.eval.txt
	cd cspenn.d &&  paste *__testset2_.tsv | python ../lib/ensembling.py -m $(ENSEMBLINGTHRESHOLD) | python ../lib/tsv2evalinput.py -P -g ../data/test_TIMESTAMP2.tsv > 00__testset2-evalin.tsv && \
	java -jar ../GermEval2017/EvaluationScript.jar category 00__testset2-evalin.tsv  ../data/test_TIMESTAMP2.tsv > test_TIMESTAMP2.eval.txt; cat test_TIMESTAMP2.eval.txt



# python lib/tsv2evalinputxml.py -g data/test_TIMESTAMP1.xml -o data/test_TIMESTAMP1.xml.tsv < system.tsv  > out.xml 2> out.err
do-task-d-cspenn-experiment-eval:
	rm -f cspenn.d/00__*tsv
	cd cspenn.d &&  paste *__testset1_.tsv | python ../lib/ensembling.py -m $(ENSEMBLINGTHRESHOLD) | python ../lib/tsv2evalinputxml.py  -o ../data/test_TIMESTAMP1.xml.tsv -g ../data/test_TIMESTAMP1.xml > 00__testset1-evalin-taskd.xml && \
	java -jar ../GermEval2017/EvaluationScript.jar OTE 00__testset1-evalin-taskd.xml  ../data/test_TIMESTAMP1.xml > test_TIMESTAMP1.eval.taskd.txt; cat test_TIMESTAMP1.eval.taskd.txt
	cd cspenn.d &&  paste *__testset2_.tsv | python ../lib/ensembling.py -m $(ENSEMBLINGTHRESHOLD) | python ../lib/tsv2evalinputxml.py  -o ../data/test_TIMESTAMP2.xml.tsv -g ../data/test_TIMESTAMP2.xml > 00__testset2-evalin-taskd.xml && \
	java -jar ../GermEval2017/EvaluationScript.jar OTE 00__testset2-evalin-taskd.xml   ../data/test_TIMESTAMP2.xml > test_TIMESTAMP2.eval.taskd.txt; cat test_TIMESTAMP2.eval.taskd.txt




.SECONDARY:
SHELL:=/bin/bash
