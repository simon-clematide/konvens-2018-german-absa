# Dynet biLSTM tutorial tagger for ABSA
The make build system is used to for data preprocessing and running the experiments. See below.
[Paper](publication/konvens_short_paper.pdf) and [Poster](publication/konvens-poster.pdf) are included in the repo.

The script applies the following system and hyperparameters:
 - Adam trainer
 - Ignore characters occurring less than 5 times
 - Use character-level word representation if occurring less than 3 times
 - character hidden layer dimensions: 32
 - use [dynet.CoupledLSTMBuilder  LSTM variant](https://dynet.readthedocs.io/en/latest/python_ref.html?highlight=CoupledLSTMBuilder#dynet.CoupledLSTMBuilder)
 - word hidden layer dimensions: 64 per LSTM (128 in biLSTM)
 - no dropout, no noise addition
 - MLP hidden layer size 128; no regularization


### Official XML data is in directory data
  - train_v1.4.xml (official training set)
  - dev_v1.4.xml (official development set)
  - test_TIMESTAMP1.xml (synchronic test set 1)
  - test_TIMESTAMP2.xml (diachronic test set 2)
  - test_TIMESTAMP1.tsv (synchronic test set 1 used for Task C evaluation)
  - test_TIMESTAMP2.tsv (diachronic test set 2 used for Task C evaluation)


### Test data output for Task C and D as reported in KONVENS paper
  - Task C A (using aspect aspect only for training; note that test data have negative aspect sentiment as dummy default):
    - konvens2018_results/TaskC_A/00__testset1-evalin.tsv
    - konvens2018_results/TaskC_A/00__testset2-evalin.tsv

  - Task C A:S (using aspect:sentiment pairs for training):
    - konvens2018_results/TaskC_AS/00__testset1-evalin.tsv
    - konvens2018_results/TaskC_AS/00__testset2-evalin.tsv
  - Task D
  	- konvens2018_results/TaskD/00__testset1-evalin-taskd.xml
  	- konvens2018_results/TaskD/00__testset2-evalin-taskd.xml

 
#### Conversion of original XML into tokenized tabulator separated format 
Call: ```python lib/absaxml2tsv.py < data/test_TIMESTAMP2.xml > data/test_TIMESTAMP2.tsv 2> data/test_TIMESTAMP2.tsv.log```

 1. column: DOCID-ONSET-OFFSET (all offsets and offset are zero-based, and as in Python slice notation the position of the offset is not included `"Wenn die"[5:8]` => `"die"`
 2. column: TOKEN (Note that tokenization is regex based and can be easily modified); token __D__ is the dummy token encoding the document-level aspect without any text mention reference.
 3. column: O|ASPECTCATEGORY:SENTIMENT (O = uppercase letter o encodes neutral label) There can be several space-separated labels if there is more than one  annotation for a single token.
 
 
```
15540-0-3       Bei     O
15540-4-7       uns     O
15540-8-15      hinterm O
15540-16-20     Haus    O
15540-21-23     is      O
15540-24-26     ne      O
15540-27-36     Baustelle       O
15540-37-39     an      O
15540-40-43     der     O
15540-44-48     Bahn    O
15540-49-52     und     O
15540-53-56     der     O
15540-57-60     Typ     O
15540-60-61     ,       O
15540-62-65     der     O
15540-66-69     mit     O
15540-70-76     seinem  O
15540-77-83     Signal  O
15540-84-89     immer   O
15540-90-93     die     O
15540-94-102    Arbeiter        O
15540-103-108   warnt   O
15540-108-109   ,       O
15540-110-113   ist     O
15540-114-117   vom     Sicherheit:negative Sonstige_Unregelmässigkeiten:negative
15540-118-121   ICE     Sicherheit:negative Sonstige_Unregelmässigkeiten:negative
15540-122-129   erfasst Sicherheit:negative Sonstige_Unregelmässigkeiten:negative
15540-130-136   worden  O
15540-137-140   T_T     O
15540-0-0       __D__   O
```

#### Penn-like POS tagger input format for tutorial tagger (one document = one "sentence")
There are two variants of all data sets (recognizable by their file extension)

 - cpenn: Only aspect categories as tags.
 - cspenn: Combined aspect and sentiment tags A:S.

### Step-by-Step Howto

```
# make sure you have dynet >= 2 under Python 2.7 available
# checkout repository
git clone --recursive  git clone https://github.com/simon-clematide/konvens-2018-german-absa

# create cpenn representation
make cpenn

# Experimente starten und und N=24 Modelle trainieren (zahl kann beliebig gesetzt werden innerhalb der auf der Maschine verfügbaren Kerne)
# Start training and apply the models with the best devset performance to the test set. The number at the end indicates how many models will be built (in parallel).
make do-cpenn-experiment-24

# Evaluate the ensemble of your models
# You find the results in cpenn.d/*eval.txt
make do-cpenn-experiment-eval


# Do the same procedure for combined aspect sentiment labels.
# Results are in cspenn.d/*eval.txt
make cspenn
make do-cspenn-experiment-24
make do-cspenn-experiment-eval

```

## How to cite

```
@inproceedings{Clematide:2018,
	Address = {Vienna, Austra},
	Author = {Simon Clematide},
	Booktitle = {PROCEEDINGS of the 14th Conference on Natural Language Processing (KONVENS 2018)},
	Editor = {Adrien Barbaresi and Hanno Biber and Friedrich Neubarth and Rainer Osswald},
	Month = {sep},
	Pages = {29-33},
	Title = {A Simple and Effective biLSTM Approach to Aspect-Based Sentiment Analysis in Social Media Customer Feedback},
	Year = {2018}}
```
