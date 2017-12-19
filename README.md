# Project02
Project 2: text classification

# LIBRARIES :
Pyenchant : http://pythonhosted.org/pyenchant/tutorial.html#installing-pyenchant
Please install using pip 

# Data Processing:
a) Install Pyenchant
b) create a directory in Project2 called tmp
c) Run the following blocks in the presented order:
1) imports 
2) Loading datasets ( only first two blocks (tokenize_tweets,loadFile)
3) Stemming words (All blocks)
4) Correcting Misspelled words (2nd block)
5) Correcting and stemming all texts (all blocks)
6) Strip initial characters (first four blocks)
7) Generate clean tweets (all blocks)

After some time the following files will be generated in the tmp directory:
clean_train_pos.txt : file having positive processed tweets
clean_train_vocab_pos.txt : file having vocabullary used to train model
clean_train_neg.txt : file having negative processed tweets
clean_train_vocab_neg.txt : file having vocabullary used to train model

Please make sure when loading the vocabullary to combine them and keep unique set of them 
