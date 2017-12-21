# Project02
Project 2: text classification using most frequent word count

## Requirements
The code was tested on a Google Cloud 8 CPU, 30GB RAM, 20G HDD instance running on Debian. It takes about two hours to run

## INSTALLATION
(in virtualenv)
```
 $ # need virtualenv and python3
 $ virtualenv -p $(which python3) create
 $ source venv/bin/activate
 $ pip install -r requirements.txt
```

## Run
1. Copy `test_data.txt`, `train_neg_full.txt` and `train_pos_full.txt` to `data` folder from Kaggle
2. Run `python run.py` in virtualenv

Will generate a file submission.txt

## Contents
The code classifies the data using word count vectors. Each vector contains number of occurences of each word in the vocabulary in the tweet.
Vectors are classified using MLPClassifier from scikit-learn, layers shapes are 100x50
