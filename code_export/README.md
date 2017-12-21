# Project02
Project 2: text classification using most frequent word count

## Requirements
The code was tested on a Google Cloud 8 CPU, 30GB RAM, 20G HDD instance. It takes about two hours to run

## INSTALLATION
(in virtualenv)
```
 $ # need virtualenv and python3
 $ virtualenv -p $(which python3) create
 $ source venv/bin/activate
 $ pip install -r requirements.txt
```

## Run
1. Copy `train_neg_full.txt` and `train_pos_full.txt` to `data` folder
2. Run `python run.py` in virtualenv

Will generate a file submission.txt

## Contents
This code does preprocess data and then uses most frequent word occurrences vector as feature vector.

It is then classified using Keras fully connected neural network
