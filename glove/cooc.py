#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
from tqdm import tqdm


def main():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    arr = {}
    counter = 1
    for fn in ['pos_train.txt', 'neg_train.txt']:
        num_lines = 0
        with open(fn) as f:
            for line in f:
                num_lines += 1
        with open(fn) as f:
            for line in tqdm(f, total = num_lines):
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        if (t, t2) not in arr: arr[(t, t2)] = 1
                        else: arr[(t, t2)] += 1

                counter += 1

    cooc = coo_matrix(([1] * len(arr), ([x for x, y in arr.keys()], [y for x, y in arr.keys()])))
    with open('cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
