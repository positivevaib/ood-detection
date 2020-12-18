# import dependencies
import argparse
import os

import numpy as np
import sklearn
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm

import transformers
from transformers import (AutoTokenizer, AutoModelForSequenceClassification)
from transformers import set_seed

import datasets
from datasets import load_dataset


def main():
    # create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length of the inputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cache', type=str, default=None, help='where datasets stored')

    args = parser.parse_args()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set seed
    set_seed(args.seed)

    # load dataset
    dataset = load_dataset('glue', 'sst2', split='validation', cache_dir=args.cache)
    orig_sentences = list(enumerate(dataset['sentence']))
    orig_labels    = list(enumerate(dataset['label']))
    s_sentences, s_labels = shuffle(orig_sentences, orig_labels, random_state=args.seed)
    indices, sentences = zip(*s_sentences)
    _, labels          = zip(*s_labels)

    split_idx = int(0.2*len(sentences))
    #calculate indices of dev_data in the original dataset without the shuffle
    indices = indices[split_idx:]

    with open('sst2_indices.txt', 'w') as indices_file:
        for index in indices:
            indices_file.write("%s\n" % index)


if __name__ == '__main__':
    main()
