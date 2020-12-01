# import dependencies
import argparse
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

import transformers
from transformers import (AutoTokenizer, AutoModelForSequenceClassification)

import datasets
from datasets import load_dataset

from msp_eval import *


if __name__ == '__main__':
    # create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--roberta_version', type=str, default='roberta-large', help='Version of RoBERTa to use')
    parser.add_argument('--dataset', help='Dataset to compute maximum softmax probability')
    parser.add_argument('--load_dir', type=str, default='output', help='Directory to load tokenizers and models')
    parser.add_argument('--output_dir', type=str, default='msp', help='Directory to save numpy array in')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--file_format', type=str, default='.tsv', help='Data file format for dataset not available for download at HuggingFace Datasets')
    parser.add_argument('--in_domain', action='store_true', help='Dataset is in-domain')
    parser.add_argument('--split', type=str, default='eval', help='Dataset split to compute maximum softmax probability on')

    args = parser.parse_args()

    # huggingface and glue datasets
    hf_datasets = ['imdb', 'sst2']
    glue = ['sst2']

    # custom dataset label keys
    label_keys = {
                    'counterfactual-imdb': 'Sentiment',
            }

    # evaluation split keys
    eval_split_keys = {
                        'imdb': 'test',
                        'sst2': 'validation',
                        'counterfactual-imdb': 'dev'
            }

    # dataset feature keys
    datasets_to_keys = {
                    'imdb': ('text', None),
                    'sst2': ('sentence', None),
                    'counterfactual-imdb': 'Text'
            }

    # load dataset
    print('Loading dataset')

    if args.dataset in hf_datasets:
        dataset = load_dataset(args.dataset, split=eval_split_keys[args.dataset]) if args.dataset not in glue else load_dataset('glue', args.dataset, split=eval_split_keys[args.dataset])
    elif args.file_format == '.tsv':
        train_df = pd.read_table(os.path.join(os.getcwd(), args.dataset, ('train' + args.file_format)))
        eval_df = pd.read_table(os.path.join(os.getcwd(), args.dataset, (eval_split_keys[args.dataset] + args.file_format)))
        test_df = pd.read_table(os.path.join(os.getcwd(), args.dataset, ('test' + args.file_format)))
        num_labels = len(np.unique(pd.Categorical(train_df[label_keys[args.dataset]], ordered=True)))

        if args.split == 'train':
            dataset = train_df
        elif args.split == 'eval':
            dataset = eval_df
        elif args.split == 'test':
            dataset = test_df
        elif args.split == 'all':
            dataset = pd.concat([train_df, eval_df, test_df])

    if args.in_domain:
        sentences, labels = shuffle(dataset[datasets_to_keys[args.dataset][0]], dataset['label'], random_state=args.seed)
        split_idx = int(0.2*len(sentences))
        dataset = {datasets_to_keys[args.dataset][0]: sentences[split_idx:], 'label': labels[split_idx:]}

    tokenizer = AutoTokenizer.from_pretrained(args.load_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.load_dir) 

    id_all_encodings = [encode(tokenizer, text) for text in dataset[datasets_to_keys[args.dataset]]]
    msp = process_msp(id_all_encodings, model)
    np.save(os.path.join(os.getcwd(), args.output_dir, args.roberta_version + '_' + args.dataset + '_msp.npy'), msp)

