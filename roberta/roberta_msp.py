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
    parser.add_argument('--entailment', action='store_true', help='Dataset originally intended for an entailment task')
    parser.add_argument('--split', type=str, default='eval', help='Dataset split to compute maximum softmax probability on')

    args = parser.parse_args()

    # huggingface and glue datasets
    hf_datasets = ['imdb', 'snli', 'sst2']
    glue = ['sst2']

    # custom dataset label keys
    label_keys = {
                    'counterfactual-imdb': 'Sentiment',
            }

    # training split keys
    train_split_keys = {
                        'imdb': 'train',
                        'snli': 'train',
                        'sst2': 'train',
                        'counterfactual-imdb': 'train'
            }

    # evaluation split keys
    eval_split_keys = {
                        'imdb': 'test',
                        'snli': 'validation',
                        'sst2': 'validation',
                        'counterfactual-imdb': 'dev'
            }

    # test split keys
    test_split_keys = {
                        'imdb': 'unsupervised',
                        'snli': 'test',
                        'sst2': 'test',
                        'counterfactual-imdb': 'test'
            }

    # dataset feature keys
    datasets_to_keys = {
                    'imdb': ('text', None),
                    'snli': ('premise', 'hypothesis'),
                    'sst2': ('sentence', None),
                    'counterfactual-imdb': ('Text', None)
            }

    # load dataset
    print('Loading dataset')

    if args.dataset in hf_datasets:
        dataset = load_dataset(args.dataset) if args.dataset not in glue else load_dataset('glue', args.dataset)

        if args.split == 'train':
            dataset = dataset[train_split_keys[args.dataset]]
        elif args.split == 'eval':
            dataset = dataset[eval_split_keys[args.dataset]]
        elif args.split == 'test':
            dataset = dataset[test_split_keys[args.dataset]]

    elif args.file_format == '.tsv':
        train_df = pd.read_table(os.path.join(os.getcwd(), args.dataset, (train_split_keys[args.dataset] + args.file_format)))
        eval_df = pd.read_table(os.path.join(os.getcwd(), args.dataset, (eval_split_keys[args.dataset] + args.file_format)))
        test_df = pd.read_table(os.path.join(os.getcwd(), args.dataset, (test_split_keys[args.dataset] + args.file_format)))
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

    if args.entailment:
        sentences = [sentence1 + ' ' + sentence2 for sentence1, sentence2 in zip(dataset[datasets_to_keys[args.dataset][0]], dataset[datasets_to_keys[args.dataset][1]])]
        labels = dataset['label']
        dataset = {datasets_to_keys[args.dataset][0]: sentences, 'label': labels}
            

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.load_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.load_dir).to(device) 

    # output sample data
    print(dataset[datasets_to_keys[args.dataset][0]][0])

    id_all_encodings = [encode(tokenizer, text) for text in dataset[datasets_to_keys[args.dataset][0]]]
    msp = process_msp(id_all_encodings, model, device)
    np.save(os.path.join(os.getcwd(), args.output_dir, args.roberta_version + '_' + args.dataset + '_msp.npy'), msp)

