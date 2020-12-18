import os
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import datasets
from datasets import load_dataset

from tqdm import tqdm

data_out = os.path.join('.','datasets')
custom_out = os.path.join(data_out, 'used_evals')

in_domains = [
    'imdb',
    'sst2',
]

seed = 42

hf_datasets = [
    'imdb',
    'rte',
    'snli',
    'sst2',
]

glue = ['rte', 'sst2']
entailment = ['snli', 'rte']

other_datasets = {
    'counterfactual-imdb':{
        'base': os.path.join(data_out, 'counterfactually-augmented-data', 'sentiment', 'new'),
        'files': [
            'dev.tsv',
            'test.tsv',
            'train.tsv',
        ]
    }
}

train_split_keys = {
                        'imdb': 'train',
                        'rte': 'train',
                        'snli': 'train',
                        'sst2': 'train',
                        'counterfactual-imdb': 'train'
            }

eval_split_keys = {
    'imdb': 'test',
    'rte': 'validation',
    'snli': 'validation',
    'sst2': 'validation',
    'counterfactual-imdb': 'dev'
}

datasets_to_keys = {
    'imdb': ('text', None),
    'rte': ('sentence1', 'sentence2'),
    'snli': ('premise', 'hypothesis'),
    'sst2': ('sentence', None),
    'counterfactual-imdb': ('Text', None)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', type=str, default=None)
    args = parser.parse_args()
    data = {}
    t = tqdm(hf_datasets)
    for data_name in t:
        t.set_description(data_name)
        if data_name in glue:
            data[data_name] = load_dataset('glue', data_name, split=eval_split_keys[data_name], cache_dir=args.cache)
        else:
            data[data_name] = load_dataset(data_name, split=eval_split_keys[data_name], cache_dir=args.cache)

    data_out = {}
    for data_name, val_data in data.items():
        domain_key = 'ood'

        # for in-domain random split
        if data_name in in_domains:
            sentences, labels = shuffle(
                val_data[datasets_to_keys[data_name][0]], val_data['label'], random_state=seed
            )
            print(type(val_data[datasets_to_keys[data_name][0]]), type(val_data[datasets_to_keys[data_name][0]][0]))
            split_idx = int(0.2*len(sentences))
            data_out[('id', 'val', data_name)] = {'text': sentences[split_idx:], 'label': labels[split_idx:]}

            # training data
            # temp = load_dataset(data_name, split=train_split_keys[data_name], cache_dir=args.cache)
            # data_out[('id', 'train', data_name)] = {'text': temp[datasets_to_keys[data_name][0]], 'label': temp['label']}

        # out-of-domain splits
        if data_name in entailment:
            sentences = [
                sentence1 + ' ' + sentence2
                for sentence1, sentence2 in zip(val_data[datasets_to_keys[data_name][0]], val_data[datasets_to_keys[data_name][1]])
            ]
            labels = val_data['label']
            data_out[(domain_key, 'val', data_name)] = {'text': sentences, 'label': labels}
        else:
            data_out[(domain_key, 'val', data_name)] = {'text': val_data[datasets_to_keys[data_name][0]], 'label': val_data['label']}


    with open(os.path.join(args.cache, 'hf_data.p'), 'wb') as f:
        pickle.dump(data_out, f)
