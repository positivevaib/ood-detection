import os

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
    t = tqdm(hf_datasets)
    for data_name in t:
        t.set_description(data_name)
        if data_name in glue:
            dataset = load_dataset('glue', data_name, split=eval_split_keys[data_name])
        else:
            load_dataset(data_name, split=eval_split_keys[data_name])
