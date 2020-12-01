# import dependencies
import argparse

import sklearn
from sklearn.utils import shuffle
import datasets
from datasets import load_dataset

from msp_eval import *

if __name__ == '__main__':
    # create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # get the best models
    with open('accuracy.txt') as file_:
        lines = file_.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n').split(' ')

    best = {'base': None, 'large': None}
    acc = {'base': 0, 'large': 0}

    for line in lines:
        key = 'base' if line[0].startswith('base') else 'large'
        if float(line[-1]) > acc[key]:
            best[key] = line[0]
            acc[key] = float(line[-1])

    # compute msp for the best models in roberta-base and roberta-large
    # in-domain: sst2-test (split from sst2-dev), out-out-domain: imdb-dev

    # load datasets
    sst2_dev = load_dataset('glue', 'sst2', split='validation')
    sentences, labels = shuffle(sst2_dev['sentence'], sst2_dev['label'], random_state=args.seed)
    split_idx = int(0.2*len(sentences))
    sst2_test = {'sentence': sentences[split_idx:], 'label': labels[split_idx:]}

    imdb_dev = load_dataset('imdb', split='test')

    for model_type in best.keys():
        print('Evaluating', model_type)

        directory = best[model_type] + '/out'
        tokenizer = AutoTokenizer.from_pretrained(directory)
        model = AutoModelForSequenceClassification.from_pretrained(directory)

        sst2_id_all_encodings = [encode(tokenizer, text) for text in sst2_test['sentence']]
        sst2_msp = process_msp(sst2_id_all_encodings, model)
        np.save('msp/' + model_type + '_sst2_msp.npy', sst2_msp)

        imdb_id_all_encodings = [encode(tokenizer, text) for text in imdb_dev['text']]
        imdb_msp = process_msp(imdb_id_all_encodings, model)
        np.save('msp/' + model_type + '_imdb_msp.npy', imdb_msp)

