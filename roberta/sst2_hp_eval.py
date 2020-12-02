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

    args = parser.parse_args()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set seed
    set_seed(args.seed)

    # load dataset
    dataset = load_dataset('glue', 'sst2', split='validation')
    sentences, labels = shuffle(dataset['sentence'], dataset['label'], random_state=args.seed)
    split_idx = int(0.2*len(sentences))
    dev_data = {'sentence': sentences[:split_idx], 'label': labels[:split_idx]}

    # create file to log accuracies
    acc_file = open('accuracy.txt', 'w+')

    for model_type in ['base', 'large']:
        for epochs in ['3', '10']:
            for batch_size in ['16', '32']:
                for learning_rate in ['1e-5', '3e-6', '3e-5', '5e-5']:
                    directory = model_type + '_' + epochs + '_' + batch_size + '_' + learning_rate + '/out'

                    # load tokenizer and model
                    print('Loading tokenizer and model for', model_type, epochs, batch_size, learning_rate)

                    tokenizer = AutoTokenizer.from_pretrained(directory)
                    model = AutoModelForSequenceClassification.from_pretrained(directory).to(device)

                    # process dataset
                    print('Processing dataset')

                    padding = 'max_length'
                    features = tokenizer(dev_data['sentence'], padding=padding, max_length=args.max_seq_length, truncation=True)
                    labels = dev_data['label']

                    all_input_ids = torch.tensor([f for f in features.input_ids], dtype=torch.long)
                    all_attention_mask = torch.tensor([f for f in features.attention_mask], dtype=torch.long)
                    all_labels = torch.tensor([f for f in labels], dtype=torch.long)

                    tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels) 
                    sampler = RandomSampler(tensor_dataset)
                    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=int(batch_size))

                    # evaluate model
                    print('Evaluating model')

                    preds = None
                    gold_labels = None

                    step = None
                    eval_iterator = tqdm(dataloader, desc='Evaluating')
                    for step, batch in enumerate(eval_iterator):
                        model.eval()
                        batch = tuple(t.to(device) for t in batch)

                        with torch.no_grad():
                            inputs = {'input_ids': batch [0].to(device), 'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
                            labels = batch[2].to(device)

                            out = model(**inputs)[1].double().to(device)

                            if preds is None:
                                preds = out.detach().cpu().numpy()
                                gold_labels = labels.detach().cpu().numpy()
                            else:
                                preds = np.append(preds, out.detach().cpu().numpy(), axis=0)
                                gold_labels = np.append(gold_labels, labels.detach().cpu().numpy(), axis=0)

                    # compute accuracy
                    preds = np.argmax(preds, axis=1)
                    accuracy = np.sum(preds == gold_labels)/len(preds)
                    print('eval accuracy: {}'.format(accuracy))

                    # write accuracy to file
                    acc_file.write(model_type + '_' + epochs + '_' + batch_size + '_' + learning_rate + ' ' + str(accuracy) + '\n')
    
    acc_file.close()

if __name__ == '__main__':
    main()
