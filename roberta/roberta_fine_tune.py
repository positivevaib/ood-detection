# import dependencies
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

import transformers
from transformers import (RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification)
from transformers import set_seed

import datasets
from datasets import load_dataset


def process_hf_dataset(dataset, split, task_name, tokenizer, padding, max_length, batch_size, truncation=True):
    eval_split_keys = {
                        'imdb': 'test',
                        'sst2': 'validation'
            }
    if split == 'eval':
        split = eval_split_keys[task_name]

    tasks_to_keys = {
                    'imdb': ('text', None),
                    'sst2': ('sentence', None)
            }

    sentence1_key, sentence2_key = tasks_to_keys[task_name]
    args = ((dataset[split][sentence1_key],) if sentence2_key is None else (dataset[split][sentence1_key], dataset[split][sentence2_key]))

    features = tokenizer(*args, padding=padding, max_length=max_length, truncation=truncation)
    labels = dataset[split]['label']

    all_input_ids = torch.tensor([f for f in features.input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([f for f in features.attention_mask], dtype=torch.long)
    all_labels = torch.tensor([f for f in labels], dtype=torch.long)

    tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels) 
    sampler = RandomSampler(tensor_dataset)
    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def process_custom_dataset(dataset, task_name, tokenizer, padding, max_length, batch_size, truncation=True):
    tasks_to_keys = {
                        'counterfactual-imdb': ('Text', None, 'Sentiment'),
            }

    sentence1_key, sentence2_key, labels_key = tasks_to_keys[task_name]
    args = ((dataset[sentence1_key].tolist(),) if sentence2_key is None else (dataset[sentence1_key].tolist(), dataset[sentence2_key].tolist()))

    features = tokenizer(*args, padding=padding, max_length=max_length, truncation=truncation)
    labels = pd.Categorical(dataset[labels_key], ordered=True).codes.tolist()

    all_input_ids = torch.tensor([f for f in features.input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([f for f in features.attention_mask], dtype=torch.long)
    all_labels = torch.tensor([f for f in labels], dtype=torch.long)

    tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels) 
    sampler = RandomSampler(tensor_dataset)
    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def main():
    # create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', help='Task to fine-tune RoBERTa on')
    parser.add_argument('--roberta_version', type=str, default='roberta-large', help='Version of RoBERTa to use')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Path to cache directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to fine-tune')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length of the inputs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Adam learning rate')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save fine-tuned models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--file_format', type=str, default='.tsv', help='Data file format for tasks not available for download at HuggingFace Datasets')

    args = parser.parse_args()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # huggingface and glue datasets
    hf_datasets = ['imdb', 'sst2']
    glue = ['sst2']

    # custom dataset label keys
    label_keys = {
                    'counterfactual-imdb': 'Sentiment',
            }

    # load dataset
    print('Loading dataset')

    if args.task_name in hf_datasets:
        dataset = load_dataset(args.task_name) if args.task_name not in glue else load_dataset('glue', args.task_name)
        num_labels = dataset['train'].features['label'].num_classes
    elif args.file_format == '.tsv':
        train_df = pd.read_table(os.path.join(os.getcwd(), args.task_name, ('train' + args.file_format)))
        eval_df = pd.read_table(os.path.join(os.getcwd(), args.task_name, ('eval' + args.file_format)))
        num_labels = len(np.unique(pd.Categorical(train_df[label_keys[args.task_name]], ordered=True)))

    # set seed
    set_seed(args.seed)

    # load RoBERTa tokenizer and model
    print('Loading RoBERTa tokenizer and model')

    config = RobertaConfig.from_pretrained(args.roberta_version, num_labels=num_labels, cache_dir=args.cache_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_version, cache_dir=args.cache_dir)
    model = RobertaForSequenceClassification.from_pretrained(args.roberta_version, config=config, cache_dir=args.cache_dir).to(device)

    # process dataset
    print('Processing dataset')

    padding = 'max_length'
    if args.task_name in hf_datasets:
        train_loader = process_hf_dataset(dataset, 'train', args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size)
        eval_loader = process_hf_dataset(dataset, 'eval', args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size)
    else:
        train_loader = process_custom_dataset(train_df, args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size)
        eval_loader = process_custom_dataset(eval_df, args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size)

    # instantiate optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # fine-tune model 
    print('Fine-tuning model')

    losses = []
    train_iterator = trange(int(args.num_epochs), desc='Epoch')
    for _ in train_iterator:
        tr_loss = 0
        step = None
        epoch_iterator = tqdm(train_loader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
            labels = batch[2].to(device)

            optimizer.zero_grad()

            out = model(**inputs)[1].double().to(device)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
        losses.append(tr_loss/(step+1))
        print('train loss: {}'.format(tr_loss/(step+1)))

    # save model and tokenizer
    print('Saving model and tokenizer')

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # evaluate model
    print('Evaluating model')

    preds = None
    gold_labels = None

    eval_loss = 0
    step = None
    eval_iterator = tqdm(eval_loader, desc='Evaluating')
    for step, batch in enumerate(eval_iterator):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch [0].to(device), 'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
            labels = batch[2].to(device)

            out = model(**inputs)[1].double().to(device)
            loss = criterion(out, labels)

            if preds is None:
                preds = out.detach().cpu().numpy()
                gold_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, out.detach().cpu().numpy(), axis=0)
                gold_labels = np.append(gold_labels, labels.detach().cpu().numpy(), axis=0)

            eval_loss += loss.item()
    eval_loss /= (step+1)
    print('eval loss: {}'.format(eval_loss))

    # compute accuracy
    preds = np.argmax(preds, axis=1)
    accuracy = np.sum(preds == gold_labels)/len(preds)
    print('eval accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    main()
