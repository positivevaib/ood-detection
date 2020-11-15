# import dependencies
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

import transformers
from transformers import (RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification)

import datasets
from datasets import load_dataset


def process_dataset(dataset, split, task_name, tokenizer, padding, max_length, batch_size, truncation=True):
    tasks_to_keys = {
                    'imdb': ('text', None),
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


def main():
    # create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', help='Task to fine-tune RoBERTa on')
    parser.add_argument('--roberta_version', type=str, default='roberta-large', help='Version of RoBERTa to use')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Path to cache directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to fine-tune')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length of the inputs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    args = parser.parse_args()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = load_dataset(args.task_name)
    num_labels = dataset['train'].features['label'].num_classes

    # load RoBERTa tokenizer and model
    config = RobertaConfig.from_pretrained(args.roberta_version, num_labels=num_labels, finetuning_task=args.task_name, cache_dir=args.cache_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_version, cache_dir=args.cache_dir)
    model = RobertaForSequenceClassification.from_pretrained(args.roberta_version, config=config, cache_dir=args.cache_dir).to(device)

    # process dataset
    padding = 'max_length'
    train_loader = process_dataset(dataset, 'train', args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size)
    test_loader = process_dataset(dataset, 'test', args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size)

    # instantiate optimizer and criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # fine-tune model 
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

if __name__ == '__main__':
    main()
