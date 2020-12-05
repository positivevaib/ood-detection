from os.path import split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
import utils

SAVE_PATH = 'output/gpt2/imdb'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

LR = '5e-5'
EPOCHS = 1

def compute_all(model, all_encodings, fname, n=None):
    print("Finding perplexities")
    perplexities, lls = [], []
    # all_encodings = all_encodings[:n]
    pbar = tqdm(total=len(all_encodings))
    for idx, encodings in enumerate(all_encodings):
        try:
            pp, ll = compute_perplexity(model, encodings, device=device)
            perplexities.append(pp)
            lls.append(ll)
        except Exception as e:
            print("Exception at idx", idx)
            print(e)
            continue
        finally:
            pbar.update(1)
    
    pbar.close()

    perplexities = np.array(perplexities)
    np.save(f'{SAVE_PATH}/{fname}_{LR}_pps.npy', perplexities)
    print(f"\nMean: {perplexities.mean()}, Std: {perplexities.std()}")

    with open(f'{SAVE_PATH}/{fname}_{LR}_lls.pkl', 'wb') as fw:
        pickle.dump(lls, fw)

    return perplexities

def compute_perplexity(model, encodings, stride=None, device='cuda'):
    max_length = model.config.n_positions
    lls = []
    if stride is None:
        stride = 1

    for i in range(1, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / i)
    return ppl.item(), torch.stack(lls).detach().cpu().numpy()

def setup(path='lvwerra/gpt2-imdb'):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    n = None
    model = AutoModelWithLMHead.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer, device, n

def process_dataset(dataset_name, model, tokenizer, device, n=None, key='text', configs=None, fname=None):
    print(f"\n------Processing perplexity for dataset: {dataset_name}-------")

    if configs is None:
        dataset = load_dataset(dataset_name, split='test')
    else:
        dataset = [load_dataset(dataset_name, config, split='test') for config in configs]
    
    print("Tokenizing data")
    if configs is None:
        all_encodings = [tokenizer(text + ' <|endoftext|>', return_tensors='pt') for text in dataset[key][:n]]
    else:
        all_encodings = [tokenizer(text + ' <|endoftext|>', return_tensors='pt') for _dataset in dataset for text in _dataset[key][:n]]

    if fname is None:
        fname = dataset_name
    return compute_all(model, all_encodings, fname, n)

def process_entailment(dataset_name, model, tokenizer, device, n=None, dataset_subname=None, fname=None, key1='premise', key2='hypothesis'):
    print(f"\n------Processing perplexity for dataset: {dataset_name}_{dataset_subname}-------")

    if dataset_subname is None:
        dataset = load_dataset(dataset_name, split='validation')
    else:
        dataset = load_dataset(dataset_name, dataset_subname, split='validation')
    dataset_texts = []
    for ex in dataset:
        dataset_texts.append(ex[key1] + ' ' + ex[key2])

    print("Tokenizing data")
    all_encodings = [tokenizer(text + ' <|endoftext|>', return_tensors='pt') for text in dataset_texts[:n]]

    if fname is None:
        fname = dataset_name
    return compute_all(model, all_encodings, fname, n)

def process_counterfactual(dataset_name, model, tokenizer, device, n=None, fname=None, key='Text'):
    print('Loading data...')
    train_df = pd.read_table(os.path.join(os.getcwd(), 'data', dataset_name, (utils.train_split_keys[dataset_name] + '.tsv')))
    eval_df = pd.read_table(os.path.join(os.getcwd(), 'data', dataset_name, (utils.eval_split_keys[dataset_name] + '.tsv')))
    test_df = pd.read_table(os.path.join(os.getcwd(), 'data', dataset_name, (utils.test_split_keys[dataset_name] + '.tsv')))

    num_labels = len(np.unique(pd.Categorical(train_df[utils.label_keys[dataset_name]], ordered=True)))
    dataset = pd.concat([train_df, eval_df, test_df])
    print(dataset)

    print("Tokenizing data")
    all_encodings = [tokenizer(text + ' <|endoftext|>', return_tensors='pt') for text in dataset[key][:n]]

    if fname is None:
        fname = dataset_name
    return compute_all(model, all_encodings, fname, n)

if __name__ == '__main__':
    print("Loading model...")
    path = f'/scratch/ua388/nlp/ckpts/gpt2-imdb-{EPOCHS}_epochs-{LR}_lr'
    # path = f'/scratch/ua388/nlp/ckpts/gpt2-glue_sst2-{EPOCHS}_epochs-{LR}_lr'
    print("Loading model...", path)
    model, tokenizer, device, n = setup(path)
    # process_dataset('imdb', model, tokenizer, device, n=3000, key='text')
    # process_dataset('yelp_polarity', model, tokenizer, device, n=3000, key='text')

    process_counterfactual('counterfactual-imdb', model, tokenizer, device, n=2)

    # process_dataset('sentiment140', model, tokenizer, device, n=2, key='text')

    # process_dataset('glue', model, tokenizer, device, configs=['sst2'], key='sentence', fname='sst2')

    # process_entailment('glue', model, tokenizer, device, dataset_subname='rte', fname='rte', key1='sentence1', key2='sentence2')
    # process_entailment('snli', model, tokenizer, device, key1='premise', key2='hypothesis')

    print("\n\n--------DONE--------")
