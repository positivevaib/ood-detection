import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

def process_entailment(dataset, tokenizer, key1='sentence1', key2='sentence2'):
    dataset_texts = []
    for ex in dataset:
        dataset_texts.append(ex[key1] + ' ' + ex[key2])
    return [encode(tokenizer, text) for text in dataset_texts]

def encode(tokenizer, text):
    return tokenizer.encode_plus(
      text,
      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
      return_token_type_ids=False,
      max_length=150,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',  # Return PyTorch tensors
    )

def process_msp(all_encodings, model, device='cpu'):
    scores = []
    for encoding in tqdm(all_encodings):
        input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
        out = model(input_ids, attention_mask)[0]
        score = F.softmax(out[0], dim=0)
        scores.append(score.detach().cpu().numpy())
    max_probs = np.max(np.array(scores), axis=1)
    return max_probs

if __name__ == '__main__':
    sst2_test = load_dataset('glue', 'sst2', split='test')
    rte_val = load_dataset('glue', 'rte', split='validation')

    tokenizer_sst2 = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    model_sst2 = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")

    id_all_encodings = [encode(tokenizer_sst2, text) for text in sst2_test['sentence']]
    rte_all_encodings = process_entailment(rte_val, tokenizer_sst2)
    
    sst2_msp = process_msp(id_all_encodings, model_sst2)
    rte_msp = process_msp(rte_all_encodings, model_sst2)

    # np.save('../output/sst2/rte_msp.npy', rte_msp)
    # np.save('../output/sst2/sst2_msp.npy', sst2_msp)
