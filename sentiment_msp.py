import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import os

if not os.path.exists('output'):
    os.makedirs('output')

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

if __name__ == '__main__':
    yelp_test = load_dataset('yelp_polarity', split='test')
    model = BertForSequenceClassification.from_pretrained('lvwerra/bert-imdb')
    tokenizer = AutoTokenizer.from_pretrained('lvwerra/bert-imdb')
    n = 3000
    all_encodings = [encode(tokenizer, text) for text in yelp_test['text'][:n]]

    scores = []
    for encoding in tqdm(all_encodings):
        input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
        out = model(input_ids, attention_mask)[0]
        score = F.softmax(out[0], dim=0)
        scores.append(score.detach().cpu().numpy())

    yelp_max_probs = np.max(np.array(scores), axis=1)
    np.save('output/imdb/yelp_msp.npy', yelp_max_probs)
