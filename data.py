from re import T
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class IMDBDataset(Dataset):
    def __init__(self, tokenizer, raw_data, max_length):

        self.data = []
        self.labels = []
        N = len(raw_data['text'])
        for id in tqdm(range(N)):
            tmp_src = tokenizer(raw_data['text'][id], max_length = max_length, padding = 'max_length', truncation = True, return_tensors='pt')
            tmp_src['input_ids'] = tmp_src['input_ids'][0]
            tmp_src['attention_mask'] = tmp_src['attention_mask'][0]
            self.data.append(tmp_src)
            self.labels.append(raw_data['label'][id])
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item],self.labels[item]

# class IMDB_GPT(Dataset):
#     def __init__(self, tokenizer, raw_data, max_length, prompt = False, train = False):

#         self.data = []
#         self.labels = []
#         N = len(raw_data['text'])
#         self.gensrc = []
#         if prompt:
#             for id in tqdm(range(N)):
#                 text = 'Identify whether the review is positive or negative: \n' + raw_data['text'][id]
#                 if raw_data['label'] == 1:
#                     if train:
#                         text = text + '\n answer: positive'
#                     else:
#                         text = text + '\n answer:'
#                 else:
#                     if train:
#                         text = text + '\n answer: negative'
#                     else:
#                         text = text + '\n answer:'
#                 tmp_data = tokenizer.encode(text, max_length = max_length, truncation = True, padding = 'max_length')
#                 self.data.append(tmp_data)
#                 self.labels.append([-100 if x == tokenizer.pad_token_id else x for x in tmp_data])
#             self.data = torch.tensor(self.data)
#             self.labels = torch.tensor(self.labels)
#         else:
#             for id in tqdm(range(N)):
#                 tmp_data = tokenizer.encode(raw_data['text'][id], max_length = max_length, truncation = True, padding = 'max_length')
#                 self.data.append(tmp_data)
#                 self.labels.append([-100 if x == tokenizer.pad_token_id else x for x in tmp_data])
#             self.data = torch.tensor(self.data)
#             self.labels = torch.tensor(self.labels)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         return self.data[item], self.labels[item]

class IMDB_BART(Dataset):
    def __init__(self, tokenizer, raw_data, max_length):
        self.data = []
        N = len(raw_data['text'])
        for id in tqdm(range(N)):
            self.data.append(tokenizer(raw_data['text'][id], max_length = max_length, padding = 'max_length', truncation = True, return_tensors='pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
def get_aug_data(filename, train_data):
    f = open('{}.txt'.format(filename),'r')
    data = f.read()
    data = data.split('[EOSUMM]')
    print(len(data))
    for id,item in enumerate(data):
        train_data['text'].append(item)
        train_data['label'].append(train_data['label'][id])
    return train_data