from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
from tqdm import tqdm
import torch
from data import *
from sklearn.model_selection import train_test_split
import datasets

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


setup_seed(123)
raw_data = datasets.load_dataset('imdb', split ='train')
train_data, val_data = train_test_split(raw_data,test_size = 0.1)

def summ_aug(data):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    model = model.to(device)
    trainset_aug = IMDB_BART(tokenizer, data, 512)
    dataloader = torch.utils.data.DataLoader(trainset_aug, batch_size=10, shuffle=False, num_workers=0)
    with torch.no_grad():
        preds = list()
        for id,data in tqdm(enumerate(dataloader), mininterval= 60):
            input = data
            input["input_ids"] = input["input_ids"].squeeze(1).to(device)
            input["attention_mask"] = input["attention_mask"].squeeze(1).to(device)
            outputs = model.generate(**input,min_length = 20, max_length = 200)
            decoded_tokens = tokenizer.batch_decode(outputs,skip_special_tokens=True)          
            for x in decoded_tokens:
                preds.append(x)
    with open('summ_aug.txt', 'w', encoding = 'utf-8') as f:
        f.write('[EOSUMM]'.join(preds))
    f.close()

summ_aug(train_data)