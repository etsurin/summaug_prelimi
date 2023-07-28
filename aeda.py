# AEDA: An Easier Data Augmentation Technique for Text classification
# Akbar Karimi, Leonardo Rossi, Andrea Prati

import random
import datasets
from sklearn.model_selection import train_test_split
from data import *

random.seed(123)

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


setup_seed(123)
raw_data = datasets.load_dataset('imdb', split ='train')
train_data, val_data = train_test_split(raw_data,test_size = 0.1)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
NUM_AUGS = 1
PUNC_RATIO = 0.3

# Insert punction words into a given sentence with the given ratio "punc_ratio"
def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(punc_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line


def main(data):
	data_aug = []
	N = len(data['text'])
	for id in range(N):
		item = data['text'][id]
		item_aug = insert_punctuation_marks(item)
		# if id == 0:
		# 	print(item_aug)
		data_aug.append(item_aug)

	with open( 'aeda_aug.txt', 'w') as f:
		f.write('[EOSUMM]'.join(data_aug))

main(train_data)