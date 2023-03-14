import os
from transformers import BertTokenizer
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler)
import numpy as np
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class Dataset(Dataset):
	def __init__(self, texts, labels, max_length=512):
		self.labels = list(labels)
		self.texts = [tokenizer(text, padding='max_length', max_length=max_length,
								truncation=True, return_tensors='pt') for text in texts]
	
	def classes(self):
		return self.labels
	
	def __len__(self):
		return len(self.labels)
	
	def get_batch_labels(self, idx):
		# Fetch a batch of labels
		return np.array(self.labels[idx])
	
	def get_batch_texts(self, idx):
		# Fetch a batch of inputs
		return self.texts[idx]
	
	def __getitem__(self, idx):
		batch_x = self.get_batch_texts(idx)
		batch_y = self.get_batch_labels(idx)

		return batch_x, batch_y


def get_data(path, y):

	texts = []
	labels = []

	with open(path, encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			texts.append(line)
			labels.append(y)

	return texts, labels



def data_loader(paths, batch_size=16, max_length=512):
	# get data as list of dict with text and label
	train_neg_x_ls, train_neg_y_ls = get_data(paths[0], 1)
	train_non_x_ls, train_non_y_ls = get_data(paths[1], 0)
	test_neg_x_ls, test_neg_y_ls = get_data(paths[2], 1)
	test_non_x_ls, test_non_y_ls = get_data(paths[3], 0)

	# combine x and y of neg and non for each train and test
	train_x_np = np.array(train_neg_x_ls + train_non_x_ls)
	train_y_np = np.array(train_neg_y_ls + train_non_y_ls)
	test_x_np = np.array(test_neg_x_ls + test_non_x_ls)
	test_y_np = np.array(test_neg_y_ls + test_non_y_ls)

	# split train to train and validation dataset
	train_x_np, valid_x_np, train_y_np, valid_y_np = train_test_split(train_x_np, train_y_np, test_size=0.2, random_state=42, shuffle=True)

	print('train:', len(train_x_np))
	print('validation: ', len(valid_x_np))
	print('test: ', len(test_x_np))

	tot_data = [ train_x_np, train_y_np, valid_x_np, valid_y_np, test_x_np, test_y_np ]

	train = Dataset(train_x_np, train_y_np, max_length)
	valid = Dataset(valid_x_np, valid_y_np, max_length)
	test = Dataset(test_x_np, test_y_np, max_length)

	train_sampler = RandomSampler(train)
	train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=batch_size, drop_last=True)

	valid_sampler = SequentialSampler(valid)
	valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=batch_size, drop_last=True)

	test_sampler = SequentialSampler(test)
	test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=batch_size, drop_last=True)

	return train_dataloader, valid_dataloader, test_dataloader
