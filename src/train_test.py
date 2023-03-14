import argparse
import pprint

import data_loader

from models.bert import Bert
from models.lastFour import LastFour

import trainer
import tester
import model_util as mu

import torch, gc
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import timeit

from transformers import AdamW


def define_argparser():
	p = argparse.ArgumentParser()

	p.add_argument(
		'--research_subject',
		required=True,
		help='The name of the research subject. (ex: server1)',
	)

	p.add_argument(
		'--research_num',
		required=True,
		help='The number of current test for a subject experiment. (ex: 01)',
	)

	p.add_argument(
		'--batch_size',
		type=int,
		default=16,
		help='Mini batch size for gradient descent. Default=%(default)s',
	)

	p.add_argument(
		'--max_length',
		type=int,
		default=512,
		help='Maximum length of the training sequence. Default=%(default)s',
	)

	p.add_argument(
		'--dropout',
		type=float,
		default=0.0,
		help='Dropout rate. Default=%(default)s',
	)

	p.add_argument(
		'--lr',
		type=float,
		default=0.0001,
		help='Initial learning rate. Default=%(default)s',
	)

	p.add_argument(
		'--n_epochs',
		type=int,
		default=20,
		help='Number of epochs to train. Default%(default)s',
	)

	p.add_argument(
		'--train_bert',
		required=True,
		help='Bert Model trained if True.',
	)

	p.add_argument(
		'--train_type',
		required=True,
		help='The type to train model. (ex. Last Four)',
	)

	config = p.parse_args()

	return config


def get_model(config):
	if config.train_type == 'normal':
		model = Bert(dropout=config.dropout)
	elif config.train_type == 'lastFour':
		model = LastFour(dropout=config.dropout)

	return model

def get_crit():
	crit = nn.CrossEntropyLoss()
	return crit

def get_optimizer(model, config):
	# optimizer = optim.Adam(model.parameters(), lr=config.lr)
	optimizer = AdamW(model.parameters(), lr=config.lr)

	return optimizer
	


def main(config):
	
	# ********** PRINT CONFIG HELP **********
	def print_config(config):
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(config))
	print_config(config)


	# ********** LOAD DATA **********
	paths = ['../data/train.negative.csv',
				'../data/train.non-negative.csv',
				'../data/test.negative.csv',
				'../data/test.non-negative.csv']

	train, valid, test = data_loader.data_loader(paths,
												batch_size=config.batch_size,
												max_length=config.max_length)

	
	# ********** GET MODEL, LOSS FUNCTION - MOVED TO DEVICE **********
	model = get_model(config)
	crit = get_crit()

	# Set bert model not to be trained
	if config.train_bert == 'False':
		for param in model.bert.parameters():
			param.requires_grad = False

	# Check if GPU is available
	if torch.cuda.is_available():
		device_num = 0
	else:
		device_num = -1
	
	print('\nUsing device number: ', device_num)

	# Clean cache memory
	gc.collect()
	torch.cuda.empty_cache()

	# Move model and loss function to GPU device if available
	if device_num >= 0:
		model.cuda(device_num)
		crit.cuda(device_num)


	# ********** GET OPTIMIZER **********
	optimizer = get_optimizer(model, config)

	# ********** INSTANTIATE TENSORBOARD **********
	subject_title = config.research_subject

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	writer = SummaryWriter('../tensorboard/'+subject_title+'/tests')

	title = subject_title + '_' + config.research_num


	# ********** TRAIN MODEL **********
	start_time = timeit.default_timer()

	trainer.train(
		model=model,
		crit=crit,
		optimizer=optimizer,
		train_loader=train,
		valid_loader=valid,
		n_epochs=config.n_epochs,
		writer=writer,
		title=title
	)

	end_time = (timeit.default_timer() - start_time) / 60.0

	print('training time: ', end_time)


	# ********** SAVE & BRING MODEL **********
	mu.saveModel(subject_title, title, model)
	# mu.graphModel(train, model, writer, device)

	model = mu.getModel(subject_title, title)


	# ********** TEST MODEL **********
	test_loss, test_acc = tester.test(
		model=model,
		crit=crit,
		test_loader=test,
		title=title,
	)

	print('test loss: ', test_loss)
	print('test acc: ', test_acc)




if __name__ == '__main__':
	config = define_argparser()
	main(config)
