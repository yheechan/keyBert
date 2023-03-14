import argparse
import pprint

import data_loader

from models.bert import Bert
import tester
import model_util as mu

import torch, gc
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

	config = p.parse_args()

	return config

def get_model(config):
	model = Bert(dropout=config.dropout)
	return model

def get_crit():
	crit = nn.CrossEntropyLoss()
	return crit

def get_optimizer(model, config):
	optimizer = optim.Adam(model.parameters(), lr=config.lr)
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

	
	# ********** LOSS FUNCTION **********
	crit = get_crit()



	# ********** BRING MODEL **********
	subject_title = config.research_subject
	title = subject_title + '_' + config.research_num

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
