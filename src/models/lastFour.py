import torch
from torch import nn
from transformers import BertModel

class Classification(nn.Module):
	def __init__(self, dropout=0.0):
		super(Classification, self).__init__()

		self.fc1 = nn.Linear(768, 2)
		self.relu1 = nn.ReLU()
		self.dp1 = nn.Dropout(dropout)
	
	def forward(self, input_vector):
		
		y_hat = self.dp1(self.relu1(self.fc1(input_vector)))

		return y_hat


class LastFour(nn.Module):
	def __init__(self, dropout=0.0):
		super(LastFour, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-uncased',
												return_dict=True,
												output_hidden_states=True)

		self.dp = nn.Dropout(dropout)

		self.lstm = nn.LSTM(
			input_size = 3072,
			hidden_size = 768,
			num_layers = 2,
			dropout = dropout,
			batch_first = True,
			bidirectional = True,
		)

		self.fc = nn.Linear(1536, 768)

		self.classification = Classification(dropout=dropout)

	
	def forward(self, input_id, mask):
		last_hidden_state, pooler_output, hidden_states = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
		# |last_hidden_state| = (batch_size, sequence_length, 768) -> output of each token in sequence
		# |pooler_output| = (batch_size, 768) -> output of [CLS] token
		# |hidden_states[i]| = (batch_size, sequence_length, 768) -> hidden state of each encoding layer in BERT model


		last_four = hidden_states[-4:]
		# |last_four| = (batch_size, sequence_length, 768)


		last_four_cls = []
		for i in range(len(last_four)):
			last_four_cls.append(last_four[i][:, 0, :])
			# last_four_cls.append(torch.unsqueeze(last_four[i][:, 0, :], dim=1))
		# |last_four_cls[i]| = (batch_size, 768) -> hidden state from cls of last four encoder of BERT

		cat_four = torch.cat(last_four_cls, dim=1)

		cat_four = torch.unsqueeze(cat_four, dim=1)
		# |cat_four| = (batch_size, 1, 3072) -> hidden state of cls from last four encoder concatenated together

		output, (hidden, cell) = self.lstm(cat_four)
		# |hidden| = (lstm_layer*bidirectional, batch_size, 768)

		# x = torch.cat([hidden[0], hidden[1], hidden[2], hidden[3]], dim=1)
		x = torch.cat([hidden[2], hidden[3]], dim=1)
		# |x| = (batch_size, 768*4)

		# dp = self.dp(x)
		fc = self.fc(x)

		y_hat = self.classification(fc)

		return y_hat
