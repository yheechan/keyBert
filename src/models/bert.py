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


class Bert(nn.Module):
	def __init__(self, dropout=0.0):
		super(Bert, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-uncased') 
		self.dp = nn.Dropout(dropout)
		self.classification = Classification(dropout=dropout)

	
	def forward(self, input_id, mask):
		_, cls_vector = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
		# |cls_vector| = (batch_size, 768) -> output of [CLS] token

		dp = self.dp(cls_vector)
		y_hat = self.classification(dp)

		return y_hat
