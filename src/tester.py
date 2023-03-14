import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pandas as pd

def test(
	model,
	crit,
	test_loader=None,
	title=None,
):

	# check for available gpu
	if torch.cuda.is_available():
		device_num = 0
		print('\nUsing device number: 0')
	else:
		device_num = -1
		print('\nUsing device number: -1')
	
	# Pass model to GPU device if it is necessary
	if device_num >= 0:
		model.cuda(device_num)
		crit.cuda(device_num)
	
	model.eval()

	tot_test_loss = []
	tot_test_acc = []

	device = next(model.parameters()).device

	# initiate for confusion matrix
	tot_preds = torch.empty(0).to(device)
	tot_labels = torch.empty(0).to(device)

	# For each batch in our test set...
	for batch_input, batch_label in tqdm(test_loader):

		# *** LOAD BATCH TO DEVICE ***
		batch_label = batch_label.to(device)
		# |battch_label = (batch_size)

		mask = batch_input['attention_mask'].to(device)
		# |mask| = (batch_size, max_length)

		input_id = batch_input['input_ids'].squeeze(1).to(device)
		# |input_id| = (batch_size, max_length)

		# *** PREDICT ***
		y_hat = model(input_id, mask)
		# |y_hat| = (batch_size, binary(2))

		# *** CALCULATE LOSS ***
		loss = crit(y_hat, batch_label)
		tot_test_loss.append(loss.item())

		# *** CALCULATE ACCURACY ***
		pred = y_hat.argmax(1).flatten()
		acc = (pred == batch_label).cpu().numpy().mean() * 100
		tot_test_acc.append(acc)

		# save for confusion matrix
		tot_preds = torch.cat((tot_preds, pred))
		tot_labels = torch.cat((tot_labels, batch_label))

	# Calculate the average loss over the entire training data
	test_loss = np.mean(tot_test_loss)
	test_acc = np.mean(tot_test_acc)

	cm = metrics.classification_report(tot_labels.cpu(), tot_preds.cpu(), output_dict=True)
	cm_df = pd.DataFrame.from_dict(cm).transpose()
	cm_df.to_csv('../confusion_matrix/'+title+'.csv')

	return test_loss, test_acc
