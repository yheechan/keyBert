import time
import torch
import numpy as np
from tqdm import tqdm

def train(
	model,
	crit,
	optimizer,
	train_loader,
	valid_loader,
	n_epochs,
	writer=None,
	title=None,
):

	best_accuracy = 0

	print('Start training...')
	print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^8} | {'Val Acc':^6} | {'Elapsed':^6}")
	print("-"*80)
	
	device = next(model.parameters()).device

	for epoch_i in range(n_epochs):
	
		# ========= TRAINING =========

		#Tracking time
		t0_epoch = time.time()

		#Put the model into training mode
		model.train()

		tot_train_acc = []
		tot_train_loss = []

		for batch_input, batch_label in tqdm(train_loader):

			# *** LOAD BATCH TO DEVICE ***
			batch_label = batch_label.to(device)
			# |battch_label = (batch_size)

			mask = batch_input['attention_mask'].to(device)
			# |mask| = (batch_size, max_length)

			input_id = batch_input['input_ids'].squeeze(1).to(device)
			# |input_id| = (batch_size, max_length)

			optimizer.zero_grad()

			# *** PREDICT ***
			y_hat = model(input_id, mask)
			# |y_hat| = (batch_size, binary(2))

			# *** CALCULATE LOSS ***
			loss = crit(y_hat, batch_label)
			tot_train_loss.append(loss.item())

			# *** CALCULATE ACCURACY ***
			pred = y_hat.argmax(1).flatten()
			acc = (pred == batch_label).cpu().numpy().mean() * 100
			tot_train_acc.append(acc)

			# *** TRAIN MODEL ***
			loss.backward()
			optimizer.step()

		# Calculate the average loss over the entire training data
		train_loss = np.mean(tot_train_loss)
		train_acc = np.mean(tot_train_acc)


		# ========= EVALUATING =========

		# After the completion of each training epoch,
		# measure the model's performance on validation set.
		val_loss, val_acc = evaluate(
			model,
			crit,
			valid_loader,
			lr_schedular=None,
		)

		if val_acc > best_accuracy:
			best_accuracy = val_acc

		
		# print performance over the entire training data
		time_elapsed = time.time() - t0_epoch
		print(f"{epoch_i + 1:^7} | {train_loss:^12.6f} | {train_acc:^10.6f} | {val_loss:^8.6f} | {val_acc:^6.2f} | {time_elapsed:^6.2f}")

		writer.add_scalars(title + '-Loss',
			{'Train' : train_loss, 'Validation' : val_loss},
			epoch_i + 1)

		writer.add_scalars(title + '-Accuracy',
			{'Train' : train_acc, 'Validation' : val_acc},
			epoch_i + 1)

	writer.flush()

	print('\n')
	print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")


def evaluate(
	model,
	crit,
	valid_loader,
	lr_schedular=None,
):

	# After the completion of each training epoch,
	# measure the model's performance on validation set.

	
	# Put the model to evaluating mode
	model.eval()

	# Tracking variables
	tot_val_loss = []
	tot_val_acc = []

	device = next(model.parameters()).device
	
	# For each batch in our validation set...
	for batch_input, batch_label in tqdm(valid_loader):

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
		tot_val_loss.append(loss.item())

		# *** CALCULATE ACCURACY ***
		pred = y_hat.argmax(1).flatten()
		acc = (pred == batch_label).cpu().numpy().mean() * 100
		tot_val_acc.append(acc)

	# Calculate the average loss over the entire training data
	val_loss = np.mean(tot_val_loss)
	val_acc = np.mean(tot_val_acc)

	return val_loss, val_acc
