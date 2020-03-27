import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
class BertDataset(Dataset):
	# get dataset for bert, and add special token, return token,segment,label
	# label eg. [0,1,1,0] 1 for correct class
	def __init__(self, file, tokenizer,max_length=256, train=True):
		self.train = train
		self.tokenizer = tokenizer
		self.data = pd.read_csv(file, encoding='utf-8')
		self.data.drop(['Id', 'Title', 'Authors', 'Created Date'], axis=1, inplace=True) #Abstract Task 2
		self.label_id = {'THEORETICAL':0, 'ENGINEERING':1, 'EMPIRICAL':2, 'OTHERS':3}
		self.category_id=json.load(open('category.json',encoding='utf-8'))
		self.len = len(self.data)
		self.max_length = max_length
	def __getitem__(self, index):
		if self.train:
			text, category, label = self.data.iloc[index].values
			category = category.split('/')
			label = label.split()
			category_tensor = torch.zeros(141)
			label_tensor = torch.zeros(4)
			for l in category:
				try:
					category_tensor[self.category_id[l]] = 1.
				except:
					continue
			for l in label:
				label_tensor[self.label_id[l]] = 1.
		else:
			text,category = self.data.iloc[index].values
			category = category.split('/')
			category_tensor = torch.zeros(141)
			for l in category:
				try:
					category_tensor[self.category_id[l]] = 1.
				except:
					continue
			label_tensor = None

		text = text.replace('$$$', ' ')
		text = self.tokenizer.tokenize(text)
		if len(text) > self.max_length-2:
			text = text[:self.max_length-2]
		# text = ['[CLS]'] + text + ['[SEP]']  #bert
		text = ['<s>'] + text + ['</s>']  #roberta
		# while len(text) < self.max_length:
		# 	text += ['[PAD]']
		text_ids = self.tokenizer.convert_tokens_to_ids(text)
		token_tensor = torch.tensor(text_ids)
		segment_tensor = torch.tensor([0]*len(text_ids))

		return (token_tensor, segment_tensor, label_tensor,category_tensor)

	def __len__(self):
		return self.len
def pad_batch(batch):
	# collate_fn for Dataloader, pad sequence to same length and get mask tensor
	if batch[0][2] is not None:
		(tokens, segments, labels, categorys) = zip(*batch)
		labels = torch.stack(labels)
		categorys = torch.stack(categorys)
	else:
		(tokens, segments, labels, categorys) = zip(*batch)
		labels = None
		categorys = torch.stack(categorys)
	t_len = [len(x) for x in tokens]

	tokens_pad = pad_sequence(tokens, batch_first=True)
	segments_pad = pad_sequence(segments, batch_first=True)
	masks = torch.zeros(tokens_pad.shape)
	for i in range(len(masks)):
		masks[i][:t_len[i]] = 1

	return tokens_pad, segments_pad, masks,categorys, labels
def get_f1_predict(model, loader,device, f1=True,use_categorys = False):
	# get f1 score and predictions
	true_pos = 0
	pred_true = 0
	target_true = 0
	predict = None
	model.eval()
	with torch.no_grad():
		for step, (data) in enumerate(loader):
			if use_categorys:
				if f1:
					tokens, segments, masks ,categorys, labels = [t.to(device) for t in data]
				else:
					tokens, segments, masks ,categorys = [t.to(device) for t in data if t is not None]
				output = model(input_ids=tokens, token_type_ids=segments, attention_mask=masks,categorys=categorys)[0]
			else:
				if f1:
					tokens, segments, masks , labels = [t.to(device) for t in data]
				else:
					tokens, segments, masks  = [t.to(device) for t in data if t is not None]
				output = model(input_ids=tokens, token_type_ids=segments, attention_mask=masks)[0]
			max_ids = torch.max(output, 1)[1]
			max_ids = max_ids.unsqueeze(1)
			output = output.scatter_(1, max_ids, 1.)
			pred = output.ge(0.5)

			if f1:
				pred_true +=  torch.sum(pred).item()
				target_true += torch.sum(labels).item()
				true_pos += torch.sum(pred.type(torch.float)*labels).item()

			if predict is None:
				predict = pred
			else:
				predict = torch.cat((predict, pred))
	if f1:
		precision = true_pos/target_true
		recall = true_pos/pred_true
		return predict, 2*precision*recall/(precision+recall)
	return predict
def draw_chart(chart_data,outfile_name):
	plt.figure()
	plt.rcParams['figure.figsize'] = (12.0, 6.0)
	plt.rcParams['savefig.dpi'] = 200
	plt.rcParams['figure.dpi'] = 200
	plt.plot(chart_data['epoch'],chart_data['tarin_loss'],label='tarin_loss')
	plt.plot(chart_data['epoch'],chart_data['train_f1'],label='train_f1')
	plt.plot(chart_data['epoch'],chart_data['val_f1'],label='val_f1')
	plt.grid(True,axis="y",ls='--')
	plt.legend(loc= 'best')
	plt.xlabel('epoch',fontsize=20)
	plt.yticks(np.linspace(0,1,11))
	plt.savefig(outfile_name+'.jpg')
	with open(outfile_name+'.json','w') as file_object:
		json.dump(chart_data,file_object)
	plt.close('all')