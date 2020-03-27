import sys
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_transformers import BertTokenizer, BertModel  #bert
from pytorch_transformers import RobertaTokenizer, RobertaModel  #roberta
from torch.utils import data
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from utils import BertDataset,pad_batch
from rnn import rnn,Config as rnn_config
from rnn_attn import rnn_attn,Config as rnn_attn_config
from rnn_cg import rnn_cg,Config as rnn_cg_config


def get_f1_predict(model, loader,device,use_categorys = False):
    # get f1 score and predictions
    predict = None
    label = None
    model.eval()
    with torch.no_grad():
        for step, (data) in enumerate(loader):
            tokens, segments, masks,categorys =  [t.to(device) for t in data if t is not None]
            if use_categorys:
                output = model(input_ids=tokens, token_type_ids=segments, attention_mask=masks,categorys=categorys)[0]
            else:
                output = model(input_ids=tokens, token_type_ids=segments, attention_mask=masks)[0]
            if predict is None:
                predict = output
            else:
                predict = torch.cat((predict, output))
            print(output.shape)
    return predict
if __name__ == '__main__':
    print(torch.cuda.get_device_name(0))
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #bert
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base') #roberta
    testset = BertDataset('test.csv', tokenizer,max_length=256, train = False)
    test_loader = DataLoader(testset, batch_size=4, collate_fn=pad_batch)
    # load trained model
    model_name= ['rnn_attn.pkl','rnn_cg.pkl']
    model_structure = ['rnn_attn','rnn_cg']
    use_categorys = [False,True]
    pred = None
    for s,n,u in zip(model_structure,model_name,use_categorys):
        print(n)
        if s == 'rnn_cg':
            model = rnn_cg(rnn_cg_config())
        elif s == 'rnn_attn':
            model = rnn_attn(rnn_attn_config())
        elif s == 'rnn':
            model = rnn(rnn_config())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load('rnn_attn.pkl'),strict=False)
        model = model.to(device)

        if pred is None:
            output = get_f1_predict(model, test_loader,device,use_categorys=u)
            # pred = (output-torch.min(output,0))/(torch.max(output,0)[0]-torch.min(output,0)[0])
            pred = output
        else:
            output = get_f1_predict(model, test_loader,device,use_categorys=u)
            # pred += (output-torch.min(output,0))/(torch.max(output,0)[0]-torch.min(output,0)[0])
            pred = output
    pred = pred/len(model_list)
    max_ids = torch.max(pred, 1)[1]
    max_ids = max_ids.unsqueeze(1)
    pred = pred.scatter_(1, max_ids, 1.)
    pred = pred.ge(0)
    pred = pred.tolist()
    reader = csv.reader(open('task2_sample_submission.csv'))
    lines = [l for l in reader]
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == 1.:
                lines[i+1][j+1] = '1'
    with open('out.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lines)