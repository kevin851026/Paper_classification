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
from utils import BertDataset,pad_batch,get_f1_predict,draw_chart
class Config():
    def __init__(self):
        # model hyperparameters
        self.num_labels = 4
        self.embedding_size = 768
        self.rnn_hidden = 256+128
        self.rnn_layer = 1
        self.rnn_bidirect = True
        self.forward_size = 128
        self.drop_pro = 0.1
        self.category_size = 141

        #training parameters
        self.max_length = 256
        self.batch_size = 16
        self.epoch = 50
        self.learning_rate = 0.000008
class rnn_attn(nn.Module):
    def __init__(self,config):
        super(rnn_attn, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-cased") #bert
        self.config = config
        self.rnn_dim = config.rnn_hidden*2 if config.rnn_bidirect else config.rnn_hidden
        self.bert = RobertaModel.from_pretrained("roberta-base")  #roberta
        self.lstm = nn.Sequential(
                        nn.LSTM(input_size=config.embedding_size,
                                 hidden_size=config.rnn_hidden,
                                 num_layers=config.rnn_layer,
                                 bidirectional=config.rnn_bidirect,
                                 batch_first=True)
                    )
        self.w_qs = nn.Linear(self.rnn_dim, 128)
        self.w_ks = nn.Linear(self.rnn_dim, 128)
        self.w_vs = nn.Linear(self.rnn_dim, self.rnn_dim)
        self.dropout = nn.Dropout(config.drop_pro)
        self.layer_norm = nn.LayerNorm(self.rnn_dim)
        self.fc1 =  nn.Sequential(
                        nn.Linear(self.rnn_dim,config.forward_size),
                        nn.ReLU(),
                        nn.Dropout(config.drop_pro)
                    )
        self.fc2 =  nn.Sequential(
                        nn.Linear(config.category_size,config.forward_size),
                        nn.ReLU(),
                        nn.Dropout(config.drop_pro)
                    )
        self.classifier =  nn.Sequential(
                        nn.Linear(config.forward_size,config.num_labels),
                        nn.Dropout(config.drop_pro)
                    )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, \
                position_ids=None, head_mask=None,categorys=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask)
        batch_size = outputs[0].shape[0]
        lstm,(h,c) = self.lstm(outputs[0]) # [batch_size,280,768] -> [batch_size,280,300*2]

        # attention
        q=self.w_qs(lstm)
        k=self.w_ks(lstm)
        v=self.w_vs(lstm)
        attn = torch.matmul(q,k.transpose(1,2))
        attn = attn/np.power(self.config.max_length,0.5)
        attn = attn.masked_fill((attention_mask-1).type(torch.uint8).view(batch_size,1,-1),-np.inf)
        attn = F.softmax(attn,2)
        feats = torch.matmul(attn,v).sum(dim=1)
        feats = self.dropout(feats)
        feats = self.layer_norm(feats)

        attn_logits = self.fc1(feats)
        logits = self.classifier(attn_logits)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs

if __name__ == '__main__':
    print(torch.cuda.get_device_name(0))
    # torch.manual_seed(12)
    config = Config()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased') #bert
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base') #roberta

    dataset = BertDataset('train.csv', tokenizer,max_length=config.max_length)
    trainset, valset = data.random_split(dataset, (int(len(dataset)*0.8), int(len(dataset)*0.2)))
    testset = BertDataset('test.csv', tokenizer,max_length=config.max_length, train = False)

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(valset, batch_size=config.batch_size, collate_fn=pad_batch)
    test_loader = DataLoader(testset, batch_size=config.batch_size, collate_fn=pad_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained model
    model = rnn_attn(config)
    print(type(model))
    model.load_state_dict(torch.load('rnn_attn.pkl'))
    print(type(model))
    model = model.to(device)
    # freeze layer
    # for p in model.bert.parameters():
    #   p.requires_grad = False
    # for p in model.bert.embeddings.parameters():
    #   p.requires_grad = False

    #num of parameters
    print('total: ',str(sum(p.numel() for p in model.parameters())))
    print('trainable: ',str(sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))))
    print('-'*10)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    chart_data={"tarin_loss":[],"train_f1":[],"val_f1":[],"epoch":[]}
    max_f1=0
    model_name='rnn_attn'
    for epoch in range(config.epoch):
      train_ls = 0
      train_step = 0
      model.train()
      batch =iter(train_loader).next()
      for step, (batch) in enumerate(train_loader):
          tokens, segments, masks,categorys, labels = [t.to(device) for t in batch]
          output = model(input_ids=tokens, token_type_ids=segments, attention_mask=masks,categorys=categorys, labels=labels)
          optimizer.zero_grad()
          loss = output[0]
          loss.backward()
          optimizer.step()

          train_ls += loss.item()
          train_step = step
      pred, tf1 = get_f1_predict(model, train_loader,device,use_categorys=True)
      pred, vf1 = get_f1_predict(model, val_loader,device,use_categorys=True)
      print('Epoch: ' + str(epoch) + ' train loss: ' + str(train_ls/(step+1)))
      print('train F1: ' + str(tf1) +' val F1: ' + str(vf1))
      chart_data['epoch'].append(epoch)
      chart_data['tarin_loss'].append(train_ls/(step+1))
      chart_data['train_f1'].append(tf1)
      chart_data['val_f1'].append(vf1)
      draw_chart(chart_data,model_name)

      if epoch+1 >= 20 and vf1 > max_f1 and vf1 > 0.675:
            max_f1 = vf1
            print(epoch)
            torch.save(model.state_dict(), model_name+'.pkl')
            pred = get_f1_predict(model, test_loader,device,use_categorys=True, f1=False)
            pred = pred.tolist()
            print(pred[:10])
            reader = csv.reader(open('task2_sample_submission.csv'))
            lines = [l for l in reader]
            for i in range(len(pred)):
                for j in range(len(pred[i])):
                    if pred[i][j] == 1.:
                        lines[i+1][j+1] = '1'
            with open('out.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(lines)