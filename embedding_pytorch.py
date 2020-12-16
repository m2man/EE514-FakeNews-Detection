import torch
import time
import numpy as np
import torch.nn as nn
device = torch.device('cuda:0')
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from myfunctions import calculate_metric

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")

class EmbeddingDataset(Dataset):
    # Generate image sgg dataset for the retrieval stage
    def __init__(self, df, col_name, list_vocab):
        self.df = df
        self.list_vocab = list_vocab
        self.col_name = col_name
        
    def __getitem__(self, i):
        headline = self.df[self.col_name][i].lower()
        label = self.df.is_sarcastic[i]
        tokens = headline.split()
        sample = []
        for tok in tokens:
            index = self.list_vocab.index(tok) 
            sample.append(index)
        sample = torch.LongTensor(sample)
        return sample, label
        
    def __len__(self):
        return(len(self.df))

def generate_batch(batch):
    label = torch.FloatTensor([entry[1] for entry in batch])
    text = [entry[0] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def make_EmbeddingDataLoader(dataset, **args):
    data = DataLoader(dataset, collate_fn=generate_batch, **args)
    return data

class HeadlineModel(nn.Module):
    def __init__(self, vocab_size, init_weight=None, embed_dim=300, layers_dim=[300, 1], batchnorm=True, dropout=None):
        '''
        init_weight is numpy matrix [vocab_size, embed_dim]
        '''
        super(HeadlineModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        if layers_dim[0] != embed_dim:
            input_dim = embed_dim
            hidden_dim = layers_dim[0:-1]
        else:
            input_dim = layers_dim[0]
            hidden_dim = layers_dim[1:-1]
        output_dim = layers_dim[-1]
        self.fc = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                      batchnorm=batchnorm, dropout=dropout)
        self.init_weight = init_weight
        self.init_weights()

    def init_weights(self):
        if self.init_weight is not None:
            self.embedding.weight.data.copy_(torch.FloatTensor(self.init_weight).to(device));
        else:
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange).to(device)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# MLP with batchnorm and dropout
class MLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=[], output_dim=300, batchnorm=True, dropout=None):
        '''
        dropout is None or an float [0,1] --> normally 0.5
        activate_fn = 'swish' - 'relu' - 'leakyrelu'
        perform_at_end = True --> apply batchnorm, relu, dropout at the last layer (output)
        hidden_dim is a list indicating unit in hidden layers --> e.g. [1024, 512, 256]
        '''
        super(MLP, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim + [output_dim]
        self.numb_layers = len(self.hidden_dim)
        
        # print(f"Hidden dim: {self.hidden_dim}")
        
        self.linear = torch.nn.ModuleList()
        for idx, numb in enumerate(self.hidden_dim):
            if idx == 0:
                self.linear.append(nn.Linear(input_dim, numb))
            else:
                self.linear.append(nn.Linear(self.hidden_dim[idx-1], numb))
                
        self.batchnorm = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()   
        
        for idx in range(self.numb_layers-1):
            if batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(num_features=self.hidden_dim[idx]))
            else:
                self.batchnorm.append(nn.Identity())              
            if dropout is not None:
                self.dropout.append(nn.Dropout(dropout))
            else:
                self.dropout.append(nn.Identity())
                
        # print(f"Summary MLP: No Linear: {len(self.linear)} --- No BN: {len(self.batchnorm)} --- No DO: {len(self.dropout)}")
              
    def forward(self, x):
        # x should has format of [1, dim]
        # print(f'MLP Network input: {x.shape} --- numb_layer: {self.numb_layers}')
        for i in range(self.numb_layers-1):
            x = self.linear[i](x)
            x = self.batchnorm[i](x)
            x = self.activate(x)
            x = self.dropout[i](x)
        x = self.linear[self.numb_layers-1](x)
        x = self.sigmoid(x)
        return x

class ModelEmbedding():
    def __init__(self, vocab_size, datasetTrain, datasetVal=None, init_weight=None, batch_size=512, optimizer_choice='adam', init_lr=0.001, layers=[300,1], weight_decay=1e-5, dropout=None, batchnorm=True, checkpoint=None, model_name='EMB'):
        super(ModelEmbedding, self).__init__()
        '''
        layers include input - hidden - output
        '''
        self.model = HeadlineModel(vocab_size=vocab_size, init_weight=init_weight, embed_dim=300, 
                                   layers_dim=layers, batchnorm=batchnorm, dropout=dropout)
        self.model = self.model.to(device)
        self.model_name = model_name
        self.dataloaderTrain = make_EmbeddingDataLoader(datasetTrain, batch_size=batch_size)
        if datasetVal is not None:
            self.dataloaderVal = make_EmbeddingDataLoader(datasetVal, batch_size=batch_size)
        else:
            self.dataloaderVal = None
            
        # self.params = list(self.model.parameters())
        if optimizer_choice.lower() == 'adam':
            self.optimizer_emb = optim.SparseAdam(list(self.model.embedding.parameters()), \
                                       lr=init_lr, \
                                       betas=(0.9, 0.999), \
                                       eps=1e-08)
            self.optimizer_fc = optim.Adam(list(self.model.fc.parameters()), \
                                       lr=init_lr, \
                                       betas=(0.9, 0.999), \
                                       eps=1e-08, \
                                       weight_decay=weight_decay)
        if optimizer_choice.lower() == 'sgd':
            self.optimizer_emb = optim.SparseAdam(list(self.model.embedding.parameters()), \
                                       lr=init_lr, \
                                       betas=(0.9, 0.999), \
                                       eps=1e-08)
            self.optimizer_fc = optim.SGD(list(self.model.fc.parameters()), \
                                       lr=init_lr, \
                                       momentum=0.9, \
                                       weight_decay=weight_decay)
            
        self.checkpoint = checkpoint
        
    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(modelCheckpoint['model_state_dict'])
            self.optimizer_emb.load_state_dict(modelCheckpoint['optimizer_emb_state_dict'])
            self.optimizer_fc.load_state_dict(modelCheckpoint['optimizer_fc_state_dict'])
        else:
            print("TRAIN FROM SCRATCH")    
            
    # ---------- RUN TRAIN ---------
    def train(self, numb_epoch=100):
        self.load_trained_model()
        
        # scheduler = ReduceLROnPlateau(self.optimizer_emb, factor = 0.5, patience=3, mode = 'min', verbose=True, min_lr=1e-4)
        
        ## LOSS FUNCTION ##
        loss = nn.BCELoss()
        loss = loss.to(device)
        
        ## REPORT ##
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        self.timestampLaunch = timestampDate + '-' + timestampTime
        
        ## TRAIN THE NETWORK ##
        lossMIN = 100000
        accMax = 0
        flag = 0
        count_change_loss = 0
        
        for epochID in range (numb_epoch):
            print(f"Training {epochID}/{numb_epoch-1}")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            lossTrain = self.train_epoch(loss)
            if self.dataloaderVal is not None:
                metricsVal = self.val_epoch(loss)
                lossVal = metricsVal['loss']
                accVal = metricsVal['accuracy']
                aucVal = metricsVal['auc']
                precisionVal = metricsVal['precision']
                recallVal = metricsVal['recall']
                f1Val = metricsVal['f1']
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
           
            # scheduler.step(lossVal+2*(1-accVal))
            info_txt = f"Epoch {epochID + 1}/{numb_epoch} [{timestampEND}]"
            
            if lossVal < lossMIN or accVal > accMax:
                count_change_loss = 0
                if lossVal < lossMIN:
                    lossMIN = lossVal
                if accVal > accMax:
                    accMax = accVal
                torch.save({'epoch': epochID, \
                            'model_state_dict': self.model.state_dict(), \
                            'optimizer_emb_state_dict': self.optimizer_emb.state_dict(), \
                            'optimizer_fc_state_dict': self.optimizer_fc.state_dict(), \
                            'best_loss': lossMIN, 'best_acc': accMax}, f"{self.model_name}-{self.timestampLaunch}.pth.tar")
                
                info_txt = info_txt + f" [SAVE]"
            else:
                count_change_loss += 1 
            info_txt = info_txt + f"\nAccVal: {accVal}\nAUCVal: {aucVal}\nPrecision: {precisionVal}\nRecall: {recallVal}\nF1Val: {f1Val}\nLossVal: {lossVal}\nLossTrain: {lossTrain}\n----------\n"
            
            print(info_txt)
            
            with open(f"{self.model_name}-{self.timestampLaunch}-REPORT.log", "a") as f_log:
                f_log.write(info_txt)
                
            if count_change_loss >= 25:
                print(f'Early stopping: {count_change_loss} epoch not decrease the loss')
                break
                
    # ---------- TRAINING 1 EPOCH ---------
    def train_epoch(self, loss):
        self.model.train()
        loss_report = 0
        count = 0
        numb_iter = len(self.dataloaderTrain)
        print(f"Total iteration: {numb_iter}")
        for batchID, (batch_sample, batch_offset, batch_label) in enumerate(self.dataloaderTrain):
            batch_sample = batch_sample.to(device)
            batch_offset = batch_offset.to(device)
            batch_label = batch_label.to(device)
            preds = self.model(batch_sample, batch_offset)
            lossvalue = loss(preds.squeeze(), batch_label.squeeze())
            self.optimizer_emb.zero_grad()
            self.optimizer_fc.zero_grad()
            lossvalue.backward()
            self.optimizer_emb.step()
            self.optimizer_fc.step()

            loss_report += lossvalue.item()
            count += 1
            if (batchID+1) % 100 == 0:
                print(f"Batch Idx: {batchID+1} / {numb_iter}: Loss Train {loss_report/count}")
                
        return loss_report/count
    
    def val_epoch(self, loss):
        self.model.eval()
        with torch.no_grad():
            outGT = torch.FloatTensor().to(device)
            outPRED = torch.FloatTensor().to(device)
            loss_report = 0
            count = 0
            for batchID, (batch_sample, batch_offset, batch_label) in enumerate(self.dataloaderVal):
                batch_sample = batch_sample.to(device)
                batch_offset = batch_offset.to(device)
                batch_label = batch_label.to(device)
                preds = self.model(batch_sample, batch_offset)
                lossvalue = loss(preds.squeeze(), batch_label.squeeze())
                loss_report += lossvalue.item()
                count += 1
                outGT = torch.cat((outGT, batch_label), 0)
                outPRED = torch.cat((outPRED, preds), 0)
                
            outGTnp = outGT.cpu().numpy().squeeze()
            outPREDnp = outPRED.cpu().numpy().squeeze()
            metrics = calculate_metric(outGTnp, outPREDnp)
            metrics['loss'] = loss_report/count
        return metrics
    
    def evaluate(self, datald):
        self.model.eval()
        with torch.no_grad():
            outGT = torch.FloatTensor().to(device)
            outPRED = torch.FloatTensor().to(device)
            for batchID, (batch_sample, batch_offset, batch_label) in enumerate(datald):
                batch_sample = batch_sample.to(device)
                batch_offset = batch_offset.to(device)
                batch_label = batch_label.to(device)
                preds = self.model(batch_sample, batch_offset)
                outGT = torch.cat((outGT, batch_label), 0)
                outPRED = torch.cat((outPRED, preds), 0)
                
            outGTnp = outGT.cpu().numpy().squeeze()
            outPREDnp = outPRED.cpu().numpy().squeeze()
            metrics = calculate_metric(outGTnp, outPREDnp)
        return metrics
