#!/usr/bin/env python
# coding: utf-8

print("Importation...")

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import os
import pprint
import re
import json

print("Modules importés")


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output


print("Chargement du modèle")

bert = BertModel.from_pretrained('bert-base-uncased')

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
model.load_state_dict(torch.load('tut6-model.pt', torch.device('cpu')))#'cuda')))

print("Modèle chargé")

#Fonction de prédiction
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()



from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

device = torch.device('cpu')#cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)
#print(predict_sentiment(model, tokenizer, "Wow i'm fucking great"))

#MONGO
from pymongo import MongoClient
MONGO_HOST = "xxxx"
client = MongoClient(MONGO_HOST)
db = client.twitterdb
collection = db.twitter_search

print("Suppresion des tweets qui ne sont pas en anglais")

collection.delete_many({"lang":{"$ne":"en"}})
print("Ajout de la colonne prediction")
collection.update_many( { "Prediction": None }, { "$set": { "Prediction": 2.0} } )

print("Chargement des données depuis la DB...")
i = 0
for search in collection.find({"lang":"en", "Prediction": 2.0},{"text":1}):

    myquery = { "_id": search["_id"] }
    
    #Filtrage du tweet
    search = search["text"].replace('\n','')#predict_sentiment(model, tokenizer, "search"))
    search = re.sub(r'@\S+', '', search)
    search = re.sub(r'http\S+', '', search)
    search = search.replace('RT', '')
    unwanted_char = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
                           "]+", flags = re.UNICODE)
    search = unwanted_char.sub(r'',search)

    #Ajout de la prédiction dans la bdd
    newvalues = { "$set": { "Prediction": predict_sentiment(model, tokenizer, search) } }
    collection.update_one(myquery, newvalues)
    i = i + 1
print("Données mises à jour.", i, "tweets analysés.")


        