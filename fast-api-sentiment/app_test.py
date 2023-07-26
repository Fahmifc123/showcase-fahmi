# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
# import uvicornc
from fastapi import FastAPI
from datetime import datetime
import nltk
# nltk.download('punkt')
# nltk.download('omw-1.4')
# nltk.download('wordnet')

import numpy as np
import timeit

from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

#v3 bert

from transformers import pipeline
import transformers

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from nltk.tokenize import TweetTokenizer

import torch

###
# Forward Function
###

# # Forward function for sequence classification
# def forward_sequence_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
#     # Unpack batch data
#     if len(batch_data) == 3:
#         (subword_batch, mask_batch, label_batch) = batch_data
#         token_type_batch = None
#     elif len(batch_data) == 4:
#         (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
#     # Prepare input & label
#     subword_batch = torch.LongTensor(subword_batch)
#     mask_batch = torch.FloatTensor(mask_batch)
#     token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
#     label_batch = torch.LongTensor(label_batch)
            
#     if device == "cuda":
#         subword_batch = subword_batch.cuda()
#         mask_batch = mask_batch.cuda()
#         token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
#         label_batch = label_batch.cuda()

#     # Forward model
#     outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
#     loss, logits = outputs[:2]
    
#     # generate prediction & label list
#     list_hyp = []
#     list_label = []
#     hyp = torch.topk(logits, 1)[1]
#     for j in range(len(hyp)):
#         list_hyp.append(i2w[hyp[j].item()])
#         list_label.append(i2w[label_batch[j][0].item()])
        
#     return loss, list_hyp, list_label

# # Forward function for word classification
# def forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
#     # Unpack batch data
#     if len(batch_data) == 4:
#         (subword_batch, mask_batch, subword_to_word_indices_batch, label_batch) = batch_data
#         token_type_batch = None
#     elif len(batch_data) == 5:
#         (subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, label_batch) = batch_data
    
#     # Prepare input & label
#     subword_batch = torch.LongTensor(subword_batch)
#     mask_batch = torch.FloatTensor(mask_batch)
#     token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
#     subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
#     label_batch = torch.LongTensor(label_batch)

#     if device == "cuda":
#         subword_batch = subword_batch.cuda()
#         mask_batch = mask_batch.cuda()
#         token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
#         subword_to_word_indices_batch = subword_to_word_indices_batch.cuda()
#         label_batch = label_batch.cuda()

#     # Forward model
#     outputs = model(subword_batch, subword_to_word_indices_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
#     loss, logits = outputs[:2]
    
#     # generate prediction & label list
#     list_hyps = []
#     list_labels = []
#     hyps_list = torch.topk(logits, k=1, dim=-1)[1].squeeze(dim=-1)
#     for i in range(len(hyps_list)):
#         hyps, labels = hyps_list[i].tolist(), label_batch[i].tolist()        
#         list_hyp, list_label = [], []
#         for j in range(len(hyps)):
#             if labels[j] == -100:
#                 break
#             else:
#                 list_hyp.append(i2w[hyps[j]])
#                 list_label.append(i2w[labels[j]])
#         list_hyps.append(list_hyp)
#         list_labels.append(list_label)
        
#     return loss, list_hyps, list_labels

# # Forward function for sequence multilabel classification
# def forward_sequence_multi_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
#     # Unpack batch data
#     if len(batch_data) == 3:
#         (subword_batch, mask_batch, label_batch) = batch_data
#         token_type_batch = None
#     elif len(batch_data) == 4:
#         (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
#     # Prepare input & label
#     subword_batch = torch.LongTensor(subword_batch)
#     mask_batch = torch.FloatTensor(mask_batch)
#     token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
#     label_batch = torch.LongTensor(label_batch)
            
#     if device == "cuda":
#         subword_batch = subword_batch.cuda()
#         mask_batch = mask_batch.cuda()
#         token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
#         label_batch = label_batch.cuda()

#     # Forward model
#     outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
#     loss, logits = outputs[:2] # logits list<tensor(bs, num_label)> ~ list of batch prediction per class 
    
#     # generate prediction & label list
#     list_hyp = []
#     list_label = []
#     hyp = [torch.topk(logit, 1)[1] for logit in logits] # list<tensor(bs)>
#     batch_size = label_batch.shape[0]
#     num_label = len(hyp)
#     for i in range(batch_size):
#         hyps = []
#         labels = label_batch[i,:].cpu().numpy().tolist()
#         for j in range(num_label):
#             hyps.append(hyp[j][i].item())
#         list_hyp.append([i2w[hyp] for hyp in hyps])
#         list_label.append([i2w[label] for label in labels])
        
#     return loss, list_hyp, list_label

import sys

#from utils.forward_fn import forward_sequence_classification
#from utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}

app = FastAPI(
    title="Sentiment Model API",
    description="Sentiment Model API Nolimit V3",
    version="0.3",
)

stop_words_id = stopwords.words("indonesian")
stop_words_en = stopwords.words("english")

# load the sentiment model
# with open(
#     join(dirname(realpath(__file__)), "model/model_sentiment_bert_10ep_combine95rb_nobalance_basep1.pkl"), "rb") as a:
#     model_id = joblib.load(a)

import gzip
import joblib
from os.path import join, dirname, realpath

model_path = join(dirname(realpath(__file__)), "model/model_sentiment_bert_10ep_combine95rb_nobalance_basep1.pkl")
compressed_path = join(dirname(realpath(__file__)), "model/model_sentiment_bert_10ep_combine95rb_nobalance_basep1.pkl.gz")

# Mengompresi file .pkl menggunakan gzip
with open(model_path, "rb") as file_in:
    with gzip.open(compressed_path, "wb") as file_out:
        file_out.write(file_in.read())

# Memuat model dari file terkompresi
model_id = joblib.load(compressed_path)

# Menghitung ukuran file sebelum dan setelah kompresi gzip
original_size = os.path.getsize(model_path)
compressed_size = os.path.getsize(compressed_path)

print("Ukuran file asli:", original_size, "bytes")
print("Ukuran file terkompresi:", compressed_size, "bytes")
print("Rasio kompresi: {:.2f}%".format((compressed_size / original_size) * 100))

#english model
with open(
    join(dirname(realpath(__file__)), "model/model_en_new_oktober.pkl"), "rb") as a:
    model_en = joblib.load(a)

def text_cleaning_id(text, remove_stop_words=False):
    # Clean the text 
    # text = re.sub(r"[^A-Za-z0-9]", " ", text) #1
    # text = re.sub(r"\'s", " ", text) #2
    # text = re.sub(r"http\S+", " link ", text) #3
    # text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # 4

    # # Remove punctuation from text
    # text = "".join([c for c in text if c not in punctuation]) #5

    text = text.lower() #lowercase atau case folding
    text = re.sub('@[^\s]+', '', text) #remove username
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text) # remove URLs
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) 
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)

    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        text = text.split()
        text = [w for w in text if not w in stop_words_id] #6
        text = " ".join(text)
        text = text.lower()

    # Optionally, shorten words to their stems
    # if lemmatize_words:
    #     text = text.split()
    #     lemmatizer = WordNetLemmatizer()
    #     lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    #     text = " ".join(lemmatized_words)
    # Return a list of words
    return text

def text_cleaning_en(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    # text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # text = re.sub(r"\'s", " ", text)
    # text = re.sub(r"http\S+", " link ", text)
    # text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    # # Remove punctuation from text
    # text = "".join([c for c in text if c not in punctuation])

    text = text.lower() #lowercase atau case folding
    text = re.sub('@[^\s]+', '', text) #remove username
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text) # remove URLs
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) 
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)

    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        text = text.split()
        text = [w for w in text if not w in stop_words_en]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    # Return a list of words
    return text
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

class Item(BaseModel):
    text: str

@app.post("/sentiment/v3/predict-id/")
def predict_id(review: Item):

    try:
        cleaned_review = text_cleaning_id(review.text)
        kata = cleaned_review.split()
        jumlah_kata = len(kata)

        if jumlah_kata > 3:
            
            subwords = tokenizer.encode(cleaned_review)
            subwords = torch.LongTensor(subwords).view(1, -1).to(model_id.device)
            longtensor_duration = (datetime.now() - start3).total_seconds()
         
            logits = model_id(subwords)[0]
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
            probability = F.softmax(logits, dim=-1).squeeze()[label].item()

        else:
            label = 1
            probability = 0.5

    except:
        label = 1
        probability = 0.9

    result = {"prediction": i2w[label], "probability": probability}
    return result

@app.post("/sentiment/v3/predict-en/")
def predict_en(review: Item):

    cleaned_review = text_cleaning_en(review.text)
    print(cleaned_review)
    
    # perform prediction
    prediction = model_en.predict([cleaned_review])
    output = int(prediction[0])
    probas = model_en.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "neutral", 1: "negative", 2: "positive"}
    
    # show results
    result = {"prediction": sentiments[output], "probability": output_probability}
    return result
