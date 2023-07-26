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
import sys

#from utils.forward_fn import forward_sequence_classification
#from utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader
torch.set_grad_enabled(False)
torch.set_num_threads(1)
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}

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
# compressed_path = join(dirname(realpath(__file__)), "model/model_sentiment_bert_10ep_combine95rb_nobalance_basep1.pkl.gz")

# Mengompresi file .pkl menggunakan gzip
#with open(model_path, "rb") as file_in:
#    with gzip.open(compressed_path, "wb") as file_out:
#        file_out.write(file_in.read())

# Memuat model dari file terkompresi
model_id = joblib.load(model_path)
# model_id = torch.quantization.quantize_dynamic(model_id, {torch.nn.Linear}, dtype=torch.qint8)
# Menghitung ukuran file sebelum dan setelah kompresi gzip
# original_size = os.path.getsize(model_path)
# compressed_size = os.path.getsize(compressed_path)
#
#print("Ukuran file asli:", original_size, "bytes")
#print("Ukuran file terkompresi:", compressed_size, "bytes")
#print("Rasio kompresi: {:.2f}%".format((compressed_size / original_size) * 100))
#
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

class Item(BaseModel):
    text: str

def predict_id(text: str):
    start_req = datetime.now()
    encode_duration = 0
    longtensor_duration = 0
    topk_duration = 0
    prob_duration = 0
    cleaning_duration = 0
    split_duration = 0
    in_duration = 0
    in2_duration = 0
    logits_duration = 0
    try:
        start_in = datetime.now()
        start1 = datetime.now()
        cleaned_review = text_cleaning_id(text)
        cleaning_duration = (datetime.now() - start1).total_seconds()
        print(cleaned_review)
        start6 = datetime.now()
        kata = cleaned_review.split()
        split_duration = (datetime.now() - start6).total_seconds()
        jumlah_kata = len(kata)
        # for _ in kata:
        #    jumlah_kata += 1

        if jumlah_kata > 3:
            start_in2 = datetime.now()
            start2 = datetime.now()                
            subwords = tokenizer.encode(cleaned_review)
            encode_duration = (datetime.now() - start2).total_seconds()
            
            start3 = datetime.now()
            subwords = torch.LongTensor(subwords).view(1, -1).to(model_id.device)
            longtensor_duration = (datetime.now() - start3).total_seconds()
            start7 = datetime.now()
            # print(model_id, subwords)
            print(model_id)
            arr_logits = model_id(subwords)
            print(arr_logits)
            logits = arr_logits[0]
            logits_duration = (datetime.now() - start7).total_seconds()
            start4 = datetime.now()
            label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
            topk_duration = (datetime.now() - start4).total_seconds()
            start5 = datetime.now()
            probability = F.softmax(logits, dim=-1).squeeze()[label].item()
            prob_duration = (datetime.now() - start5).total_seconds()
            in2_duration = (datetime.now() - start_in2).total_seconds()
        else:
            label = 1
            probability = 0.5
        in_duration = (datetime.now() - start_in).total_seconds()
    except:
        label = 1
        probability = 0.9


    # show results
    duration = (datetime.now() - start_req).total_seconds()
    print(f"""
        cleaning_duration: {cleaning_duration}
        encode_duration: {encode_duration}
        split_duration: {split_duration}
        longtensor_duration: {longtensor_duration}
        logits_duration: {logits_duration}
        topk_duration: {topk_duration}
        prob_duration: {prob_duration}
        in_duration: {in_duration}
        in2_duration: {in2_duration}
        Duration: {duration}
            """)
    print('Duration', (datetime.now() - start_req).total_seconds())
    result = {"prediction": i2w[label], "probability": probability}
    return result

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

i = 1
while i > 0:
    predict_id("@ganjarpranowo @brttransjateng Arep numpak BRT nunggu suwi 30 menit durung karuan numpak.tegowanu weta")
    i = i -1
