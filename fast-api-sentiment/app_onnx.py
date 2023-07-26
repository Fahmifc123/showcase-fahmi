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
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
import numpy as np
import timeit

import csv
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import string
from datetime import datetime
import torch
import onnxruntime as rt
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer

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

# first create session 
import onnxruntime
onnx_session = onnxruntime.InferenceSession('model/model_10ep.onnx', providers=['CPUExecutionProvider'])

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}

app = FastAPI(
    title="Sentiment Model API",
    description="Sentiment Model API Nolimit V3",
    version="0.3",
)

import re
import csv
import nltk
import pandas as pd
from nltk.corpus import stopwords

objects = pd.read_csv('list-objects.csv')
eng = pd.read_csv('unigram_freq.csv')
eng = eng['word'].tolist()

sw_project_name = set()
sw_content = set()
sw_extra_content = set()

for _, row in objects.iterrows():
    project_name = str(row['project_name']).lower()
    content = str(row['content']).lower()
    extra_content = str(row['extra_content']).lower()

    # Tokenize words in the project_name column
    project_name_tokens = project_name.split()
    sw_project_name.update(project_name_tokens)

    # Tokenize words in the content column
    content_tokens = content.split()
    sw_content.update(content_tokens)

    # Replace NaN values with empty string and tokenize words in the extra_content column
    if pd.isnull(extra_content):
        extra_content = ''
    extra_content_tokens = extra_content.split()
    sw_extra_content.update(extra_content_tokens)

stop_words = stopwords.words('indonesian')
stop_words.extend(sw_project_name)
# stop_words.update(sw_content)
# stop_words.update(sw_extra_content)
stop_words.extend(eng)
tambah_words = ['milo', 'jasa marga', 'jasa', 'allah', 'tuhan', 'swt', 'lemineral', 'mineral',
    'mineralnya', 'danone', 'marga', 'pssi', 'pak', 'buk', 'ibu', 'bapak', 'ayah', 'bu', 'aliexpres',
    'indomaret', 'cukai', 'umkm', 'menko', 'airlangga', 'pemerintah', 'pemkot', 'kediri', 'jawa',
     'timur', 'dinkop', 'jawa timur', 'min', 'adira', 'mendag', 'menteri perdagangan', 'menteri',
     'masya', 'wali kota', 'walikota', 'permai', 'cibubur', 'ganjar', 'pranowo', 'insya', 'pariwisata',
     'sperti', 'gojek', 'rakerda', 'partai', 'golkar', 'jatim', 'senin', 'selasa',' rabu', 'kamis',
     'jumat', 'sabtu', 'minggu', 'zulhas', 'perdagangan', 'kemayoran', 'mulyani', 'besok', 'nasabah',
     'dipermasalahkan', 'digoel', 'kumkm', 'pemkab', 'kejaksaan', 'bumn', 'menparekraf', 'kemenkop',
     'menit', 'detik', 'jam', 'menkop', 'cawe', 'julkifli', 'rohil', 'destinasi', 'basarnas', 'lpdb-kumkm',
     'lpdb', 'kumkm', 'muhaimin', 'cc', 'duonkcc', 'si cepat', 'sicepat', 'tollroad', 'contraflow',
     'garnita', 'malahayati', 'nasional', 'demokrat', 'minerale', 'botol', 'zoel', 'menteri', 'luar', 'negeri',
     'menlu', 'jaringan','orangnya', 'etol', 'etoll', 'alfamart', 'boyalali', 'provinsi', 'kabupaten', 'rezim',
     'kota', 'ibukota', 'tiktok', 'media', 'sosial', 'sosmed', 'wkwk', 'gelas', 'check', 'it', 'out', 'cekidot',
     'online', 'insyaallah', 'tampomas', 'sugbk', 'pempek', 'ekspor', 'wamendag', 'grapari', 'tsel',
     'simbolis', 'melepas', 'grabcar', 'anoc', 'bacawapres', 'awbg', 'ppmse', 'permendag', 'nomor', 'masduki',
     'geliat', 'alumunium', 'inglot', 'milik', 'nalindo', 'tiongkok', 'zonknya', 'bpom', 'kpkp']

stop_words.extend(tambah_words)

a = pd.read_csv('clean/daftar_tempat_split_ordered.csv')
a = a['nama_tempat'].str.lower().tolist()
stop_words.extend(a)

b = pd.read_csv('clean/daftar_susunan_kementerian_only.csv')
b = b['nama'].str.lower().tolist()
stop_words.extend(b)

c = pd.read_csv('clean/daftar_bumn_name_split.csv')
c = c['nama'].str.lower().tolist()
stop_words.extend(c)

d = pd.read_csv('clean/daftar_jabatan_asn_split.csv')
d = d['jabatan'].str.lower().tolist()
stop_words.extend(d)

stop_words = set(stop_words)
stop_words = list(stop_words)

rmv = ['promo', 'menarik',' strategi', 'radikal', 'tegas', 'motor', 'tidak', 'mohon', 'bukan', 'manfaat',
    'ganja', 'tai', 'bego', 'strategi', 'terima', 'banyak', 'keselamatan', 'selamat', 'pengelolaan',
    'semakin','sulit', 'susah', 'konsumsi', 'bangkit', 'dukung', 'tenang', 'lebih', 'baik', 'terima']

for i in rmv:
    if i in stop_words:
        stop_words.remove(i)

with open('clean/new_kamus_alay_gaul_nolimit.csv', encoding='latin-1', mode='r') as infile:
    reader = csv.reader(infile)
    slangwords = {rows[0]: rows[1] for rows in reader}

def convertToSlangword(review):
    pattern = re.compile(r'\b(' + '|'.join(slangwords.keys()) + r')\b')
    content = []
    for kata in review.split():
        filteredSlang = pattern.sub(lambda x: slangwords[x.group()], kata)
        content.append(filteredSlang.lower())
    review = content
    return review

def text_cleaning_id(text):

    def casefolding(review):
        review = review.lower()
        return review

    def tokenize(review):
        tokens = nltk.word_tokenize(review)
        return tokens

    def filtering(review):
        # Menghapus URL
        review = re.sub(r'https?://\S+', '', review)

        review = re.sub(r'@\w+\b', '', review)
        review = re.sub(r'@\w+', '', review)

        # Menghapus kata setelah tanda pagar (#) hanya jika jumlah hashtag tepat 3
        hashtags = re.findall(r'#([^\s]+)', review)
        for hashtag in hashtags:
            review = re.sub(r'#' + re.escape(hashtag) + r'\b', '', review)
        
        review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", review)
        review = re.sub(r'[^\x00-\x7f]', r'', review)
        review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', review)
        review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
        review = re.sub(r'\\u\w\w\w\w', '', review)
        review = re.sub(r'@\w+\b', '', review)

        return review

    def replaceThreeOrMore(review):
        pattern = re.compile(r"(\w)\1{2,3}")
        return pattern.sub(r"\1", review)

    text = filtering(text)
    text = casefolding(text)
    tokens = tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    text = " ".join(tokens)
    text = replaceThreeOrMore(text)
    tokens = tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_review = " ".join(tokens)

    # Apply convertToSlangword
    cleaned_review = convertToSlangword(cleaned_review)

    return cleaned_review

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

@app.on_event("startup")
def startup():
    print("start")
    RunVar("_default_thread_limiter").set(CapacityLimiter(2))

from typing import List
class Item(BaseModel):
    text: str
    id: str

from concurrent.futures import ThreadPoolExecutor
import asyncio

@app.post("/sentiment/v3/predict-id-bulk/")
async def predict_id(reviews: List[Item]):
    results = []
    for review in reviews:
        result = await process_review(review)
        results.append(result)
    return results

async def process_review(review: Item):
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
        cleaned_review = text_cleaning_id(review.text)
        # cleaned_tokens = cleaned_review.split()
        cleaned_review = [token for token in cleaned_review if token not in stop_words]
        cleaned_review = " ".join(cleaned_review)
        words = cleaned_review.split()
        unique_words = list(dict.fromkeys(words))
        cleaned_review = ' '.join(unique_words)
        print(cleaned_review)

        cleaning_duration = (datetime.now() - start1).total_seconds()
        start6 = datetime.now()
        kata = cleaned_review.split()
        split_duration = (datetime.now() - start6).total_seconds()
        jumlah_kata = len(kata)

        if len(review.text.split()) > 50:
            label = 1
            probability = 0.5

        elif jumlah_kata > 4:
            start_in2 = datetime.now()
            start2 = datetime.now()
            inputs = tokenizer.encode_plus(cleaned_review, add_special_tokens=True, return_tensors='pt')
            encode_duration = (datetime.now() - start2).total_seconds()

            start7 = datetime.now()
            logits = onnx_session.run(None, {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy(),
                'token_type_ids': inputs['token_type_ids'].numpy()
            })[0]
            logits_duration = (datetime.now() - start7).total_seconds()
            start4 = datetime.now()
            label = torch.topk(torch.from_numpy(logits), k=1, dim=-1)[1].squeeze().item()
            topk_duration = (datetime.now() - start4).total_seconds()
            start5 = datetime.now()
            probability = F.softmax(torch.from_numpy(logits), dim=-1).squeeze()[label].item()
            prob_duration = (datetime.now() - start5).total_seconds()
            in2_duration = (datetime.now() - start_in2).total_seconds()

        else:
            label = 1
            probability = 0.5

        in_duration = (datetime.now() - start_in).total_seconds()
    except Exception as e:
        label = 1
        probability = 0.9

    # output dictionary
    i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}

    duration = (datetime.now() - start_req).total_seconds()
    print(f"""
        cleaning_duration: {cleaning_duration}
        encode_duration: {encode_duration}
        split_duration: {split_duration}
        longtensor_duration: {longtensor_duration}
        topk_duration: {topk_duration}
        prob_duration: {prob_duration}
        in_duration: {in_duration}
        in2_duration: {in2_duration}
        logits_duration: {logits_duration}
        duration: {duration}
        """)

    result = {   
    "id": review.id,
    "text": cleaned_review,
    "prediction": i2w[label],
    "probability": probability,
    "duration": duration
    }

    print(result)

    return result

# model english
@app.post("/sentiment/v3/testing-predict-en/")
def predict_en(reviews: List[Item]):
    results = []
    for review in reviews:
        start_req = datetime.now()
        cleaned_review = text_cleaning_en(review.text)
        print(cleaned_review)
        
        # perform prediction
        start_pred = datetime.now()
        prediction = model_en.predict([cleaned_review])
        output = int(prediction[0])
        probas = model_en.predict_proba([cleaned_review])
        output_probability = "{:.2f}".format(float(probas[:, output]))
        prediction_duration = (datetime.now() - start_pred).total_seconds()

        # output dictionary
        sentiments = {0: "neutral", 1: "negative", 2: "positive"}

        # show results
        duration = (datetime.now() - start_req).total_seconds()
        print(f"""
            Prediction Duration: {prediction_duration}
            Total Duration: {duration}
        """)

        result = {
            "id": review.id,
            "prediction": sentiments[output],
            "probability": output_probability
        }
        results.append(result)

        print(results)

    return results
