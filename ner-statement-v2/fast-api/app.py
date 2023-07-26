import os
from fastapi import FastAPI
import torch
from simpletransformers.ner import NERModel, NERArgs

from predictor import predict_person

from transformers import BertTokenizer, BertForTokenClassification

import nltk
import re
import pandas as pd
import numpy as np
import string
nltk.download('punkt')
nltk.download('stopwords')

from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

import pickle

model = pickle.load(open('model/model_simple_ner_10.pkl', 'rb'))

app = FastAPI(
    title="NER Indobert API",
    description="NER Indobert API Nolimit 2023",
    version="2.0",
)

class Item(BaseModel):
    text: str = """JAKARTA, KOMPAS. TV - Komisi Pemberantasan Korupsi menemukan 134 pegawai pajak memiliki saham di 280 perusahaan. Temuan fakta tersebut diperoleh setelah lembaga antirasuah melakukan analisis terkait Laporan Harta Kekayaan Penyelenggara Negara atau LHKPN.

KPK menyebut, kepemilikan saham oleh pegawai pajak yang berstatus Pegawai Negeri Sipil (PNS) akan menimbulkan celah gratifikasi dan tidak etis. Namun sebenarnya, apakah boleh PNS membeli saham perusahaan?

Pengamat Pajak Bastanul Siregar mengatakan, pegawai pajak sah-sah saja mempunyai saham untuk investasi. Ia mencontohkan dalam LHKPN banyak PNS menyampaikan memiliki harta dalam bentuk surat berharga.

"Surat berharga itu bisa saham bisa juga Surat Berharga Negara (SBN) atau obligasi," kata Bastanul saat dihubungi Kompas TV, Kamis (9/3/2023).

"Ketika PNS punya saham sebenarnya sah-sah saja. Yang enggak boleh itu jika saham itu menjadi alat atau modus PNS untuk mencari uang atau trading harian. PNS itu ya termasuk TNI Polri juga," ujarnya.

Ia menyampaikan, PNS tidak boleh menjadi trader saham harian, karena rawan konflik kepentingan.

"Kalau jadi insider trading itu, misalkan dia pegawai pajak, dia tahu perusahaan X akan dapat proyek dari Direktorat Jenderal Pajak, lalu dia beli duluan saham itu. Jadi saat kinerja saham perusahaan itu bagus karena kerja sama dengan DJP, dia sudah tahu duluan," ujarnya.

Dalam Peraturan Pemerintah No 94 Tahun 2021 tentang Disiplin PNS, memang tidak disebutkan jika PNS tidak boleh mempunyai saham. Pasal 5 huruf a hanya menyebutkan, PNS dilarang menyalahgunakan wewenang.

Kemudian dalam huruf f PP yang sama tertulis, PNS dilarang untuk memiliki, menjual, membeli, menggadaikan, menyewakan, atau meminjamkan barang baik bergerak atau tidak bergerak, dokumen, atau surat berharga milik negara secara tidak sah.

Lantas, bagaimana caranya mengetahui mana PNS yang membeli saham untuk investasi jangka panjang dan mana PNS yang menjadi insider trading? Lantaran tidak ada hukum normatif yang mengatur larangan itu.

Bastanul mengakui, dengan tidak adanya pengawasan yang konsisten, masih sulit untuk mencegah PNS menjadi insider trading.

Kecuali pemerintah menerapkan sistem Single Identity Number (SIN) atau Nomor Identitas Tunggal. SIN ini bisa memonitor terjadinya percobaan atau upaya melakukan korupsi.

"Negara seperti Jepang saja masih kesulitan mengatur soal insider trading yang memang rumit ini," ucapnya.

"Tapi dengan adanya SIN, jangankan beli saham, beli mobil bekas saja akan terdeketsi. Akan tercatat di polisi itu lalu terkoneksi ke data di Ditjen Pajak. Sayang Indonesia belum bisa terapkan SIN," ucapnya.

Bastanul menuturkan, dengan sistem ini semua transaksi bisa terdeteksi. Jadi tidak ada lagi wajib pajak yang bisa menyembunyikan hartanya dan berani melakukan kongkalikong. Karena datanya terbuka. Termasuk para pegawai pajak sendiri tak akan berani.

"Kalau sekarang, misalkan Anda punya tanah 10, lalu yang Anda laporkan hanya 5. Apakah pegawai pajak akan tahu? Tidak akan. Mereka juga enggak akan negecek satu-satu," tuturnya.

Bastanul bilang, menerapkan SIN Pajak ini butuh political will. Malaysia, Singapura, Thailand, dan bahkan Estonia sudah menerapkan sistem nomor identitas tunggal ini. Indonesia masih ketinggalan.

"Ini tidak bisa diterapkan karena yang menghambat soal itu ya Menteri Keuangan Sri Mulyani sendiri. Kementerian Keuangan terlihat malas kalau soal pengawasan," ucapnya.

Dihubungi terpisah, mantan Dirjen Pajak Hadi Poernomo juga menyebut SIN bisa menjadi solusi atas masalah yang ada di Kemenkeu saat ini.

Hadi mengibaratkan, SIN sebagai kamera pengawas CCTV yang bisa mengawasi keuangan wajib pajak. Dengan penerapan SIN, setiap instansi pemerintah pusat/daerah, lembaga, swasta dan pihak-pihak lain wajib saling membuka dan menyambung sistem ke otoritas pajak yang rahasia dan non rahasia baik yang finansial dan non finansial.

Dengan begitu, setiap uang yang diterima dari berbagai sumber dapat diketahui secara langsung melalui sistem perpajakan. Lalu, otoritas pajak juga bisa memeriksa kebenaran data dan informasi yang disampaikan wajib pajak secara akurat.

"Sistem tersebut mengawasi seluruh transaksi keuangan, sehingga menciptakan transparansi yang bisa mencegah korupsi sekaligus meningkatkan penerimaan pajak," kata Hadi dalam keterangan tertulisnya kepada Kompas TV, Kamis (9/3).

Mantan Ketua BPK ini menjelaskan, secara psikologis wajib pajak akan selalu melakukan penghindaran pajak jika ada kesempatan. Inilah yang menyebabkan munculnya korupsi yang ujungnya membuat penerimaan pajak rendah.

Nah dengan SIN Pajak, petugas pajak tidak bisa sembarangan lagi memeriksa SPT wajib pajak. Petugas pajak juga tidak bisa "bermain" karena diawasi CCTV.

"Persoalannya, SIN Pajak ini keberadaannya antara ada dan tiada. Dalam Undang-undang ada, tapi dalam pelaksanaan di lapangan seperti tiada karena adanya inkonsistensi regulasi yang menyebabkan kenapa SIN Pajak tidak bisa terlaksana," ucapnya.

"Undang-undangnya sudah benar, tetapi peraturan pelaksanaannya seperti peraturan pemerintah, peraturan menteri tidak konsisten dengan bunyi undang-undang," ucapnya.

Adapun KPK bakal segera menyerahkan temuan 134 pegawai Direktorat Jenderal Pajak yang memiliki saham di 280 perusahaan kepada Kemenkeu.

Deputi Pencegahan dan Monitoring KPK Pahala Nainggolan menyebut penyerahan data tersebut akan dilakukan hari ini, Jumat (10/3/2023).

"Mungkin besok (dilaporkan ke Kemenkeu)," ucap Pahala.

Dia pun mengatakan nama-nama pegawai tersebut telah dihimpun KPK untuk diserahkan. Sehingga nantinya bisa segera ditindaklanjuti oleh Kemenkeu.

"Kita lakukan pendalaman terhadap data yang kita punya, tercatat bahwa ada 134 pegawai pajak ternyata punya saham di 280 perusahaan," ujarnya.
"""

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ner/")
def ner(review: Item):

  new_text = review.text

  print(new_text)

  sentences = []
  start = 0
  for match in re.finditer(r'\n', new_text):
      end = match.start()
      sentences.append(new_text[start:end])
      start = match.end()

  # add the last sentence
  sentences.append(new_text[start:])
  sentences.remove(sentences[-1])
  print(sentences)
  print(len(sentences))

  for i, y in enumerate(sentences):
    if len(y) < 10:
      sentences.remove(sentences[i])
    else:
      pass

  for i, y in enumerate(sentences):
    print(i, '-', y)  

  # predict
  # Membuat kolom baru
  df = pd.DataFrame(columns=["words", "label"])

  predictions, raw_outputs = model.predict(sentences)

  # Looping untuk membaca data dan memasukkan ke dalam dataframe
  for i in predictions:
      text = []
      label = []
      for j in i:
          text.append(list(j.keys())[0])
          label.append(list(j.values())[0])
      df = df.append({"words": text, "label": label}, ignore_index=True)

  word = [item for sublist in df['words'].tolist() for item in sublist]
  label = [item for sublist in df['label'].tolist() for item in sublist]

  hasil = pd.DataFrame({'words': word, 'label': label})
  hasil_tanpaO = hasil.drop_duplicates().reset_index(drop=True)
  hasil_tanpaO = hasil[hasil['label'] != 'O'].reset_index(drop=True)
  # hapus titik dan koma pada kolom text
  hasil['words'] = hasil['words'].str.replace('.com', '', regex=True)
  hasil['words'] = hasil['words'].str.replace('[.,()-:;"â”]', '', regex=True)
  hasil['words'] = hasil['words'].str.lower()
  indeks = hasil[(hasil['label'] == 'B-PER') & (hasil['label'].shift(-1) != 'I-PER') & (hasil['words'].str.len() <= 3)].index
  hasil = hasil.drop(indeks).reset_index(drop=True)
  hasil = hasil[hasil['label'] != 'O'].reset_index(drop=True)
  hasil

  #full dibawah ini yang combine B-I

  # print(text_mentah)
  print('-'*100)

  output = {}

  for i in range(len(hasil)):

    # --------------- PERSON -----------------------------
    try:
      if hasil['label'][i]=='B-PER' and hasil['label'][i+1]!='I-PER':
        print(hasil['words'][i], '=> PERSON')
        output[hasil['words'][i]] = 'PERSON'
      else:
        pass

    except:
      pass
      
    try:
      if hasil['label'][i]=='B-PER' and ((i+1)==len(hasil)):
        print(hasil['words'][i], '=> PERSON')
        output[hasil['words'][i]] = 'PERSON'
      
      else:
        pass
        
    except:
      pass

    try:
      if hasil['label'][i]=='B-PER' and hasil['label'][i+1]=='I-PER' and hasil['label'][i+2]=='I-PER':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2], '=> PERSON')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2]] = 'PERSON'
      elif hasil['label'][i]=='B-PER' and hasil['label'][i+1]=='I-PER':
        print(hasil['words'][i],hasil['words'][i+1], '=> PERSON')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1]] = 'PERSON'
      else:
        pass

    except:
      pass

    # --------------- LOCATION -----------------------------
    try:
      if hasil['label'][i]=='B-LOC' and hasil['label'][i+1]!='I-LOC':
        print(hasil['words'][i], '=> LOCATION')
        output[hasil['words'][i]] = 'LOCATION'
      else:
        pass

    except: 
      if hasil['label'][i]=='B-LOC' and ((i+1)==len(hasil)):
        print(hasil['words'][i], '=> LOCATION')
        output[hasil['words'][i]] = 'LOCATION'
      else:
        pass

    try:
      if hasil['label'][i]=='B-LOC' and hasil['label'][i+1]=='I-LOC' and hasil['label'][i+2]=='I-LOC':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2], '=> LOCATION')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2]] = 'LOCATION'
      elif hasil['label'][i]=='B-LOC' and hasil['label'][i+1]=='I-LOC':
        print(hasil['words'][i],hasil['words'][i+1], '=> LOCATION')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1]] = 'LOCATION'
      else:
        pass
    
    except:
      pass

    # --------------- ORGANIZATION ----------------------------
    try:
      if hasil['label'][i]=='B-ORG' and hasil['label'][i+1]!='I-ORG':
        print(hasil['words'][i], '=> ORGANIZATION') 
        output[hasil['words'][i]] = 'ORGANIZATION'
      else:
        pass

    except: 
      if hasil['label'][i]=='B-ORG' and ((i+1)==len(hasil)):
        print(hasil['words'][i], '=> ORGANIZATION')
        output[hasil['words'][i]] = 'ORGANIZATION'
      else:
        pass

    try:
      if hasil['label'][i]=='B-ORG' and hasil['label'][i+1]=='I-ORG' and hasil['label'][i+2]=='I-ORG':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2], '=> ORGANIZATION')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2]] = 'ORGANIZATION'
      elif hasil['label'][i]=='B-ORG' and hasil['label'][i+1]=='I-ORG':
        print(hasil['words'][i],hasil['words'][i+1], '=> ORGANIZATION')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1]] = 'ORGANIZATION'
      else:
        pass

    except:
      pass

    # --------------- POLITICAL ORGANIZATION ----------------------------
    try:
      if hasil['label'][i]=='B-NOR' and hasil['label'][i+1]!='I-NOR':
        print(hasil['words'][i], '=> POLITICAL ORGANIZATION') 
        output[hasil['words'][i]] = 'POLITICAL ORGANIZATION'
      else:
        pass

    except: 
      if hasil['label'][i]=='B-NOR' and ((i+1)==len(hasil)):
        print(hasil['words'][i], '=> POLITICAL ORGANIZATION')
        output[hasil['words'][i]] = 'POLITICAL ORGANIZATION'
      else:
        pass

    try:
      if hasil['label'][i]=='B-NOR' and hasil['label'][i+1]=='I-NOR' and hasil['label'][i+2]=='I-NOR':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2], '=> POLITICAL ORGANIZATION')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2]] = 'POLITICAL ORGANIZATION'
      else:
        pass

    except:
      if hasil['label'][i]=='B-NOR' and hasil['label'][i+1]=='I-NOR':
        print(hasil['words'][i],hasil['words'][i+1], '=> POLITICAL ORGANIZATION')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1]] = 'POLITICAL ORGANIZATION'
      else:
        pass

    # --------------- GEOPOLITICAL ENTITY ----------------------------
    try:
      if hasil['label'][i]=='B-GPE' and hasil['label'][i+1]!='I-GPE':
        print(hasil['words'][i], '=> GEOPOLITICAL ENTITY') 
        output[hasil['words'][i]] = 'GEOPOLITICAL ENTITY'
      else:
        pass

    except: 
      if hasil['label'][i]=='B-GPE' and ((i+1)==len(hasil)):
        print(hasil['words'][i], '=> GEOPOLITICAL ENTITY')
        output[hasil['words'][i]] = 'GEOPOLITICAL ENTITY'
      else:
        pass

    try:
      if hasil['label'][i]=='B-GPE' and hasil['label'][i+1]=='I-GPE' and hasil['label'][i+2]=='I-GPE':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2], '=> GEOPOLITICAL ENTITY')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2]] = 'GEOPOLITICAL ENTITY'
      elif hasil['label'][i]=='B-GPE' and hasil['label'][i+1]=='I-GPE':
        print(hasil['words'][i],hasil['words'][i+1], '=> GEOPOLITICAL ENTITY')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1]] = 'GEOPOLITICAL ENTITY'
      else:
        pass

    except:
      pass


    # --------------- EVENT ----------------------------
    try:
      if hasil['label'][i]=='B-EVT' and hasil['label'][i+1]!='I-EVT':
        print(hasil['words'][i], '=> EVENT') 
        output[hasil['words'][i]] = 'EVENT'
      else:
        pass

    except: 
      if hasil['label'][i]=='B-EVT' and ((i+1)==len(hasil)):
        print(hasil['words'][i], '=> EVENT')
        output[hasil['words'][i]] = 'EVENT'
      else:
        pass

    try:
      if hasil['label'][i]=='B-EVT' and hasil['label'][i+4]=='I-EVT' and hasil['label'][i+5]=='I-EVT':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2],hasil['words'][i+3],hasil['words'][i+4],hasil['words'][i+5], '=> EVENT')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2] + ' ' + hasil['words'][i+3] + ' ' + hasil['words'][i+4] + ' ' + hasil['words'][i+5]] = 'EVENT'
      elif hasil['label'][i]=='B-EVT' and hasil['label'][i+3]=='I-EVT' and hasil['label'][i+4]=='I-EVT':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2],hasil['words'][i+3],hasil['words'][i+4], '=> EVENT')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2] + ' ' + hasil['words'][i+3] + ' ' + hasil['words'][i+4]] = 'EVENT'
      elif hasil['label'][i]=='B-EVT' and hasil['label'][i+2]=='I-EVT' and hasil['label'][i+3]=='I-EVT':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2],hasil['words'][i+3], '=> EVENT')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2] + ' ' + hasil['words'][i+3]] = 'EVENT'
      elif hasil['label'][i]=='B-EVT' and hasil['label'][i+1]=='I-EVT' and hasil['label'][i+2]=='I-EVT':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2], '=> EVENT')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2]] = 'EVENT'
      elif hasil['label'][i]=='B-EVT' and hasil['label'][i+1]=='I-EVT':
        print(hasil['words'][i],hasil['words'][i+1], '=> EVENT')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1]] = 'EVENT'
      else:
        pass

    except:
      pass

    try:
      if hasil['label'][i]=='B-EVT' and hasil['label'][i+2]=='I-EVT' and hasil['label'][i+3]=='I-EVT':
        print(hasil['words'][i],hasil['words'][i+1],hasil['words'][i+2],hasil['words'][i+3], '=> EVENT')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1] + ' ' + hasil['words'][i+2] + ' ' + hasil['words'][i+3]] = 'EVENT'
      elif hasil['label'][i]=='B-EVT' and hasil['label'][i+1]=='I-EVT':
        print(hasil['words'][i],hasil['words'][i+1], '=> EVENT')
        output[hasil['words'][i] + ' ' + hasil['words'][i+1]] = 'EVENT'
      else:
        pass

    except:
      pass


    # --------------- O -----------------------------
    try:
      if hasil['label'][i]=='O':
        print(hasil['words'][i], '=> O')
        output[hasil['words'][i]] = 'O'
      else:
        pass

    except:
      pass


  new_output = {}
  for k,v in output.items():
    if type(k) == tuple:
      new_output[' '.join(k)] = v
    
    else:
      new_output[''.join(k)] = v
      
  # new_output

  df = pd.DataFrame()

  df['text'] = new_output.keys()
  df['label'] = new_output.values()

  # show results
  result = {'tokens' : new_output.keys(), 'labels' : new_output.values()}

  # Membuat kamus kosong untuk menampung hasil modifikasi
  modified_result = {}

  # Loop through tokens and labels and add modified tokens to dictionar
  for token, label in zip(result['tokens'], result['labels']):
    # Replace unwanted characters with empty string
    print(f'{token}: {label}')
    modified_result[token] = label

  # Return the modified dictionary
  return modified_result

# people = predict_person('Fahira Fahmi')
# print(people)

@app.post("/statement/")
def statement(review: Item):

  new_text = review.text

  print(new_text)

  new_text = re.sub(r".*- ", "", new_text)
  # print(new_text)

  sentences = []
  start = 0
  for match in re.finditer(r'\n', new_text):
      end = match.start()
      sentences.append(new_text[start:end])
      start = match.end()

  # add the last sentence
  sentences.append(new_text[start:])
  sentences.remove(sentences[-1])
  print(sentences)
  print(len(sentences))

  for i, y in enumerate(sentences):
    if len(y) < 10:
      sentences.remove(sentences[i])
    else:
      pass

  for i, y in enumerate(sentences):
    print(i, '-', y)


  # coba generate statement pakai kalimat dari paragraf aja

  sentences = [x.lower() for x in sentences]
  people = ''
  full_people = []
  full_statement = []

  for sentence in sentences:
    
    try:

      #mengatakan
      match = re.search(r"\bmengatakan\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()
          
          if len(statement) < 20:
            statement = ''
            people = people

          print(people, "==>", statement)

          full_people.append(people)
          full_statement.append(statement)

        except:
          pass

      #menjelaskan
      match = re.search(r"\bmenjelaskan\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()
          
          if len(statement) < 20:
            statement = ''

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)

        except:
          pass

      #menurut
      match = re.search(r"menurut (.*?),", sentence)
      if match:
        try:
          statement = sentence.split(match.group(0))[1]

          if 'dia' not in match.group(1):
            people = predict_person(match.group(1))
          elif 'dia' in match.group(1):
            people = ''
            statement = ''

          if len(people) > 20:
            people = ''
            statement = ''
          else:
            pass

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)

        except:
          pass

      #menurut
      match = re.search(r"\bmenurut\b", sentence)
      if match:
        try:
          people = predict_person(sentence[match.end():].strip())
          print(people)
          statement = sentence[match.end():].strip().split(people)[-1]
          
          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #kata
      match = re.search(r"kata (.*)", sentence)
      if match:
        try:
          people = predict_person(match.group(1))

        except:
          pass

        statement = sentence.split(match.group(0))[0]
        if len(statement) < 20:
          try:
            statement = sentence.split(",", 1)[1]

          except:
            pass

        if len(statement) < 20:
          statement = ''
        else:
          pass

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #jelas
      match = re.search(r"\bjelas\b", sentence)
      if match:
        try:
          people = predict_person(sentence[match.end():].strip())
          statement = sentence[:match.start()].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #ujarnya
      match = re.search(r"(.*?) ujarnya\.", sentence)
      if match:
        # people = people.mode()
        people = people
        statement = match.group(1)
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #sebaliknya
      match = re.search(r"sebaliknya,(.*)", sentence)
      if match:
        # people = people.mode()
        after_keyword = match.group(1)
        statement = after_keyword.split(",", 1)[1]
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #ucap
      match = re.search(r"ucap (.*)", sentence)
      if match:
        if 'dia' not in match.group(1).split(" ")[0]:
          people = match.group(1).split(" ")[0]
        elif 'dia' in match.group(1).split(" ")[0]:
          people = people

        if people.endswith("."):
          people = people[:-1]
        statement = sentence.split(match.group(0))[0]
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #tuturnya
      match = re.search(r"(.*?) tuturnya\.", sentence)
      if match:
        # people = people.mode()
        statement = match.group(1)
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #lanjut
      match = re.search(r"\blanjut\b (.*?),", sentence)
      if match:
          people = match.group(1)
          statement = sentence.split(people)[1]
          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)

      #menegaskan
      match = re.search(r"\bmenegaskan\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()
          
          if len(statement) < 20:
            statement = ''
            people = people

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)

        except:
          pass

      #menuturkan
      match = re.search(r"menuturkan (.*)", sentence)
      if match:
        statement = match.group(1)
        people = sentence.split("menuturkan")[0]
        if "," in people:
          people = people.split(",", 1)[1]
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #tutur
      match = re.search(r"\btutur\b", sentence)
      if match:
        try:
          people = predict_person(sentence[match.end():].strip())
          statement = sentence[:match.start()].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #menurutnya
      match = re.search(r"menurutnya,", sentence)
      if match:
        start_index = match.end()
        end_index = len(sentence)
        statement = sentence[start_index:end_index].strip()
        try:
          people = predict_person(sentence[start_index-50:start_index].strip())
        except:
          pass
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #ungkap
      match = re.search(r"\bungkap\b", sentence)
      if match:
        try:
          people = predict_person(sentence[match.end():].strip())
          statement = sentence[:match.start()].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #imbuh
      match = re.search(r"\bimbuh\b", sentence)
      if match:
        try:
          people = predict_person(sentence[match.end():].strip())
          statement = sentence[:match.start()].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #ujar
      match = re.search(r"\bujar\b", sentence)
      if match:
        try:
          people = predict_person(sentence[match.end():].strip())
          if len(people) > 2:
            statement = sentence[:match.start()].strip()

            print(people, "==>", statement)
            full_people.append(people)
            full_statement.append(statement)
          
          else:
            pass
        
        except:
          pass

      # #mengungkapkan
      # match = re.search(r"(?<=mengungkapkan ).+?(?=\.)", sentence)
      # if match:
      #   people = sentence[:match.start()].replace("mengungkapkan", "").strip()
      #   statement = match.group().strip()
      #   print(people, "==>", statement)

      #mengungkapkan
      match = re.search(r"\bmengungkapkan\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #tutup
      match = re.search(r"\btutup\b (.*)", sentence)
      if match:
        words = match.group(1).strip()
        if len(words.split()) == 1:
          people = words.split()[0]
        else:
          people = people

        if '.' in people:
          split_titik = people.split(".")
          people = split_titik[0].strip()

        statement = sentence.split(match.group(0))[0]
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #sambungnya
      match = re.search(r"sambungnya", sentence)
      if match:
        people = people
        statement = sentence.split(match.group(0))[0].strip()

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #tambahnya
      match = re.search(r"tambahnya", sentence)
      if match:
        people = people
        statement = sentence.split(match.group(0))[0].strip()

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #pungkas
      match = re.search(r"\bpungkas\b", sentence)
      if match:
        people = sentence.split(match.group(0))[1].strip()
        statement = sentence.split(match.group(0))[0].strip()

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #katanya
      match = re.search(r"katanya", sentence)
      if match:
        people = people
        statement = sentence.split(match.group(0))[0].strip()
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #pungkasnya
      match = re.search(r"pungkasnya", sentence)
      if match:
        people = people
        statement = sentence.split(match.group(0))[0].strip()

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #menuturkan
      match = re.search(r'([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+menuturkan,\s+(.*)', sentence)
      if match:
        if 'pun' in match.group(1):
          people = people
        
        else:

          if match.group(1) == 'ia':
            people = people
          else:
            people = match.group(1)
            if len(people.split()) == 1:
              people = people.split()[0]
            
        statement = match.group(2)
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #beber
      match = re.search(r'(.*)\bbeber\b\s(\w+)\s(.*)', sentence)
      if match:
          statement = match.group(1).strip()
          people = match.group(2)
          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
      else:
          pass

      #berharap
      match = re.search(r"\bberharap\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #tandas
      match = re.search(r"\btandas\b", sentence)
      if match:
        index = match.start()
        statement = sentence[:index].strip()
        people = sentence[index+len("tandas"):].strip()
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)
      else:
        pass

      #sebut
      match = re.search(r"\bsebut\b", sentence)
      if match:
        index = match.start()
        statement = sentence[:index].strip()
        people = sentence[index+len("sebut"):].strip()
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)
      else:
        pass

      #menyebutkan
      match = re.search(r"\bmenyebutkan\b", sentence)
      if match:
        if sentence[:match.start()].strip() == 'ia':
          people = people
        else:
          people = sentence[:match.start()].strip()

        statement = sentence[match.end()+2:].strip()

        if len(people)>20:
          people = ''
          statement = ''

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #imbuhnya
      match = re.search(r"imbuhnya", sentence)
      if match:
        people = people
        statement = sentence.split(match.group(0))[0].strip()

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #tambah
      match = re.search(r"\btambah\b", sentence)
      if match:
        try:
          people = predict_person(sentence[match.end():].strip())
          if not hasil['label'].str.contains('B-PER').any():
            pass

          else:
            if len(people) > 2:
              statement = sentence[:match.start()].strip()
              print(people, "==>", statement)
              full_people.append(people)
              full_statement.append(statement)

            else:
              pass
        
        except:
          pass

      #ucapnya
      match = re.search(r"ucapnya", sentence)
      if match:
        people = people
        statement = sentence.split(match.group(0))[0].strip()

        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

      #menekankan 
      match = re.search(r"\bmenekankan\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass
      
      #menyatakan
      match = re.search(r"\bmenyatakan\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #menyebut
      match = re.search(r"\bmenyebut\b", sentence)
      if match:
        try:
          people = predict_person(sentence[:match.start()].strip())
          statement = sentence[match.end():].strip()

          print(people, "==>", statement)
          full_people.append(people)
          full_statement.append(statement)
        
        except:
          pass

      #lanjutnya
      match = re.search(r"\blanjutnya\b", sentence)
      if match:
        people = people
        statement = sentence.split(match.group(0))[0].strip()
        print(people, "==>", statement)
        full_people.append(people)
        full_statement.append(statement)

    except:
      pass

  # Membuat list kosong untuk menampung hasil modifikasi
  modified_result = {"result": []}

  # Loop through tokens and labels and add modified tokens to dictionary
  for one_people, one_statement in zip(full_people, full_statement):
    # Check if both "person" and "statement" are not empty
    if one_people != "" or one_statement != "":
      modified_dict = {"person": one_people, "statement": one_statement}
      modified_result["result"].append(modified_dict)

  # Return the modified dictionary
  return modified_result