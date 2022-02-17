import keras
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize


import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


from gensim.models.keyedvectors import KeyedVectors
from datetime import datetime
import numpy as np


top_words = 7000
max_words = 500
weights = {0:1, 1:1}
keylen = 100

try:
    import google.colab
    IN_COLAB = True
    path = '/content/drive/MyDrive/swarog/datasets/grafn/'
    import pickle5 as pickle
    print('using colab')
except:
    import pickle
    IN_COLAB = False
    print('using local')
    path = '/media/rkozik/0C129CFC129CEBC8/data/swarog/datasets/grafn/'


# Naive implementation of Doc2Vec
class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                #print("not found word:",word)
                word_vecs.append(300*[0])
                #pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.

        if len(word_vecs) == 0:
            print("no words in ", doc)
            return np.array(300*[0])

        vector = np.mean(word_vecs, axis=0)
        return vector


    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim
      
def set_path(x):
    global path
    path = x

def sumarise_text(text):
    words = (text).split(" ")
    sentences = sent_tokenize(text)
    if len(sentences) <= 5:
        return text
    freqTable = dict()
    for bword in words:
        word = bword.lower()
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    sentenceValue = dict()
    sentenceLen = dict()
    for sentence in sentences:
        sentence_key = sentence[:keylen]
        word_count_in_sentence = len((sentence).split(" "))
        sentenceLen[sentence_key] = word_count_in_sentence
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence_key in sentenceValue:
                    sentenceValue[sentence_key] += freqTable[wordValue]
                else:
                    sentenceValue[sentence_key] = freqTable[wordValue]
        if sentence_key in sentenceValue:
            sentenceValue[sentence_key] = sentenceValue[sentence_key] / word_count_in_sentence
    arr = []
    for sentence in sentences:
        sentence_key = sentence[:keylen]
        if sentence_key in sentenceValue:
            arr.append((sentenceLen[sentence_key],sentenceValue[sentence_key]))
    arr = sorted(arr, key=lambda tup: -tup[1])
    acum = 0
    i = 0
    while True:
        acum = acum + arr[i][0]
        if (i+1) >= len(arr) or acum + arr[i+1][0] >= 500:
            break
        i = i+1
    threshold =  arr[i][1]
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:keylen] in sentenceValue and sentenceValue[sentence[:keylen]] > (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary

def sumarise_dataset(txtN):
  i = 0
  for it in range(0,len(txtN)):
      doc = txtN[it]
      if i%6000 == 0 or (i+1)==len(txtN):
          print(i+1,"/",len(txtN))
      i=i+1  
      if len(doc) < 5*500:
          continue
      words = doc.split(" ")
      if len(words) > 500:
          sd = sumarise_text(doc)
          if len(doc) < 10:
              print("wird length of sumarised sentence len=",len(doc))
          else:
              txtN[it] = sd
  return txtN    

def read_grafn():
  fake = pd.read_csv(path + "badnews.csv").dropna()
  good = pd.read_csv(path + "goodnews.csv").dropna()
  print("LABELS:\n",fake["type"].value_counts())
  print("Get Real About The Fake News","fake:",fake.shape,"legit:",good.shape)
  return (good, fake)

def remove_stop_words(fakes):
  stop_words = set(stopwords.words())
  txtY = []
  i = 0
  for text in fakes:
    text_tokens = (text.replace("\n","")).split(" ")
    tokens_without_sw = [word for word in text_tokens if not word.lower() in stop_words]
    txtY.append(" ".join(tokens_without_sw))
    if i%6000 == 0 or (i+1)==len(fakes):
      print(i+1,"/",len(fakes))
    i=i+1
  return txtY

def store_sumarised(txtY, txtN):
  # saving
  with open(path + 'badnews_sumarised.pickle', 'wb') as handle:
    pickle.dump(txtY, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open(path + 'goodnews_sumarised.pickle', 'wb') as handle:
    pickle.dump(txtN, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("stored")

def load_sumarised():
  with open(path + 'badnews_sumarised.pickle', 'rb') as handle:
    txtY = pickle.load(handle)
  with open(path + 'goodnews_sumarised.pickle', 'rb') as handle:
    txtN = pickle.load(handle)
  return (txtY,txtN)

def fit_tokenizer(txtY, txtN):
  docs = txtY + txtN
  myTokenizer = Tokenizer(top_words)
  myTokenizer.fit_on_texts(docs)
  # saving
  with open(path + 'tokenizer.pickle', 'wb') as handle:
    pickle.dump(myTokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("done")
  return myTokenizer

def tokenize_corpus(txtY,txtN,t):
  s1 = t.texts_to_sequences(txtY)
  s2 = t.texts_to_sequences(txtN)
  s1pad =keras.preprocessing.sequence.pad_sequences(s1, max_words)
  s2pad =keras.preprocessing.sequence.pad_sequences(s2, max_words)
  s1lab = [1 for i in range(0,len(txtY))]
  s2lab = [0 for i in range(0,len(txtN))]
  x = s1pad.tolist() + s2pad.tolist()
  y = s1lab+s2lab
  with open(path + 'x_compact.pickle', 'wb') as handle:
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open(path + 'y_compact.pickle', 'wb') as handle:
    pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_compact():
  good, fake = read_grafn()
  txtY = remove_stop_words(fake['text'])
  txtN = remove_stop_words(good['text'])
  txtY=sumarise_dataset(txtY)
  txtN=sumarise_dataset(txtN)
  store_sumarised(txtY, txtN)
  t = fit_tokenizer(txtY,txtN)
  tokenize_corpus(txtY,txtN,t)

def load_raw_summarized():
  tY, tN = load_sumarised()
  with open(path + 'y_compact.pickle', 'rb') as handle:
    y = pickle.load(handle)
  return (tY+tN,y)

def load_raw_txt():
  good, fake = read_grafn()    
  tn = good['text'].tolist()
  ty = fake['text'].tolist()
  ylab = [1 for i in range(0,len(ty))]
  nlab = [0 for i in range(0,len(tn))]
  return (ty+tn, ylab+nlab)


def load_compact():
  with open(path + 'x_compact.pickle', 'rb') as handle:
    x = pickle.load(handle)
  with open(path + 'y_compact.pickle', 'rb') as handle:
    y = pickle.load(handle)
  return (x, y)

def make_doc2vec():
  model = KeyedVectors.load_word2vec_format("/content/drive/MyDrive/swarog/models/GoogleNews-vectors-negative300-SLIM.bin", binary=True)
  with open("/content/myutils01/stopwords_en.txt", 'r') as fh:
    stopwords = fh.read().split(",")
  print("word2vec model loaded...",flush=True)
  ty,tn = garfn_ds.load_sumarised()
  ds = DocSim(model,stopwords=stopwords)
  t_fake_wor2vec_avg = [ds.vectorize(t) for t in ty]
  t_legit_wor2vec_avg = [ds.vectorize(t) for t in tn]

  x = t_fake_wor2vec_avg + t_legit_wor2vec_avg

  with open(path + 'wor2vec_avg.pickle', 'wb') as handle:
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def load_doc2vec():
    with open(path + 'wor2vec_avg.pickle', 'rb') as handle:
        x = pickle.load(handle)
    
    with open(path + 'y_compact.pickle', 'rb') as handle:
        y = pickle.load(handle)
    return (x, y)


def dump_bert_out(mojadata):
    with open(path + 'distilber_vectors.pickle', 'wb') as handle:
        pickle.dump(mojadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_bert_out():
    with open(path + 'distilber_vectors.pickle', 'rb') as handle:
        mojadata = pickle.load(handle)
    
    with open(path + 'y_compact.pickle', 'rb') as handle:
        y = pickle.load(handle)
    
    return (mojadata,np.array(y))



def dump_bert_emb(tokenized_text_d):
    with open(path + 'distilber_embed.pickle', 'wb') as handle:
        pickle.dump(tokenized_text_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def tokenize_with_distilbert():
    x,y = load_raw_txt()
    df = pd.DataFrame({'text':x, 'label':y})
    tokenized_text = tokenize(df['text'].tolist())
    tokenized_text_d = dict(tokenized_text)
    dump_bert_emb(tokenized_text_d)
    

from utils.models import tokenize
def load_bert_emb():
    with open(path + 'distilber_embed.pickle', 'rb') as handle:
        tokenized_text_d = pickle.load(handle)
    
    with open(path + 'y_compact.pickle', 'rb') as handle:
        y = pickle.load(handle)  
        
    
    return (tokenized_text_d, np.array(y))
 
 

