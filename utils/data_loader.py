import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, DistilBertConfig, TFDistilBertModel, DistilBertTokenizerFast

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


from gensim.models.keyedvectors import KeyedVectors
from datetime import datetime
import numpy as np

import string
import re
import sqlite3
import time      

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=   

class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords
        
    def mean_pool(self, word_vecs):
        if len(word_vecs) == 0:
            return np.array(300*[0])

        vector = np.mean(word_vecs, axis=0)
        return vector
    
    def topN(self, word_vecs, top_nnum):
        for i in range(len(word_vecs),top_nnum):
            word_vecs.append(300*[0])
        vector = word_vecs
        return np.array(vector)

    def vectorize(self, doc, mode="avg", top_nnum=64):
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

        if mode=="avg":
            return self.mean_pool(word_vecs)
        elif mode=="topN":
            return self.topN(word_vecs, top_nnum)

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=   



MAX_SEQUENCE_LENGTH = 256 
top_words = 7000
weights = {0:1, 1:1}
keylen = 100


class DataLoader():
    def store_results(self,wyniki,dataset,method,db='swarog_wyniki_v1.sqlite'):
        curt = time.time()
        conn = sqlite3.connect(db)
        conn.execute("""
            create table if not exists 
            wyniki(date text, dataset text, method text, metric text, value real)
        """)

        cursor = conn.cursor()

        statement = """
            replace into  wyniki values(?,?,?,?,?)
        """

        for metric in wyniki.keys():
            for v in wyniki[metric]['list']:
                tp = (curt,dataset,method,metric, v)
                #print(tp)
                cursor.execute(statement,tp)
        conn.commit()    
        cursor.close()
    

    def setpaths(self,dsname='fakenewskdd2020', readcb=None):
        self.readcb = readcb
        self.datasetname = dsname
        self.pickleversion = -1
        print("Using", self.datasetname, 'dataset')
        try:
            import google.colab
            self.IN_COLAB = True
            self.path = f'/content/drive/MyDrive/swarog/datasets/{self.datasetname}/'
            self.modelspath=f"/content/drive/MyDrive/swarog/models/{self.datasetname}/"
            self.modelspath2=f"/content/drive/MyDrive/swarog/models/"
            self.pickleversion = 5
            print('using colab')
            print(self.path)
            print(self.modelspath)
        except:
            self.IN_FNKDDCOLAB = False
            import os
            if 'nt' == os.name:
                self.pickleversion = 5
                print("using my windows...")
                self.path = f'c:/Users/demo/Desktop/sciwork/data/swarog/datasets/{self.datasetname}/'
                self.modelspath = f'c:/Users/demo/Desktop/sciwork/data/swarog/models/{self.datasetname}/'
                self.modelspath2 = f'c:/Users/demo/Desktop/sciwork/data/swarog/models/'
            else:
                print('using local')
                self.path = f'/media/rkozik/0C129CFC129CEBC8/data/swarog/datasets/{self.datasetname}/'
                self.modelspath=f"/media/rkozik/0C129CFC129CEBC8/data/swarog/models/{self.datasetname}/"
                self.modelspath2=f"/media/rkozik/0C129CFC129CEBC8/data/swarog/models/"



    def set_path(self,x):
        self.path = x
        
        
    def read(self):
        if self.readcb:
            return self.readcb(self)
        
        #
        # Default parsers!
        #
        if self.datasetname == "grafn":
            from pandasql import sqldf
            _fake = pd.read_csv(self.path + "badnews.csv", sep=",").dropna()
            _true = pd.read_csv(self.path + "goodnews.csv", sep=",").dropna()
            real = sqldf("select text, 0 as label from _true where length(text) > 200 limit 20000", locals())
            print("GRAFN LABELS:\n Fake=",_fake.shape[0], " True=", _true.shape[0])
            return _fake, real
        
        if self.datasetname == "isot":
            _fake = pd.read_csv(self.path + "Fake.csv", sep=",").dropna()
            _true = pd.read_csv(self.path + "True.csv", sep=",").dropna()
            print("ISOT LABELS:\n Fake=",_fake.shape[0], " True=", _true.shape[0])
            return _fake, _true
  
        if self.datasetname == "fakenewskdd2020":
            from pandasql import sqldf
            print("load",self.path + "train.csv")
            dsfile = pd.read_csv(self.path + "train.csv",sep="\t")
            fake = sqldf("select text, label from dsfile where label=1 and length(text) > 10", locals())
            real = sqldf("select text, label from dsfile where label=0 and length(text) > 10", locals())
            print("FN-KDD LABELS:\n Fake=",fake.shape[0], " True=", real.shape[0])
            return fake, real
        #
        # orgins:
        # https://zenodo.org/record/4282522#.Yga6zt-M6Lk
        if self.datasetname == "covid19fn":
            from pandasql import sqldf
            print("load",self.path + "data.csv")
 
            # yes, labels are swpped 
            fake = sqldf("select headlines as text, outcome from dsfile where outcome=0 and length(text) > 10", locals())
            real = sqldf("select headlines as text, outcome from dsfile where outcome=1 and length(text) > 10", locals())
            print("Covid19 LABELS:\n Fake=",fake.shape[0], " True=", real.shape[0])
            return fake, real

        if self.datasetname == "mmcovid":
            from pandasql import sqldf
            print("load",self.path + "news_collection.csv")
            dsfile = pd.read_csv(self.path + "news_collection.csv",sep="\t")
            fake = sqldf("select text, label from dsfile where label='fake' and lang='en' and length(text) > 10", locals())
            real = sqldf("select text, label from dsfile where label='real' and lang='en' and length(text) > 10", locals())
            print("MM-Covid19 LABELS:\n Fake=",fake.shape[0], " True=", real.shape[0])
            return fake, real
        
        
        if self.datasetname == "pubhealth":
            from pandasql import sqldf
            print("load",self.path + "train.tsv")
            dsfile = pd.read_csv(self.path + "train.tsv",sep="\t")
            fake = sqldf("""select main_text as text, 
                            label from dsfile where label='false' and length(main_text) > 10""", locals())
            real = sqldf("""select main_text as text, 
                            label from dsfile where label='true' and length(main_text) > 10""", locals())
            print("PUBHealth LABELS:\n Fake=",fake.shape[0], " True=", real.shape[0])
            return fake, real

        if self.datasetname == "politifact":
            from pandasql import sqldf
            print("load",self.path + "train.csv")
            dsfile = pd.read_csv(self.path + "train.csv",sep="\t")
            real = sqldf("""select Statement as text, Rating as
                            label from dsfile where Rating <= 2 and length(Statement) > 10""", locals())
            fake = sqldf("""select Statement as text, Rating as
                            label from dsfile where Rating > 2 and length(Statement) > 10""", locals())
            print("PolitiFact LABELS:\n Fake=",fake.shape[0], " True=", real.shape[0])
            return fake, real
 

    def sumarise_text(self,text):
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
            if (i+1) >= len(arr) or acum + arr[i+1][0] >= MAX_SEQUENCE_LENGTH:
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

    def sumarise_dataset(self,txtN):
        i = 0
        print("sumarise_dataset")
        for it in range(0,len(txtN)):
            doc = txtN[it]
            if i%6000 == 0 or (i+1)==len(txtN):
                print(i+1,"/",len(txtN))
            i=i+1  
            if len(doc) < 5*MAX_SEQUENCE_LENGTH:
                continue
            words = doc.split(" ")
            if len(words) > MAX_SEQUENCE_LENGTH:
                sd = self.sumarise_text(doc)
                doc_len = len(sd)
                if doc_len < 10:
                    pass
                    # print("wird length of sumarised sentence len=",doc_len,"was", len(doc) )
                else:
                    txtN[it] = sd
        return txtN    

    def remove_stop_words(self,fakes):
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in words]
        # stripped
        
        
        stop_words = set(stopwords.words())
        txtY = []
        i = 0
        print("remove_stop_words")
        for text in fakes:
            #text_tokens = (text.replace("\n","")).split(" ")
            inwords = re.split(r'\W+', text)
            table = str.maketrans('', '', string.punctuation)
            text_tokens = [w.translate(table) for w in inwords]
            tokens_without_sw = [word for word in text_tokens if len(word) > 0 and not word.lower() in stop_words]
            txtY.append(" ".join(tokens_without_sw))
            if i%6000 == 0 or (i+1)==len(fakes):
                print(i+1,"/",len(fakes))
            i=i+1
        return txtY

    def store_sumarised(self,txtY, txtN):
        with open(self.path + 'badnews_sumarised.pickle', 'wb') as handle:
            pickle.dump(txtY, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.path + 'goodnews_sumarised.pickle', 'wb') as handle:
            pickle.dump(txtN, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("stored")

    def load_sumarised(self):
        with open(self.path + 'badnews_sumarised.pickle', 'rb') as handle:
            txtY = pickle.load(handle)
        with open(self.path + 'goodnews_sumarised.pickle', 'rb') as handle:
            txtN = pickle.load(handle)
        return (txtY,txtN)

    def fit_tokenizer(self, txtY, txtN):
        docs = txtY + txtN
        myTokenizer = Tokenizer(top_words)
        myTokenizer.fit_on_texts(docs)
        # saving
        with open(self.path + 'dtokenizer.pickle', 'wb') as handle:
            pickle.dump(myTokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("done")
        return myTokenizer

    def tokenize_corpus(self,txtY,txtN,t):
        s1 = t.texts_to_sequences(txtY)
        s2 = t.texts_to_sequences(txtN)
        s1pad =keras.preprocessing.sequence.pad_sequences(s1, MAX_SEQUENCE_LENGTH)
        s2pad =keras.preprocessing.sequence.pad_sequences(s2, MAX_SEQUENCE_LENGTH)
        s1lab = [1 for i in range(0,len(txtY))]
        s2lab = [0 for i in range(0,len(txtN))]
        x = s1pad.tolist() + s2pad.tolist()
        y = s1lab+s2lab
        with open(self.path + 'x_compact.pickle', 'wb') as handle:
            pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.path + 'y_compact.pickle', 'wb') as handle:
            pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def make_compact(self):
        fake, good = self.read()
        txtY = self.remove_stop_words(fake['text'])
        txtN = self.remove_stop_words(good['text'])
#         txtY=self.sumarise_dataset(txtY)
#         txtN=self.sumarise_dataset(txtN)
        
        self.store_sumarised(txtY, txtN)
        t = self.fit_tokenizer(txtY,txtN)
        self.tokenize_corpus(txtY,txtN,t)


    def load_raw_txt(self):
        fake, good = self.read()    
        tn = good['text'].tolist()
        ty = fake['text'].tolist()
        ylab = [1 for i in range(0,len(ty))]
        nlab = [0 for i in range(0,len(tn))]
        return (ty+tn, ylab+nlab)
    
 
    def load_raw_txt_as_pandas(self):
        txt, lab = self.load_raw_txt()
        return pd.DataFrame({'text':txt, 'labels':lab})
 

    def load_compact(self):
        with open(self.path + 'x_compact.pickle', 'rb') as handle:
            x = pickle.load(handle)
        with open(self.path + 'y_compact.pickle', 'rb') as handle:
            y = pickle.load(handle)
        return (np.array(x), np.array(y))

    def make_doc2vec(self, mode="avg", top_nnum=MAX_SEQUENCE_LENGTH, data=None):
        self.model = KeyedVectors.load_word2vec_format(self.modelspath2 + 
                                                       'GoogleNews-vectors-negative300-SLIM.bin', binary=True)
        with open( self.modelspath2 + "stopwords_en.txt", 'r') as fh:
            stopwords = fh.read().split(",")
        print("word2vec model loaded...",flush=True)
        
        if data is None:
            ty,tn = self.load_sumarised()
        else:
            ty,tn = data
    
        ds = DocSim(self.model,stopwords=stopwords)
       
        t_fake_wor2vec =  [ds.vectorize(t, mode, top_nnum) for t in ty]
        t_legit_wor2vec_avg = [ds.vectorize(t, mode, top_nnum) for t in tn]
        print(np.array(t_fake_wor2vec).shape)
        
        if data is None:
            x = t_fake_wor2vec + t_legit_wor2vec_avg
            filepath = self.path + 'wor2vec_'+mode+'.pickle'
            print("writig to",filepath)
            with open(filepath, 'wb') as handle:
                pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return x
        
    def sumarized_to_csv(self):
        ty,tn = self.load_sumarised()
        ylab = [1 for i in range(0,len(ty))]
        nlab = [0 for i in range(0,len(tn))]
        pdf = pd.DataFrame({'x':ty+tn , 'y':ylab+nlab})
        pth = self.path + 'sumarized.csv'
        pdf.to_csv(pth)
        print('saved to',pth)
        
    def sumarized_to_flair_corpus(self):
        from pandasql import sqldf
        ty,tn = self.load_sumarised()
        ylab = ['__label__fake' for i in range(0,len(ty))]
        nlab = ['__label__real' for i in range(0,len(tn))]
        pdf = pd.DataFrame({'text':ty+tn , 'label':ylab+nlab})
        
        data = sqldf("select label, text  from pdf order by random()", locals())
        
        pth = self.path + 'flair_sumarized_'
        
        data.iloc[0:int(len(data)*0.8)].to_csv(pth + 'train.csv', sep='\t', index = False, header = False)
        data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv(pth + 'test.csv', sep='\t', index = False, header = False)
        data.iloc[int(len(data)*0.9):].to_csv(pth + 'dev.csv', sep='\t', index = False, header = False);
  
        print('saved to',pth+'*')

    def load_doc2vec(self, mode="avg"):
        filepath = self.path + 'wor2vec_'+mode+'.pickle'
        print("opening",filepath)
        with open(filepath, 'rb') as handle:
            x = pickle.load(handle)
        with open(self.path + 'y_compact.pickle', 'rb') as handle:
            y = pickle.load(handle)
        return (np.array(x), np.array(y))

    
    #
    #
    # ##### BERT
    # 
    #
    
    # -------------------------  tokenizer output   ------------------------------------------
    
    def tokenize_with_distilbert(self, sumarized_text=False):
        def tokenize(sentences, max_length=MAX_SEQUENCE_LENGTH, padding='max_length'):
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            return tokenizer(
                sentences,
                truncation=True,
                padding=padding,
                max_length=max_length,
                return_tensors="tf"
            )
        
        if sumarized_text == False:
            x,y = self.load_raw_txt()
            df = pd.DataFrame({'text':x, 'label':y})
            tokenized_text = tokenize(df['text'].tolist())
            tokenized_text_d = dict(tokenized_text)
            self.dump_bert_emb(tokenized_text_d)
        
        else:
            with open(self.path + 'y_compact.pickle', 'rb') as handle:
                y = pickle.load(handle)
            f,r = self.load_sumarised()
            df = pd.DataFrame({'text':(f+r), 'label':y})
            tokenized_text = tokenize(df['text'].tolist())
            tokenized_text_d = dict(tokenized_text)
            self.dump_bert_emb(tokenized_text_d,sumarized_text=sumarized_text)
 

    # dump tokenized (with DistilBertTokenizerFast) text
    def dump_bert_emb(self,tokenized_text_d, sumarized_text=False):
        name = 'distilber_embed.pickle' if sumarized_text == False else 'distilber_sumarized_embed.pickle'
        with open(self.path + name, 'wb') as handle:
            print('  opened',self.path + name)
            pickle.dump(tokenized_text_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    
    def load_bert_emb(self,sumarized_text=False):
        name = 'distilber_embed.pickle' if sumarized_text == False else 'distilber_sumarized_embed.pickle'
        with open(self.path + 'y_compact.pickle', 'rb') as handle:
            y = pickle.load(handle)
        with open(self.path + name, 'rb') as handle:
            print('  opened',self.path + name)
            tokenized_text_d = pickle.load(handle)
        return (tokenized_text_d, np.array(y))

    # -------------------------  model output   ------------------------------------------

    # CLS output from BERT model
    def dump_bert_out(self, sumarized_text=False):
        print("dump_bert_out, sumarized_text:",sumarized_text)
        base = TFDistilBertModel.from_pretrained(
            'distilbert-base-uncased',
        )
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')
        layoutput = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]
        new_model = tf.keras.models.Model(inputs=[input_ids, attention_mask], 
                                      outputs=layoutput)

        tokenized_text_d,npy=self.load_bert_emb(sumarized_text=sumarized_text) 
            
        indata = tf.data.Dataset.from_tensor_slices((
            (tokenized_text_d),  # Convert BatchEncoding instance to dictionary
            npy
        )).batch(64).prefetch(2)
        
        print("predict")
        mojadata = new_model.predict(indata,batch_size=64, verbose=2)
        name = 'distilber_vectors.pickle' if sumarized_text == False else 'distilber_sumarized_vectors.pickle'
        with open(self.path + name, 'wb') as handle:
            print('  opened',self.path + name)
            pickle.dump(mojadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    # load bert model outputs        
    def load_bert_out(self, sumarized_text=False):
        name = 'distilber_vectors.pickle' if sumarized_text == False else 'distilber_sumarized_vectors.pickle'
        print("dump_bert_out, sumarized_text:",sumarized_text)
        with open(self.path + name, 'rb') as handle:
            print('  opened',self.path + name)
            mojadata = pickle.load(handle)
        with open(self.path + 'y_compact.pickle', 'rb') as handle:
            y = pickle.load(handle)
        return (mojadata,np.array(y))



    
_dl__ = DataLoader()
_dl__.setpaths()

if _dl__.pickleversion == 5:
    import pickle5 as pickle
    print("using pickle5")
else: 
    import pickle
    


