import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON
from annoy import AnnoyIndex
import re

import transformers
transformers.logging.set_verbosity_error()
from torch.utils.data import DataLoader 

from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import TFDistilBertModel, DistilBertTokenizerFast
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import pickle5 as pickle
import sqlite3
from lematizer import LemmaTokenizer

nltk.download('omw-1.4')

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "lematizer"
        return super().find_class(module, name)

print("load vectorizer")    
with open(f'../../pickles/swarog_data/tfidf_vectorizer_full.pickle', 'rb') as handle:
    unp = MyCustomUnpickler(handle)
    tfidf_vectorizer = unp.load()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device:", device)

if "disilbert_model" not in locals():
    disilbert_tokenizer =  AutoTokenizer.from_pretrained("distilbert-base-uncased")
    disilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    handle = disilbert_model.to(device)

# tfidf = AnnoyIndex(7000, 'angular')
# print("load tfidf annoy index...")
# tfidf.load('../../pickles/swarog_data/swarog_tfidf_7k.ann')

# bert = AnnoyIndex(768, 'angular')
# print("load bert annoy index...")
# bert.load('../../pickles/swarog_data/swarog_bertcls.ann')

def get_related_articles_ft5(txt):
    _names = tfidf_vectorizer.get_feature_names_out()
    _chain = sorted(list(zip(vec[np.where(vec > 0)], _names[np.where(vec > 0)[0]])), key=lambda tup: -tup[0])[:10]
    _chainstr = " ".join([re.sub(r'[^a-zA-Z0-9]', '', t[1]) for t in _chain])
    
    vec = tfidf_vectorizer.transform([txt]).toarray()[0]
    hits = tfidf.get_nns_by_vector(vec, 10, search_k=-1, include_distances=True)
    resp = []
    conn = sqlite3.connect('../../pickles/swarog_data/swarog.sqlite')
    c = conn.cursor()
    c.execute("""select rowid,label, dataset, body from rawsearch where body match ?""",_chainstr)
    r=c.fetchall()
    
    for index,_id in enumerate(hits[0]):
        c.execute("""select raw.body,raw.label, raw.dataset 
                from raw join tfidf on (tfidf.gid=raw.rowid) 
                where tfidf.rowid=?""",[_id+1])
        r=c.fetchall()
        resp.append({
            'text':r[0][0],
            'label':r[0][1],
            'dataset':r[0][2],
            'distance':hits[1][index]})
    conn.close()
    return resp


def get_related_articles(txt):
    vec = tfidf_vectorizer.transform([txt]).toarray()[0]
    hits = tfidf.get_nns_by_vector(vec, 10, search_k=-1, include_distances=True)
    resp = []
    conn = sqlite3.connect('../../pickles/swarog_data/swarog.sqlite')
    c = conn.cursor()
    for index,_id in enumerate(hits[0]):
        c.execute("""select raw.body,raw.label, raw.dataset 
                from raw join tfidf on (tfidf.gid=raw.rowid) 
                where tfidf.rowid=?""",[_id+1])
        r=c.fetchall()
        resp.append({
            'text':r[0][0],
            'label':r[0][1],
            'dataset':r[0][2],
            'distance':hits[1][index]})
    conn.close()
    return resp


def get_related_articles_bert(_vector):
    hits = bert.get_nns_by_vector(_vector, 10, search_k=-1, include_distances=True)
    resp = []
    conn = sqlite3.connect('../../pickles/swarog_data/swarog.sqlite')
    c = conn.cursor()
    for index,_id in enumerate(hits[0]):
        c.execute("""select raw.body,raw.label, raw.dataset 
                from raw join bertcls on (bertcls.gid=raw.rowid) 
                where bertcls.rowid=?""",[_id+1])
        r=c.fetchall()
        resp.append({
            'text':r[0][0],
            'label':r[0][1],
            'dataset':r[0][2],
            'distance':hits[1][index]})
    conn.close()
    return resp
    
    
class BERTEmbeddings:
    def __init__(self):
        self.tokenizer =  disilbert_tokenizer
        self.model = disilbert_tokenizer
        self.max_length = 256
        self.model_name = disilbert_model

    def fit(self, X=None, y=None):
        pass
    
    def encode(self, txt):
        return self.tokenizer(txt, max_length=self.max_length, 
                              truncation=True, padding=True, return_tensors="pt")

    def transform(self, X=None):
        dataloader = DataLoader(X, batch_size=4, shuffle=False)
        allembeds = []
        for batch in tqdm(dataloader):
            batchenc = disilbert_tokenizer(batch, max_length=256, truncation=True, padding=True, return_tensors="pt")
            input_ids = batchenc['input_ids'].to(device)
            attention_mask = batchenc['attention_mask'].to(device)
            batchout = disilbert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embeds = [vec[0].cpu().detach().numpy() for vec in batchout[1][-1]]
            allembeds.extend(embeds)
        return np.array(allembeds)


bertemb = BERTEmbeddings()
domain_cls = bentoml.sklearn.get("domain_cls:latest")

runners=[domain_cls.to_runner()]

for i in range(6):
    saved_model = bentoml.sklearn.get(f"model_{i}")
    runners.append(saved_model.to_runner())

model = bentoml.Service("swarog", runners=runners)

@model.api(input=JSON(), output=JSON())
def predict(input_series: np.ndarray) -> np.ndarray:
    vec = bertemb.transform([input_series['text']])
    category = runners[0].predict.run(vec)[0]
    category_proba = runners[0].predict_proba.run(vec)[0]
    result = runners[1+category].predict.run(vec)[0]
    result_proba = runners[1+category].predict_proba.run(vec)[0]

    similar_articles = get_related_articles(input_series['text'])
    #similar_articles = get_related_articles_bert(vec[0])
    
    return {'result': result, 
            'result_proba': result_proba,
            'domain': category, 
            'domain_proba' : category_proba,
            'similar_articles': similar_articles
           }