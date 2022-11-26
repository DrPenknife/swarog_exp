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

PATH_PICKLES = './pickles'

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "lematizer"
        return super().find_class(module, name)

print("load vectorizer")    
with open(f'{PATH_PICKLES}/tfidf_vectorizer_full.pickle', 'rb') as handle:
    unp = MyCustomUnpickler(handle)
    tfidf_vectorizer = unp.load()
    
vocabulary_tfidf_words = tfidf_vectorizer.get_feature_names_out()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device:", device)

if "disilbert_model" not in locals():
    disilbert_tokenizer =  AutoTokenizer.from_pretrained("distilbert-base-uncased")
    disilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    handle = disilbert_model.to(device)

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
            batchenc = disilbert_tokenizer(batch, max_length=256, 
                                           truncation=True, padding=True, return_tensors="pt")
            input_ids = batchenc['input_ids'].to(device)
            attention_mask = batchenc['attention_mask'].to(device)
            batchout = disilbert_model(input_ids, attention_mask=attention_mask, 
                                       output_hidden_states=True)
            embeds = [vec[0].cpu().detach().numpy() for vec in batchout[1][-1]]
            allembeds.extend(embeds)
        return np.array(allembeds)


def get_related_docs(_sentence):
    # get tf-idf values
    vec = tfidf_vectorizer.transform([_sentence]).toarray()[0]
    
    # zip (word, word_wieght)
    _words = sorted(list(zip(vec[np.where(vec > 0)], vocabulary_tfidf_words[np.where(vec > 0)[0]])), 
                    key=lambda tup: -tup[0])

    conn = sqlite3.connect(f'{PATH_PICKLES}/swarog.sqlite')
    wordmap = {}
    # in which doc is the word_id (iterate over)
    for _range in range(0,min(20,len(_words))):
        x=_words[_range]
        _chain = (x[0],re.sub(r'[^a-zA-Z0-9]', '', x[1]))
        if len(_chain[1]) == 0:
            continue
            
        c = conn.cursor()
        c.execute(f"""select rowid from rawsearch where body match '{_chain[1]}' limit 10000""")
        docsids = c.fetchall()
        for _r in docsids:
            if not _r[0] in wordmap:
                wordmap[_r[0]] = 0
            wordmap[_r[0]] = _chain[0] + 1 #+ wordmap[_r[0]]

    common = sorted(wordmap.items(), key=lambda tup: -tup[1])[:10]
    
    c = conn.cursor()
    resp = []
    for doc_index, _docid in enumerate(common):
        c.execute(f"""select label, dataset, body from rawsearch where rowid = {_docid[0]} """)
        _results_hits = c.fetchall()[0]
        resp.append({
            'text':_results_hits[2],
            'label':_results_hits[0],
            'dataset':_results_hits[1],
            'distance':1.0 - _docid[1]*1.0/min(20,len(_words))})

    conn.close()
    
    print(resp)
    return resp
    #_resp

    
    
#
# Unpickle models
#
          
bertemb = BERTEmbeddings()

with open(f'{PATH_PICKLES}/domain_cls.pickle', 'rb') as handle:
    pipe_domain = pickle.load(handle)
    
domain_model_pipe = []
for i in range(6):
    with open(f'{PATH_PICKLES}/model_{i}.pickle', 'rb') as handle:
        p=pickle.load(handle)
        domain_model_pipe.append(p)
        
          
def predict(_input):
    vec = bertemb.transform([_input])
    category = pipe_domain.predict(vec)[0]
    category_proba = pipe_domain.predict_proba(vec)[0]
    result = domain_model_pipe[1+category].predict(vec)[0]
    result_proba = domain_model_pipe[1+category].predict_proba(vec)[0]

    #similar_articles = get_related_articles(input_series['text'])
    #similar_articles = get_related_articles_bert(vec[0])
          
    similar_articles = get_related_docs(_input)
    
    return {'result': result, 
            'result_proba': result_proba,
            'domain': category, 
            'domain_proba' : category_proba,
            'similar_articles': similar_articles
           }