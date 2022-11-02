import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON

import transformers
transformers.logging.set_verbosity_error()
from torch.utils.data import DataLoader 

from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import TFDistilBertModel, DistilBertTokenizerFast
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

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
    
    return {'result': result, 
            'result_proba': result_proba, 
            'domain': category, 
            'domain_proba' : category_proba
           }