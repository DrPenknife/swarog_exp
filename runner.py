from utils.models import BertHead 
from utils.models import BertLSTM, BaseLine, CNN2, LSTMModel, Baseline2
from utils.data_loader import DataLoader
from utils import cv52
from utils import models
import numpy as np
import pandas as pd


dl = DataLoader() 
dl.setpaths('mmcovid', environment="lenovo")

print("running baseline")
model = BaseLine()
x, y = dl.load_doc2vec()
wynik = cv52.experiment(x,y,model) 
dl.store_results(wynik[0],dl.datasetname,model.name)

