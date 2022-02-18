from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf


class BaseLine:
    name = "Baseline"
    def __init__(self, class_weight={0:1, 1:1}, trees=50, verbose=0):
        self.verbose=verbose
        self.trees = trees
        self.class_weight = class_weight
        
    def predict(self, trn_x, trn_y, val_x, val_y):
        rf = RandomForestClassifier(n_estimators = self.trees,class_weight= self.class_weight)
        rf.fit(trn_x, trn_y)
        pred_proba = rf.predict_proba(val_x)
        pred = rf.predict(val_x)
        p = np.array([pred_proba[i][cl] for i,cl in enumerate(pred)])
        if self.verbose > 0:
            print(classification_report(val_y, pred))
        return pred,p
    
    
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from pandasql import sqldf
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

class Baseline2:
    name = "Tfidf+RF"
    def __init__(self, class_weight={0:1, 1:1}, trees=50, replicatedata=1, max_features=1024, verbose=0):
        self.trees = trees
        self.class_weight = class_weight
        self.replicatedata = replicatedata
        self.max_features=max_features
        self.verbose=verbose

    def clean(self, text):
        wn = nltk.WordNetLemmatizer()
        stopword = nltk.corpus.stopwords.words('english')
        tokens = nltk.word_tokenize(text)
        lower = [word.lower() for word in tokens]
        no_stopwords = [word for word in lower if word not in stopword]
        no_alpha = [word for word in no_stopwords if word.isalpha()]
        lemm_text = [wn.lemmatize(word) for word in no_alpha]
        clean_text = lemm_text
        return clean_text
    
    def vectorize(self,data,tfidf_vect_fit):
        X_tfidf = tfidf_vect_fit.transform(data)
        words = tfidf_vect_fit.get_feature_names()
        X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
        X_tfidf_df.columns = words
        return(X_tfidf_df)
        
    def predict(self, trn_x, trn_y, val_x, val_y):
        
        trDF = pd.DataFrame({'raw':trn_x, 'labels':trn_y})
        clean_text=trDF['raw'].map(self.clean)
        trDF['text']=[ " ".join(x) for x in clean_text ]

        if self.replicatedata == 1:
            train_80 = sqldf("""
            select text,labels from (select * from trDF 
                where 
                length(text) between 20 and 100
                and labels=1 order by random() limit 9000)
            union all
            select text,labels from (
            select substr( a.text, 0, length(a.text)*0.6 ) || substr( b.text, length(b.text)*0.6 )  as text, 0 as labels 
            --select a.text  ||" "||  b.text  as text, 0 as labels 
            from trDF as a join trDF as b 
                where 
                (length(a.text) + length(b.text)) between 20 and 100 
                and a.labels=0 and b.labels=0 order by random() limit 9000 )
            -- union all
            -- select text, labels from good
            """,locals())
        elif self.replicatedata == 2:
            train_80 = sqldf("""
            select text,labels from (select * from trDF 
                where 
                length(text) between 20 and 10000
                and labels=1 order by random())
            union all
            select text,labels from (select * from trDF 
                where 
                length(text) between 20 and 10000
                and labels=0 order by random())
            union all
            select text,labels from (
                select substr( a.text, 0, length(a.text)*0.5 ) || substr( b.text, length(b.text)*0.5 )  as text, 0 as labels 
            from trDF as a join trDF as b 
                where 
                (length(a.text) + length(b.text)) between 20 and 10000 
                and a.labels=0 and b.labels=0 order by random() limit 3000 )
            union all
            select text,labels from (
                select substr( a.text, 0, length(a.text)*0.5 ) || substr( b.text, length(b.text)*0.5 )  as text, 1 as labels 
            from trDF as a join trDF as b 
                where 
                (length(a.text) + length(b.text)) between 20 and 10000 
                and a.labels=1 and b.labels=1 order by random() limit 6000 )
            """,locals())
        else:
            train_80=trDF
        
        
        #tfidf_vect = TfidfVectorizer(max_features=1024)
        tfidf_vect = TfidfVectorizer(max_features=self.max_features)
        tfidf_vect_fit=tfidf_vect.fit(trDF['text'])
        
        #rf = RandomForestClassifier(n_estimators=42, max_depth=42, max_features=42)
        rf = RandomForestClassifier(n_jobs=-1,n_estimators=self.trees, class_weight= self.class_weight)
        
        X_train_feat=self.vectorize(train_80['text'],tfidf_vect_fit)
        rf.fit(X_train_feat,train_80['labels'].values)

        X_test_feat=self.vectorize(val_x,tfidf_vect_fit)

        pred_proba = rf.predict_proba(X_test_feat)
        pred = rf.predict(X_test_feat)
        p = np.array([pred_proba[i][cl] for i,cl in enumerate(pred)])
        
        if self.verbose > 0:
            print(classification_report(val_y, pred))
            print("balanced_accuracy\n",balanced_accuracy_score(val_y, pred))
            print("confusion_matrix\n",confusion_matrix(val_y, pred))
            
        return pred,p
        
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import callbacks
from sklearn.metrics import confusion_matrix


MAX_SEQUENCE_LENGTH = 256 
top_words = 7000
weights = {0:1, 1:10}

top_words = 7000
 
        
class CNN2:
    def __init__(self, class_weight={0:1, 1:5}, lr = 0.001, epochs=5, verbose=0):
        self.class_weight = class_weight
        self.lr = lr
        self.epochs = epochs
        self.verbose=verbose
    
    name = "cnn"
    def predict(self, trn_x, trn_y, val_x, val_y):
        mycnn = Sequential()      # initilaizing the Sequential nature for CNN model
        mycnn.add(Embedding(top_words, 32, input_length=MAX_SEQUENCE_LENGTH))
        
        
        mycnn.add(Conv1D(MAX_SEQUENCE_LENGTH, 12, padding='same', activation='relu' ))
        
        mycnn.add(tf.keras.layers.BatchNormalization())
        
        mycnn.add(MaxPooling1D())
        mycnn.add(Conv1D(MAX_SEQUENCE_LENGTH/2, 6, padding='same', activation='relu'))
        mycnn.add(MaxPooling1D())
        mycnn.add(Conv1D(MAX_SEQUENCE_LENGTH/4, 7, padding='same', activation='relu'))
        
       # mycnn.add(Dropout(0.1))
        
        mycnn.add(MaxPooling1D())
        mycnn.add(Conv1D(MAX_SEQUENCE_LENGTH/8, 5, padding='same', activation='relu' ))
        mycnn.add(MaxPooling1D())
        mycnn.add(Conv1D(MAX_SEQUENCE_LENGTH/16, 3, padding='same', activation='relu' ))
        mycnn.add(MaxPooling1D())
        
        mycnn.add(tf.keras.layers.BatchNormalization())
        
        mycnn.add(Flatten())
        mycnn.add(Dense(140, activation='relu'))
        mycnn.add(Dense(30, activation='relu'))
        mycnn.add(Dense(10, activation='relu'))
        mycnn.add(Dense(1, activation='sigmoid'))

        mycnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                            mode ="min", patience = 5, 
                                            restore_best_weights = True)

        mycnn.fit(trn_x, trn_y, validation_data=(val_x, val_y), epochs=self.epochs, 
                  batch_size=64, verbose=self.verbose, class_weight=self.class_weight)
        self.model = mycnn
   
        pred, prob = self.forward(trn_x)        
        
        if self.verbose > 0:
            print(classification_report(trn_y, pred))
            print(confusion_matrix(trn_y, pred))
            print("----")

        pred, prob = self.forward(val_x)

        if self.verbose > 0:
            print(classification_report(val_y, pred))
            print(confusion_matrix(val_y, pred))

        return pred, prob

    def forward(self, val_x):
        mycnn = self.model
        ypred = mycnn.predict(val_x, batch_size=64, verbose=0)
        return [1 if y > 0.5 else 0 for y in ypred], ypred
    

#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#


from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow import keras

class LSTMModel:
    name = "lstm"
    
    def __init__(self, class_weight={0:1, 1:1}, lr = 0.001, epochs=5, verbose=0):
        self.class_weight = class_weight
        self.lr = lr
        self.epochs = epochs
        self.verbose=verbose

    def predict(self, trn_x, trn_y, val_x, val_y):
        embedding_vector_features=32
        model=Sequential()
        model.add(Embedding(top_words,embedding_vector_features,input_length=MAX_SEQUENCE_LENGTH))
        ##model.add(Dropout(0.7))
        model.add(Bidirectional(LSTM(32)))
        
        model.add(tf.keras.layers.Dropout(
            rate=0.15,
            name="01_dropout",
        ))

        model.add(tf.keras.layers.Dense(
            units=16,
            kernel_initializer='glorot_uniform',
            activation=None,
            name="01_dense_relu_no_regularizer",
        ))
                   
        model.add(tf.keras.layers.BatchNormalization(
            name="01_bn"
        ))
                  
        
        ##model.add(Dropout(0.7))
        model.add(Dense(1,activation='sigmoid'))
        
        METRICS = [
              keras.metrics.BinaryAccuracy(name='accuracy'),
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall') # precision-recall curve
        ]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                   loss='binary_crossentropy',
                      metrics=METRICS)

        earlystopping = callbacks.EarlyStopping(monitor ="accuracy", 
                                            mode ="min", patience = 5, 
                                            restore_best_weights = True)
        model.fit(trn_x, trn_y, epochs=3, verbose=self.verbose, callbacks =[earlystopping], validation_data=(val_x,val_y))
        self.model = model
        yprob = np.array([v[0] for v in model.predict(val_x, batch_size=64, verbose=0)])
        
        #print(yprob,yprob.shape)
        ##print("--\n",trn_y)
 
        ypred = [1 if y > 0.5 else 0 for y in yprob]
       
        if self.verbose:
            print(classification_report(val_y, ypred))
        
        return ypred, yprob
    
        

    def forward(self, val_x, val_y):
        model = self.model
        modelpred = np.array([v[0] for v in model.predict(val_x, batch_size=64, verbose=0)])
        pred = [1 if y > 0.5 else 0 for y in modelpred]
        return pred, modelpred


#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils import shuffle
import tensorflow as tf
from transformers import AutoTokenizer, DistilBertConfig, TFDistilBertModel, DistilBertTokenizerFast
from tensorflow.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from sklearn.metrics import classification_report
#from datasets import Dataset
from transformers import DataCollatorWithPadding
import warnings
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
import nltk
from nltk.corpus import stopwords

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.metrics import geometric_mean_score  

 

class BertLSTM:
    name="bert_lstm"
    arch_created = False
    
    def __init__(self, class_weight={0:1, 1:1}, verbose=0):
        self.class_weight = class_weight
        self.verbose=verbose
    
    def load_bert_base(self):
        config = DistilBertConfig()
        config.output_hidden_states = False
        # Preliminary BERT layer configuration
        config = DistilBertConfig()
        config.output_hidden_states = False
        transformer_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config) 
        MAX_LEN_SEQUENCE = 64
        base = TFDistilBertModel.from_pretrained(
            'distilbert-base-uncased',
            # num_labels=NUM_LABELS    # no effect
        )
        for layer in base.layers:
            layer.trainable = False
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')
        
        bertbase = tf.keras.models.Model(inputs=[input_ids, attention_mask], 
                               outputs=base([input_ids, attention_mask]).last_hidden_state, name="bert_base")
       
        return bertbase

    def build_model(self):
        self.base = self.load_bert_base()
        inlayer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,768,), dtype=tf.float32, name='bertout')
        X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(25, return_sequences=True, 
                                                               dropout=0.6, recurrent_dropout=0.6))(inlayer)
        X = tf.keras.layers.GlobalMaxPool1D()(X)
        X = tf.keras.layers.Dense(25, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.7)(X)
        X = tf.keras.layers.Dense(2, activation='softmax')(X)

        head = tf.keras.models.Model(inputs=inlayer, 
                                      outputs=X, name="head")
        
        self.head = head
 

    def predict(self, trn_x, trn_y, val_x, val_y): 
        NUM_EPOCHS = 5
        BATCH_SIZE = 32
        LEARNING_RATE = 1e-2  
        L2 = 1e-3
        REDUCE_LR_PATIENCE = 1
        EARLY_STOP_PATIENCE = 3

        tf.keras.backend.clear_session()

        if not self.arch_created: 
            self.build_model()
            self.arch_created = True

        model = tf.keras.models.Model(inputs=self.base.inputs, outputs=self.head(self.base(self.base.inputs)), name="bert")

        model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          optimizer=tf.keras.optimizers.Adam(),
          metrics=['accuracy']
        )

        self.model = model


        BATCH_SIZE = 64

        t_input_ids, t_attention_mask = zip(*trn_x)
        v_input_ids, v_attention_mask = zip(*val_x)

        trndata = {
          'attention_mask': tf.convert_to_tensor(t_attention_mask),
          'input_ids': tf.convert_to_tensor(t_input_ids)
        }

        evaldata = {
          'attention_mask': tf.convert_to_tensor(v_attention_mask),
          'input_ids': tf.convert_to_tensor(v_input_ids)
        }


        X = tf.data.Dataset.from_tensor_slices((
          (trndata),  # Convert BatchEncoding instance to dictionary
          trn_y
        )).batch(BATCH_SIZE).prefetch(1)

        V = tf.data.Dataset.from_tensor_slices((
          (evaldata),  # Convert BatchEncoding instance to dictionary
          val_y
        )).batch(BATCH_SIZE).prefetch(1)

        model.fit(
          x=X,    # dictionary 
          # y=Y,
          y=None,
          epochs=1,
          batch_size=64, 
          verbose=self.verbose
        )
        xpred = model.predict(V,batch_size=64)
        pred = np.argmax(xpred, axis=-1)
        pred_proba = xpred[:, 1]

        if self.verbose > 0:
            print(classification_report(val_y, pred))
    
        return pred, pred_proba


    def forward(self, val_x,val_y):
        model = self.model
        BATCH_SIZE = 32
        v_input_ids, v_attention_mask = zip(*val_x)
        evaldata = {
        'attention_mask': tf.convert_to_tensor(v_attention_mask),
        'input_ids': tf.convert_to_tensor(v_input_ids)
        }

        V = tf.data.Dataset.from_tensor_slices((
        (evaldata),  # Convert BatchEncoding instance to dictionary
        val_y
        )).batch(BATCH_SIZE).prefetch(1)

        xpred = model.predict(V,batch_size=64)
        pred = np.argmax(xpred, axis=-1)
        pred_proba = xpred[:, 1]

        if self.verbose > 0:
            print(classification_report(val_y, pred))

        return pred, pred_proba
    
          
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#

from keras import backend as K
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from sklearn.metrics import classification_report



class BertHead:
    name = "bert_head"
    
    def __init__(self, class_weight={0:1, 1:5}, lr = 0.001, epochs=5,batch_size=1024,verbose=0):
        self.class_weight = class_weight
        self.lr = lr
        self.epochs = epochs
        self.batch_size=batch_size
        self.verbose=verbose
    
    def predict(self,trn_x, trn_y, val_x, val_y): 
        tf.keras.backend.clear_session()
        mycnn = Sequential()      # initilaizing the Sequential nature for CNN model
        
        mycnn.add(tf.keras.layers.BatchNormalization())
        mycnn.add(Dropout(0.4))
        
           
        mycnn.add(Dense(240,  activation="relu"))
        
        mycnn.add(Dropout(0.3))
 
        mycnn.add(Dense(140, activation='relu'))
        mycnn.add(Dense(70, activation='relu'))
        
       
        mycnn.add(Dense(140, activation='relu'))
        mycnn.add(Dense(70, activation='relu'))
       
    
        mycnn.add(Dropout(0.2))
        mycnn.add(Dense(20, activation='relu'))
        mycnn.add(Dense(2, activation='softmax'))

        mycnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


        if self.verbose > 0:
            print(trn_x.shape, trn_y.shape,self.class_weight)
        
        

        mycnn.fit(trn_x, trn_y, epochs=self.epochs, 
                  batch_size=self.batch_size, verbose=self.verbose, class_weight=self.class_weight)

        self.model = mycnn

        pred, prob = self.forward(trn_x)       

        if self.verbose > 0: 
            print(classification_report(trn_y, pred))
            print(confusion_matrix(trn_y, pred))
            
            print("----")

        pred, prob = self.forward(val_x)    

        if self.verbose > 0:    
            print(classification_report(val_y, pred))
            print(confusion_matrix(val_y, pred))

        return pred, prob
    
    def forward(self, val_x):
        mycnn = self.model
        mout = mycnn.predict(val_x, batch_size=64, verbose=0)
        pred = np.argmax(mout, axis=-1)
        prob = mout[:, 1]
        return pred, prob
 
