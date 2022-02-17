import numpy as np 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold

from tqdm.notebook import tqdm

import numpy as np 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report

import numpy as np 
import pandas as pd
from IPython.display import display



def experiment_random_split(x,y, modelobj):
    #df = pd.DataFrame({'text':x,'label':y})
 
    #scores = cv52.__cross_validation_5_2_scores(pf_callback,df['text'].values,df['label'].values)
    #scores = cv52.__cross_validation_5_2_scores(pf_callback,x,y)
    #scores = costuniedziala(pf_callback,x,y)
    scores = __cross_random_split(modelobj,x,y)
    
    avg_scores_df=pd.DataFrame([(score_name,score_dict['avg']) for score_name, score_dict in scores.items()])
    avg_scores_df.columns = ["name","value"]
    display(avg_scores_df)

    return scores,avg_scores_df


def experiment_which_with_erros(x,y, txt,modelobj):
    #df = pd.DataFrame({'text':x,'label':y})
 
    #scores = cv52.__cross_validation_5_2_scores(pf_callback,df['text'].values,df['label'].values)
    #scores = cv52.__cross_validation_5_2_scores(pf_callback,x,y)
    #scores = costuniedziala(pf_callback,x,y)
    #scores = __cross_random_split(modelobj,x,y)
    __custom_eval(modelobj, x,y)
    
    pred, pred_proba = modelobj.forward(x, y)
    
    
    errors = np.where(np.array(pred)!=np.array(y))[0]
    
    print(errors)
    print(pred_proba[errors])


def experiment(x,y, modelobj):
    #df = pd.DataFrame({'text':x,'label':y})
 
    #scores = cv52.__cross_validation_5_2_scores(pf_callback,df['text'].values,df['label'].values)
    #scores = cv52.__cross_validation_5_2_scores(pf_callback,x,y)
    #scores = costuniedziala(pf_callback,x,y)
    scores = __cross_validation_5_2_scores(modelobj,x,y)
    
    avg_scores_df=pd.DataFrame([(score_name,score_dict['avg'],score_dict['std']) for score_name, score_dict in scores.items()])
    avg_scores_df.columns = ["name","avg","std"]
    display(avg_scores_df)

    return scores,avg_scores_df

global_scores={}


def __custom_eval(modelobj,val_x,val_y):
    scores = {
        'Accuracy': {'func': accuracy_score, 'list':[],'avg':0},
        'Balanced Accuracy': {'func': balanced_accuracy_score, 'list':[],'avg':0},
        'F1_score': {'func': f1_score, 'list':[],'avg':0},
        'Precision Score': {'func': precision_score, 'list':[],'avg':0},
        'Recall Score': {'func': recall_score, 'list':[],'avg':0},
        'AUC': {'func': roc_auc_score, 'list': [],'avg':0},
        'G-mean': {'func': geometric_mean_score, 'list': [],'avg':0}
    }
    
   
    pred, pred_proba = modelobj.forward(val_x, val_y)
    
#     for i in range(len(pred)):
#         if pred[i] == 0:
#             if pred_proba[i] < 0.5:
#                 pred[i]=1
    print(np.min(pred_proba), np.mean(pred_proba),np.max(pred_proba))
    print(classification_report(val_y, pred))

    for score_name, score_dict in scores.items():
        used_pred = pred
        if score_name == 'AUC':
            used_pred=pred_proba
        scorval=score_dict['func'](val_y, used_pred)
        score_dict['list'].append(scorval)
        print(score_name, scorval)
 
    for score_name, score_dict in scores.items():
        score_dict['avg'] = np.mean(score_dict['list'])
        score_dict['std'] = np.std(score_dict['list'])
        score_dict['fun'] = ""
    
 
    return scores


def __cross_random_split(modelobj,
                               data,
                               target):
     
    scores = {
        'Accuracy': {'func': accuracy_score, 'list':[],'avg':0},
        'Balanced Accuracy': {'func': balanced_accuracy_score, 'list':[],'avg':0},
        'F1_score': {'func': f1_score, 'list':[],'avg':0},
        'Precision Score': {'func': precision_score, 'list':[],'avg':0},
        'Recall Score': {'func': recall_score, 'list':[],'avg':0},
        'AUC': {'func': roc_auc_score, 'list': [],'avg':0},
        'G-mean': {'func': geometric_mean_score, 'list': [],'avg':0}
    }
    
    xa,xb,ya,yb = train_test_split(data,target,test_size=0.3)
    for i in range(1):
        if i == 0:
            trn_x, trn_y = xa,ya
            val_x, val_y = xb,yb
        else:
            trn_x, trn_y = xb,yb
            val_x, val_y = xa,ya
        print(xa.shape, trn_x.shape)
        pred, pred_proba = modelobj.predict(trn_x, trn_y, val_x, val_y)
 
        for score_name, score_dict in scores.items():
            used_pred = pred
            if score_name == 'AUC':
                used_pred=pred_proba
            scorval=score_dict['func'](val_y, used_pred)
            score_dict['list'].append(scorval)
            print(score_name, scorval)
 
    for score_name, score_dict in scores.items():
        score_dict['avg'] = np.mean(score_dict['list'])
        score_dict['std'] = np.std(score_dict['list'])
        score_dict['fun'] = ""
    
    
    return scores



def __cross_validation_5_2_scores_v2(modelobj,
                               data,
                               target,
                               seeds=[8, 247, 68, 333421, 52]):
    progresbar=tqdm(range(5*2))
    scores = {
        'Accuracy': {'func': accuracy_score, 'list':[],'avg':0},
        'Balanced Accuracy': {'func': balanced_accuracy_score, 'list':[],'avg':0},
        'F1_score': {'func': f1_score, 'list':[],'avg':0},
        'Precision Score': {'func': precision_score, 'list':[],'avg':0},
        'Recall Score': {'func': recall_score, 'list':[],'avg':0},
        'AUC': {'func': roc_auc_score, 'list': [],'avg':0},
        'G-mean': {'func': geometric_mean_score, 'list': [],'avg':0}
    }
    for i_s, seed in enumerate(seeds):
        progresbar.refresh()
        xa,xb,ya,yb = train_test_split(data,target,test_size=0.5, random_state=seed)
        for i in range(2):
            if i == 0:
                trn_x, trn_y = xa,ya
                val_x, val_y = xb,yb
            else:
                trn_x, trn_y = xb,yb
                val_x, val_y = xa,ya
            print(xa.shape, trn_x.shape)
            pred, pred_proba = modelobj.predict(trn_x, trn_y, val_x, val_y)
            progresbar.update();
            for score_name, score_dict in scores.items():
                used_pred = pred
                if score_name == 'AUC':
                    used_pred=pred_proba
                scorval=score_dict['func'](val_y, used_pred)
                score_dict['list'].append(scorval)
                print(score_name, scorval)

 
    for score_name, score_dict in scores.items():
        score_dict['avg'] = np.mean(score_dict['list'])
        score_dict['std'] = np.std(score_dict['list'])
        score_dict['fun'] = ""

    return scores



def __cross_validation_5_2_scores(modelobj,
                               data,
                               target,
                               seeds=[8, 247, 68, 333421, 52]):
    progresbar=tqdm(range(5*2))
    scores = {
        'Accuracy': {'func': accuracy_score, 'list':[],'avg':0},
        'Balanced Accuracy': {'func': balanced_accuracy_score, 'list':[],'avg':0},
        'F1_score': {'func': f1_score, 'list':[],'avg':0},
        'Precision Score': {'func': precision_score, 'list':[],'avg':0},
        'Recall Score': {'func': recall_score, 'list':[],'avg':0},
        'AUC': {'func': roc_auc_score, 'list': [],'avg':0},
        'G-mean': {'func': geometric_mean_score, 'list': [],'avg':0}
    } 
    for i_s, seed in enumerate(seeds):
        progresbar.refresh()
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        scores_differences = np.zeros(2)
        for i_f, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
            # print(f"Iteration: {i_s+1}, Fold:{i_f+1} for the model {model_name} and dataset {dataset_name}")
            # Split the data
            trn_x, trn_y = data[trn_idx], target[trn_idx]
            val_x, val_y = data[val_idx], target[val_idx]
            pred, pred_proba = modelobj.predict(trn_x, trn_y, val_x, val_y)
            progresbar.update();
            for score_name, score_dict in scores.items():
              used_pred = pred
              if score_name == 'AUC':
                used_pred=pred_proba
              score_dict['list'].append(score_dict['func'](val_y, used_pred))
       
 
    for score_name, score_dict in scores.items():
        score_dict['avg'] = np.mean(score_dict['list'])
        score_dict['std'] = np.std(score_dict['list'])
        score_dict['fun'] = ""

    return scores
