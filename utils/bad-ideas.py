class Ensemble:
    name = "Ensemble"
    def __init__(self, class_weight={0:1, 1:1}, trees=10, ensemble_size=50):
        self.trees = trees
        self.class_weight = class_weight
        self.feature_map = []
        self.ensemble_size=ensemble_size
        for i in range(ensemble_size):
            self.feature_map.append({'wid':[],'widv':[]})
            self.feature_map[i]['wid'] = np.random.randint(256, size=50)
            self.feature_map[i]['widv'] = np.random.randint(300, size=20)
        
    def predict(self, trn_x, trn_y, val_x, val_y):
        print("shape",trn_x.shape)
        
        resp = []
        for clsid in range(self.ensemble_size):
            wid_samples = self.feature_map[clsid]['wid']
            widv_samples = self.feature_map[clsid]['widv']
            new_trn_x = [[doc[wid][widv] for wid in wid_samples for widv in widv_samples] for doc in trn_x]
            new_val_x = [[doc[wid][widv] for wid in wid_samples for widv in widv_samples] for doc in val_x]
            rf = RandomForestClassifier(n_estimators = self.trees,class_weight= self.class_weight)
            rf.fit(new_trn_x, trn_y)
           
            pred_proba = rf.predict_proba(new_val_x)
            pred = rf.predict(new_val_x)
            p = np.array([pred_proba[i][cl] for i,cl in enumerate(pred)])
            resp.append(pred_proba)
            
        pred_proba = np.mean(np.array(resp), axis=0)
        pred = np.argmax(pred_proba,axis=1)
        p = np.array([pred_proba[i][cl] for i,cl in enumerate(pred)])
        
        print(classification_report(val_y, pred))
        print(confusion_matrix(val_y, pred))
        return pred,p