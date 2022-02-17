import pandas as pd

class ResultsAnalysis:
    def collect_errors(self, model, x, y, txt):
        print("model name is:",model.name)
        pred, prob = model.forward(x)
        
        ctr = 0
    
        vvv=[[],[],[]]
    
        for i in range(len(x)):
            if pred[i] != y[i]:
                ctr = ctr+1
                vvv[0].append(txt[i])
                vvv[1].append(y[i])
                vvv[2].append(pred[i])
                #print(txt[i],"\n----------------------\n")
        
        df = pd.DataFrame({'text':vvv[0],'lab':vvv[1],'pred':vvv[2]})
        print(ctr)
        print("done")
        return df
    