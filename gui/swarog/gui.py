from fastapi import FastAPI
import chevron
from fastapi.responses import HTMLResponse
from os import listdir
from os.path import isfile, join
import json 
import requests
from pydantic import BaseModel
import datetime
import model
import numpy as np

app = FastAPI()

##########################################################################
# STORAGE
#  

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def history_get():
    with open("history.json", 'r',  encoding="utf-8") as fhandle:
        db = json.load(fhandle)
    return db   

def history_add(db):
    with open("history.json", 'w',  encoding="utf-8") as fhandle:
        json.dump(db,fhandle, indent = 6, cls=NpEncoder)    

        
##########################################################################
#
#  

component_template = """
// {{{name}}}
Vue.component('{{{name}}}', { template: `{{{template}}}`, {{{script}}} } );
// end of  {{{name}}}
/////////////////////////////////////////////////////////////////////////////////////////
"""

class TxtDocument(BaseModel):
    title: str
    text: str


@app.get("/", response_class=HTMLResponse)
async def root():
    path = join("vue", "components")
    payload = ""
    for f in listdir(path):
        if not isfile(join(path, f)):
            continue 
            
        with open(join(path,f), 'r') as fhandle:
            _txt = fhandle.read()
            _x = _txt.replace("</script>","").split("<script>")
            _out = chevron.render(component_template, {'name': f.split(".html")[0],
                                                       'template':_x[0], 
                                                       'script': _x[1]})
            payload += _out + "\n\n"
    
    with open('./vue/main.html', 'r') as fhandle:
        output = chevron.render(fhandle, {'payload': payload})

    return output
        
##########################################################################
#
#
@app.get("/swarog/api/history")
async def history():
    return history_get()    
     
    

##########################################################################
#
#
@app.post("/swarog/api/analysis")
async def analysis(doc: TxtDocument):
    db = history_get() 
    
    pred = model.predict(doc.text)
    
    #print(pred)
    
    db.append({
          'cls': pred,
        'title': doc.title,
         'body': doc.text,
         'time': datetime.datetime.now().isoformat()
    })
    
    history_add(db)
    
    return doc
 
      
    