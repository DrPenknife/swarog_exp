import sqlite3
from urllib.request import urlopen
from bs4 import BeautifulSoup


#
# https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
#

def get_text_from_html(html):
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text



def create_html_files():
    PATH_PICKLES = './pickles'

    conn = sqlite3.connect(f'{PATH_PICKLES}/swarog.sqlite')
    c = conn.cursor()

    data=[]
    c.execute("select label, body from rawsearch where dataset='isot' and label = 1 limit 25;")
    data.extend(c.fetchall())
    c.execute("select label, body from rawsearch where dataset='isot' and label = 0 limit 25;")
    data.extend(c.fetchall())

    labels, texts = [d[0] for d in data], [d[1] for d in data]


    conn.close()

    PATH = "50_html_swarog"
    total=0
    with open(f'{PATH}/plik.csv', 'w') as fcsv:
        for index,text in enumerate(texts):
            label = "TAK" if labels[index] else "NIE"
            fcsv.write(f'{PATH}.zip,isot/file-{index}.html,{label}\n') 

            with open(f'{PATH}/isot/file-{total}.html', 'w') as f:
                f.write(f'<html><body>{text}</body></html>') 
            total = total + 1    



#if __name__ == "__main__":
#    create_html_files()