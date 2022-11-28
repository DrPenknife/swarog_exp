import sqlite3
import model

PATH_PICKLES = './pickles'

print("""
#
# TEST STARTED
#
""")

conn = sqlite3.connect(f'{PATH_PICKLES}/swarog.sqlite')
c = conn.cursor()

data=[]
c.execute("select label, body from rawsearch where dataset='isot' and label = 1 limit 25;")
data.extend(c.fetchall())
c.execute("select label, body from rawsearch where dataset='isot' and label = 0 limit 25;")
data.extend(c.fetchall())

labels, texts = [d[0] for d in data], [d[1] for d in data]

total = 0
errors = 0
for index,text in enumerate(texts):
    pred = model.predict_label(text)
    passed = pred == labels[index]
    if not passed:
        errors = errors + 1
    print(f"File-{total}", "CORRECT" if passed else "ERROR")
    total = total + 1

conn.close()

print("Stats:")
print("number of tests:",total)
print("errors:",errors)

