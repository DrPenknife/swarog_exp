import crudeparser as parser
import sys

PATH_PICKLES = './pickles'



if(len(sys.argv)<2):
    print("error, the correct syntax is:")
    print(f"{sys.argv[0]} ./plik.csv")
    sys.exit(1)


total=0

labels = []
texts = []

with open(f'{sys.argv[1]}', 'r') as fcsv:
    csvrows = fcsv.readlines()
    for index,csvrow in enumerate(csvrows):
        if len(csvrow)==0:
            continue;
        _zip,_path,_lab = csvrow.rstrip().split(",")
        PATH = _zip.split(".")[0]
        print("reading file",f'{PATH}/{_path}')
        with open(f'{PATH}/{_path}', 'r') as f:
            body = str(f.readlines()[0])
            _txt = parser.get_text_from_html(body)
            texts.append(_txt)
            labels.append(1 if _lab=="TAK" else 0)

print("preparing ML model...")
import model

print("""
#
# TEST STARTED
#
""")


total = 0
errors = 0
for index,text in enumerate(texts):
    pred = model.predict_label(text)
    passed = pred == labels[index]
    if not passed:
        errors = errors + 1
    print(f"File-{total}", "CORRECT" if passed else "ERROR")
    total = total + 1


print("Stats:")
print("-----------------------")
print("number of tests:",total)
print("         errors:",errors)

