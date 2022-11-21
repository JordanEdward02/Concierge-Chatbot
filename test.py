import pandas as pd
qaa = pd.read_excel("./dataset/QAA/QAA.xlsx")
for item in qaa["Questions"].tolist():
    print(item)
# we can get the answer using the index as the second argument of the matrix thingy
print(qaa["Answers"][0])