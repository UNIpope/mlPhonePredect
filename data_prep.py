import pandas as pd
from pprint import pprint
import collections
from sklearn.preprocessing import LabelEncoder

"""
from sklearn.preprocessing import OneHotEncoder
oe = OneHotEncoder()
df[cat_cols] = df[cat_cols].apply(lambda col: le.fit_transform(col))

print(df[cat_cols])
"""

#read in hedders
head = []
with open("feature_names.txt") as f:
    for line in f:
        head.append(line.strip())

#read in training data set
df = pd.read_csv("trainingset.csv", na_values="unknown ")
df.columns = head

#read in queries 
dfq = pd.read_csv("queries.csv", na_values="unknown ")
dfq.columns = head

#drop cols
drop_list = ["poutcome", "id", "contact"]
df.drop(drop_list, inplace=True, axis=1)
dfq.drop(drop_list, inplace=True, axis=1)

#encode cat vals
le = LabelEncoder()
cat_mask = df.dtypes == object
cat_cols = df.columns[cat_mask].tolist()
df[cat_cols] = df[cat_cols].apply(lambda col: le.fit_transform(col))

cat_maskq = dfq.dtypes == object
cat_colsq = dfq.columns[cat_maskq].tolist()
dfq[cat_colsq] = dfq[cat_colsq].apply(lambda col: le.fit_transform(col))

#imputate missing feature values below 30%
imp_list = ["job","education"]
for feat in imp_list:
    df[feat] = df[feat].fillna(method='pad')
    print(df[feat])

for featq in imp_list:
    dfq[feat] = dfq[feat].fillna(method='pad')
    print(dfq[feat])

#output csv
df.to_csv("cl_data.csv", header=True) 
dfq.to_csv("cl_queries.csv", header=True)