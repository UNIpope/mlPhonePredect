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

#read in data set
df = pd.read_csv("trainingset.csv", na_values="unknown ")
df.columns = head

#drop cols
drop_list = ["poutcome", "id"]
df.drop(drop_list, inplace=True, axis=1)

#encode cat vals
le = LabelEncoder()
cat_mask = df.dtypes == object
cat_cols = df.columns[cat_mask].tolist()
df[cat_cols] = df[cat_cols].apply(lambda col: le.fit_transform(col))

#imputate missing feature values below 30%
imp_list = ["job","contact"]
df["contact"] = df["contact"].fillna(method='pad')
print(df[imp_list])

#output csv
df.to_csv("cleaned.csv", header=True) 
