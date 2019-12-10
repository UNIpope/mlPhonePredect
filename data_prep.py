import pandas as pd
from pprint import pprint
import collections


#read in hedders
head = []
with open("feature_names.txt") as f:
    for line in f:
        head.append(line.strip())

#read in data set
df = pd.read_csv("trainingset.csv", na_values="unknown ")
df.columns = head