import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


#read in hedders
head = []
with open("feature_names.txt") as f:
    for line in f:
        head.append(line.strip())

#read in data set
df = pd.read_csv("trainingset.csv", na_values="unknown ")
df.columns = head

le = preprocessing.LabelEncoder()


y = df.target

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print(predictions)