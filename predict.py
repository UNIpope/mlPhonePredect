from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.metrics import log_loss

head = []
with open("cl_feat.txt") as f:
    for line in f:
        head.append(line.strip())

#read in data set
df_train = pd.read_csv("cl_data.csv", na_values="?")
df_train.columns = head


df_pred = pd.read_csv("cl_queries.csv")
df_pred.columns = head

x = df_train.drop("target", axis=1)
y = df_train[['target']]

p = df_pred.drop("target", axis=1)

print(x)
print(x.shape)
print(y.shape)

# splits data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


# https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train.values.ravel())

#Predict Output of test set 
predictions = model.predict(y_test)
print("Predicted Value:", predictions)
print((metrics.accuracy_score(y_test, predictions))*100)

#Predict Output of actual queries 
queries = model.predict(p)
print("Predicted Value:", queries)

#output file for queries
ls = []
for i in queries:
    if i == 0:
        ls.append("no")
    else:
        ls.append("yes")

with open("predictions","w") as out:
    i = 1
    for row in ls:
        out.write("TEST{},{}".format(i,row))
        out.write("\n")
        i += 1

