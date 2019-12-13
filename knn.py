import csv

from sklearn import preprocessing
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics



head = []
with open("feature_names.txt") as f:
    for line in f:
        head.append(line.strip())

#read in data set
df_train = pd.read_csv("trainingset.csv", na_values="?")
df_train.columns = head


df_test = pd.read_csv("queries.csv")
df_test.columns = head
# x = df_train[['job', 'marital', 'education', 'housing', 'loan', 'age', 'campaign', 'previous']]

x = df_train[['age', 'campaign', 'previous']]

y = df_train[['target']]

y_array = np.array(y)

print(x)

print(x.shape)
print(y.shape)

# splits data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(X_train, y_train.values.ravel())
# https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
predictions = model.predict(X_test)

print(predictions)
print((metrics.accuracy_score(y_test, predictions))*100)

# exports as csv
# df = pd.DataFrame(data=predictions)
# df.to_csv("./knnpred.csv", sep=',', index=False)


# another good ewbsite for knn
# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn