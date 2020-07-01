import pandas as pd
import numpy as np

#importing dataset
dataset = pd.read_csv(r'Final Folder\Dataset\breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

#Dataset Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#Hyperparameter Tuning and Training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

#listing parameters to tune
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#creating dictionary of parameters
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Creating model
knn = KNeighborsClassifier()
#Using GridSearch to tune parameters
clf = GridSearchCV(knn, hyperparameters, cv=10)

#Train model
best_model = clf.fit(X_train, y_train)

print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

#Predict test data set.
y_pred = clf.predict(X_test)
#Checking performance our model with classification report.
print(classification_report(y_test, y_pred))
#Checking performance our model with ROC Score.
print(roc_auc_score(y_test, y_pred, average='weighted'))