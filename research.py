# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dow_jones_index.csv')

dataset['high/low'] = dataset['next_weeks_open']-dataset['close']
dataset['high/low']= (dataset['high/low']>0)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
dataset.iloc[:,8:11]=imputer.fit_transform(dataset.iloc[:,8:11])

del dataset['date']
del dataset['stock']
del dataset['quarter']
del dataset['next_weeks_open']
del dataset['next_weeks_close']
del dataset['percent_change_next_weeks_price']

#newdata = pd.get_dummies(dataset,columns=['stock'],drop_first=True)

X = dataset.iloc[:,dataset.columns!='high/low'].values
y = dataset.iloc[:, dataset.columns=='high/low'].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifierforest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierforest.fit(X_train, y_train)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Best Classifier:
from sklearn.ensemble import AdaBoostClassifier
classifier= AdaBoostClassifier(n_estimators = 175)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = np.reshape(y_pred,(-1,1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[50,60,70,75,80,100,125,155,175,225,230,250,300,325,350,360,375,400]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_





