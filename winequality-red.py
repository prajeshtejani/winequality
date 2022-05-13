import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  

data = pd.read_csv('winequality-red.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values



'''#for model Selection
seed = 6
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=20, random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
#From the result i can easily say RandomForestClassifier is the best suit'''
 
#Traintest split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

classifier = RandomForestClassifier(criterion='entropy',n_estimators = 100)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 6)

#confusion_metrics
y_test = (y_test > 6)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
    

'''#This one id for best Parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[10,20,30,40,50,60,70,80,90,100],'criterion':['gini'],'criterion':['entropy']}]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs= -1,
                           cv = 10)
grid_search.fit(x_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best_Accuracy = {:.2f} %".format(best_accuracy * 100))
print("Best_Param=",best_parameters)'''

   




















