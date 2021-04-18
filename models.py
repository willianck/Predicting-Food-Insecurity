import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC 

from sklearn import tree

from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#Obtaining data

data  = pd.read_pickle('./imputed_data.pkl')
#data.to_csv("imputed_data.csv")

y = data["Food_InsecurityLevel"]

all_columns = list(data.columns)

columns_of_interest = ['HHsizemembers', 'HHsizeMAE', 'LandOwned']

col_indeces = []

for col in columns_of_interest:
	col_indeces.append(all_columns.index(col))


X_reduced = data.iloc[:, col_indeces]
#print(data_reduced)

#print(data_reduced_array)


print(X_reduced["LandOwned"].min())


# Support functions

def scale_data(X):
	scaler = MinMaxScaler()
	for col in X.columns:
		X[col] = scaler.fit_transform(X[col].to_numpy().reshape(-1,1))[:,0]	
	return X

def split_data(X,y):
	X_array = X.to_numpy()
	y_array = y.to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.3) 
	return X_train, X_test, y_train, y_test




X_reduced = scale_data(X_reduced)
X_train, X_test, y_train, y_test = split_data(X_reduced, y)



#Metrics

# Models
# Logistic regression
def ridge(X_train, X_test, y_train, y_test):
	reg = LogisticRegression(penalty='l1', solver='saga') 
	reg.fit(X_train, y_train)
	accuracy = reg.score(X_test, y_test)
	print(accuracy)

def ridgecv(X_train, X_test, y_train, y_test):
	reg = LogisticRegressionCV(cv=5, random_state=0, penalty='l1', solver='saga')
	reg.fit(X_train, y_train)
	accuracy = reg.score(X_test, y_test)
	print(accuracy)

def lasso(X_train, X_test, y_train, y_test):
	reg = LogisticRegression(penalty='l2')
	reg.fit(X_train, y_train)
	accuracy = reg.score(X_test, y_test)
	print(accuracy)

def lassocv(X_train, X_test, y_train, y_test):
	reg = LogisticRegressionCV(cv=5)
	reg.fit(X_train, y_train)
	accuracy = reg.score(X_test, y_test)
	print(accuracy)



# SVMs
def svc(X_train, X_test, y_train, y_test):
	clf = SVC()
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(accuracy)



def linear_svc(X_train, X_test, y_train, y_test):
	clf = LinearSVC()
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(accuracy)




def nu_svc(X_train, X_test, y_train, y_test):
	clf = NuSVC()
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(accuracy)







# KNN
def knn(X_train, X_test, y_train, y_test):
	clf = KNeighborsClassifier(n_neighbors=9)
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(accuracy)




# Trees
# Decision Tree 
def dct(X_train, X_test, y_train, y_test):
	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	clf.predict(X_test, y_test)
	#tree.plot_tree(clf) 


# Random Forest




# Neural Network
def nn(X_train, X_test, y_train, y_test):
	clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=1)
	clf.fit(X_train, y_train)
	accuracy =  clf.score(X_test, y_test)
	print(accuracy)

# Model calling

"""
ridge(X_train, X_test, y_train, y_test)
ridgecv(X_train, X_test, y_train, y_test)
lasso(X_train, X_test, y_train, y_test)
lassocv(X_train, X_test, y_train, y_test)

knn(X_train, X_test, y_train, y_test)

svc(X_train, X_test, y_train, y_test)
linear_svc(X_train, X_test, y_train, y_test)
nu_svc(X_train, X_test, y_train, y_test)

dct(X_train, X_test, y_train, y_test)

nn(X_train, X_test, y_train, y_test)
"""
