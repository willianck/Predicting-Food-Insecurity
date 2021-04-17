import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR 



#Obtaining data

data  = pd.read_pickle('./imputed_data.pkl')
#data.to_csv("imputed_data.csv")

X = data.drop(['Food_InsecurityLevel'],axis=1)
y = data["Food_InsecurityLevel"]





#Feature selection helper functions

#Returns a dataframe including the feature names and the scores from the feature selection method output
def get_top_features_and_scores(fit, X):
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X.columns)
	featureScores = pd.concat([dfcolumns,dfscores],axis=1) #concat two dataframes for better visualization 
	featureScores.columns = ['Feature','Score']  #naming the dataframe columns
	return featureScores.nlargest(10,'Score')  #print 10 best features


def indeces_to_column(df, cols):
	top_features_names = df.iloc[:,cols].columns
	return list(top_features_names)






# Feature selection methods

def univariate_selection(X,y,k, score_func):
	bestfeatures = SelectKBest(score_func=score_func, k=k)
	#bestfeatures = SelectKBest(score_func=f_classif, k=k)
	fit = bestfeatures.fit(X,y)

	scores = get_top_features_and_scores(fit, X)
	print(scores)

	cols = bestfeatures.get_support(indices=True)
	return cols

def trees(X,y):
	clf = ExtraTreesClassifier()
	clf = clf.fit(X, y)
	return clf.feature_importances_
	#X_new = model.transform(X)


def trees_vis(X,y,k):
	model = ExtraTreesClassifier()
	model = model.fit(X,y)
	print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
	#plot graph of feature importances for better visualization
	model = SelectFromModel(model, prefit=True)
	X_new = model.transform(X)

	# Other way of selecting most important features
	feat_importances = pd.Series(model.feature_importances_, index=X.columns)
	feat_importances.nlargest(k).plot(kind='barh')
	plt.show()


# Need to pick a subset of features as a 95x95 matrix is too much
def correlation_matrix(data):
	corrmat = data.corr()
	top_corr_features = corrmat.index
	plt.figure(figsize=(20,20))
	#plot heat map
	sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
	plt.show()



def forward_sfs(X,y):
	#Need an estimator so that it can determine which feature to drop (here it's KNN)
	knn = KNeighborsClassifier(n_neighbors=3)
	sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
	fit = sfs.fit(X,y)
	print(fit.support_)
	#X_transformed = sfs.fit_transform(X,y)


def backward_sfs(X,y):
	#Need an estimator so that it can determine which feature to drop (here it's KNN)
	knn = KNeighborsClassifier(n_neighbors=3)
	sfs = SequentialFeatureSelector(knn, n_features_to_select=3, direction='backward')
	fit = sfs.fit(X,y)
	print(fit.support_)
	#X_transformed = sfs.fit_transform(X,y)


def rfe(X,y):
	estimator = SVR(kernel="linear")
	selector = RFE(estimator, n_features_to_select=5, step=1)
	selector = selector.fit(X, y)
	print(selector.support_)
	print(selector.ranking_)
	data_reduced = selector.transform(X)


def rfecv(X,y):
	estimator = SVR(kernel="linear")
	selector = RFECV(estimator, step=1, min_features_to_select=5, cv=5)
	selector = RFE(estimator, n_features_to_select=5, step=1)
	selector = selector.fit(X, y)
	print(selector.support_)
	print(selector.ranking_)
	data_reduced = selector.transform(X)


def lasso(X, y):
	clf = Lasso(alpha=0.1)
	fit = clf.fit(X,y)
	#print(clf.coef_)
	#print(clf.intercept_)


# Running feature selection


"""
features_usfc = univariate_selection(X, y, 10, f_classif)
print(indeces_to_column(data, features_usfc))

features_usfr = univariate_selection(X,y,10, f_regression)
print(indeces_to_column(data, features_usfr))
"""
