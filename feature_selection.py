import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR 



data  = pd.read_pickle('./preprocessed_data.pkl')
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range#apply SelectKBest class to extract top 10 best features

#fit_transform() is usefule for obtaining the dataset with the selected features, fit() is better for determining which features were chosen


def univariate_selection(X,y,k):
	bestfeatures = SelectKBest(score_func=chi2, k=k).fit_transform(X,y)
	#bestfeatures = SelectKBest(score_func=chi2, k=10)
	fit = bestfeatures.fit(X,y)
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X.columns)
	#concat two dataframes for better visualization 
	featureScores = pd.concat([dfcolumns,dfscores],axis=1)
	featureScores.columns = ['Specs','Score']  #naming the dataframe columns
	print(featureScores.nlargest(10,'Score'))  #print 10 best features


def trees(X,y):
	clf = ExtraTreesClassifier()
	clf = clf.fit(X, y)
	clf.feature_importances_
	X_new = model.transform(X)


def feature_importance(X,y,k):
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


def correlation_matrix(data):
	corrmat = data.corr()
	top_corr_features = corrmat.index
	plt.figure(figsize=(20,20))
	#plot heat map
	g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")



def forward_sfs(X,y):
	#Need an estimator so that it can determine which feature to drop (here it's KNN)
	#knn = KNeighborsClassifier(n_neighbors=3)
	sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
	X_transformed = sfs.fit_transform(X,y)



def backward_sfs(X,y):
	#Need an estimator so that it can determine which feature to drop (here it's KNN)
	#knn = KNeighborsClassifier(n_neighbors=3)
	sfs = SequentialFeatureSelector(knn, n_features_to_select=3, direction='backward')
	X_transformed = sfs.fit_transform(X,y)


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
	clf = linear_model.Lasso(alpha=0.1)
	clf.fit(X,y)
	print(clf.coef_)
	print(clf.intercept_)


