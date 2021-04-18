import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR 







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

X_reduced_array = X_reduced.to_numpy()
#print(data_reduced_array)






# Models

#Linear regression (OLS)
def ols(X, y):
	reg = LinearRegression()
	reg.fit(X, y)
	print(reg.coef_)
	print(reg.intercept_)


def ridge(X, y):
	reg = Ridge(alpha=.5)
	reg.fit(X, y)
	print(reg.coef_)
	print(reg.intercept_)

def ridgecv(X, y):
	reg = RidgeCV(alphas=np.logspace(-6, 6, 13))
	reg.fit(X, y)
	print(reg.coef_)
	print(reg.intercept_)

def lasso(X, y):
	reg = Lasso(alpha=0.1)
	reg.fit(X, y)
	print(reg.coef_)
	print(reg.intercept_)


def lassocv(X, y):
	reg = LassoCV(cv=5, random_state=0)
	reg.fit(X, y)
	print(reg.coef_)
	print(reg.intercept_)


"""
ols(X_reduced, y)
ridge(X_reduced, y)
ridgecv(X_reduced, y)
lasso(X_reduced, y)
lassocv(X_reduced, y)
"""
