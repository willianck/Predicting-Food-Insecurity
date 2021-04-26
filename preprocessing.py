import re
import string
import string
from pathlib import Path
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


"""
from sklearn.model_selection import cross_validate as cross_validation, ShuffleSplit, cross_val_score, train_test_split, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, auc
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
"""




#data  = pd.read_pickle('./preprocessed_data.pkl')

# Data inspection ---------------------------------------------------------------------------
def inspect_data(data):
	print(data.head())
	print(data.count())
	print(data.shape)
	print(data.info())
	for column in data:
	    print(data[column].describe())

#inspect_data(data)

# Data Wrangling --------------------------------------------------------------------------------


# replace negative values for features that are bounded to be positive only  as distance metrics like Land cultivated measured in hectares or Income and PPP earned 

def replace_negative(data,columns):
    for col in columns:
        data.loc[data[col] < 0] = 0
    


def drop_columns(data):
	negative_col = ['LandCultivated', 'LandOwned', 'currency_conversion_factor','total_income_USD_PPP_pHH_Yr','offfarm_income_USD_PPP_pHH_Yr','value_livestock_prod_consumed_USD_PPP_pHH_Yr','NrofMonthsWildFoodCons']

	categorical_col = ['Country','HouseholdType','Head_EducationLevel', 'WorstFoodSecMonth' ,'BestFoodSecMonth','HFIAS_status']
	# Head_EducationLevel specification about  possible values was not given so we omit this for now 


	replace_negative(data, negative_col)

	data_model = data.copy()
	data_model = data.drop(['ID_PROJ','ID_COUNTRY','SURVEY_ID','Region'],axis=1)
	data_model.drop(['Head_EducationLevel'],axis=1)
	data_model.set_index('RHoMIS_ID')
	return data_model



"""
# Print objects
for col in data:
    if data[col].dtype == object:
        print(col)
"""


def replace_missing_with_nan(data_model):
	# replace  HFIAS status with 0 with missing value 
	data_model.loc[data_model['HFIAS_status'] == 0] = np.NaN


	#replace WorstFoodSecMonth and BestFoodSecMonth with No_answer or none with  missing value 
	data_model.loc[data_model['WorstFoodSecMonth'] == 'No_answer'] = np.NaN
	data_model.loc[data_model['WorstFoodSecMonth'] == 'None'] = np.NaN
	data_model.loc[data_model['WorstFoodSecMonth'] == 'no_answer'] = np.NaN
	data_model.loc[data_model['BestFoodSecMonth'] == 'No_answer'] = np.NaN
	data_model.loc[data_model['BestFoodSecMonth'] == 'no_answer'] = np.NaN
	data_model.loc[data_model['BestFoodSecMonth'] == 'None'] = np.NaN

	#replace HouseHold type with no answer to missing value 
	data_model.loc[data_model['HouseholdType'] == 'no_answer'] = np.NaN
	data_model.loc[data_model['HouseholdType'] == '0'] = np.NaN

	return data_model



# Dictionary for months in different languange to english


def process_months(var, translations):
    if var in translations:
            return  translations.get(var)
    else: return var    
                


def translate_words(data_model):
	months_to_eng = {'ukuboza': 'dec', 'gashyantare' : 'feb', 'kamena' : 'jun', 'mutarama': 'jan', 'nyakanga' : 'jul' , 'nzeri' : 'sep', 'ukwakira' : 'oct',
                 'gicurasi' : 'may' , 'werurwe' : 'mar', 'kanama' : 'aug', 'ugushyingo' : 'nov' , 'mata' : 'apr'}

	translate = lambda x : process_months(x, months_to_eng)
	data_model['BestFoodSecMonth'] = data_model.BestFoodSecMonth.apply(translate)
	data_model['WorstFoodSecMonth'] = data_model.WorstFoodSecMonth.apply(translate)
	return data_model

def encode_categoricals(data_model):
	enc = OneHotEncoder(handle_unknown='ignore')
	enc_data = enc.fit_transform(data_model)
	return enc_data
		

	
#enc_data = pd.get_dummies(data_model, prefix=['Nat','Type'], columns=['Country','HouseholdType'])

def hfias_status_vis(data_model):
	HFIAS_status_count = data_model['HFIAS_status'].value_counts()
	sns.set(style="darkgrid")
	sns.barplot(x = HFIAS_status_count.index, y = HFIAS_status_count.values, alpha=0.9)
	plt.title('Frequency Distribution of HFIAS_status')
	plt.ylabel('Number of Occurrences', fontsize=12)
	plt.xlabel('HFIAS_status', fontsize=12)
	plt.show()


"""
# encode ordinal data 
HFIAS_status = {'SeverelyFI':0,'ModeratelyFI':1,'MildlyFI':2,'FoodSecure':3 }           
data_model['HFIAS_status'] = data_model.HFIAS_status.apply(process_status)

data_model['HFIAS_status'].value_counts()
"""
#print(data_model['Country'].value_counts())



def preprocess_data():
	data = pd.read_csv('../RHoMIS_ADS_Project_2021/Data/RHoMIS_Indicators.csv',encoding='latin1')
	data_model = drop_columns(data)
	data_model = replace_missing_with_nan(data_model)
	data_model = translate_words(data_model)
	enc_data = encode_categoricals(data_model)
	#enc_data.to_pickle('preprocessed_data.pkl')
	print(enc_data)

#preprocess_data()

data  = pd.read_pickle('./preprocessed_data.pkl')

#print(enc_data)
# Visualizations
#hfias_status_vis(data_model)

# Imputation ---------------------------------------------------------------------------------

# Graph displaying amount of missing data for each featurei

def missing_data_vis():
	missing_data = pd.DataFrame(data[data.columns[data.isnull().any()]].isnull().sum()/len(data)*100)

	names = []
	for i in range(len(missing_data)):
		names.append(missing_data.iloc[i].name)
	values = []
	for i in range(len(missing_data)):
		values.append(missing_data.iloc[i][0])

	data_1 = {'Features': names,'Missing Data Percentage': values}

	# Dictionary loaded into a DataFrame       
	df = pd.DataFrame(data=data_1)
	df.plot.bar(x="Features", y="Missing Data Percentage", title="Features with Missing Data",figsize=(10,6))
	plt.show(block=True)


#missing_data_vis()




# K means imputation

def knn_imputer(df, column_name):
	imputer = KNNImputer()
	df[column_name] = imputer.fit_transform(df[column_name].to_numpy().reshape(-1,1))
	return df

# Takes a while to run, there are 94 features
def knn_imputer_all_columns(df):
	for col in df.columns:
		print(df[col].dtype)
		if df[col].dtype != object:
			imputer = KNNImputer()
			df[col] = imputer.fit_transform(df[col].to_numpy().reshape(-1,1))
	return df


# Example for running knn_imputer
"""
col_name = 'GPS_ALT'
df = knn_imputer(enc_data, col_name)
print(df['GPS_ALT'])
"""


#df = knn_imputer_all_columns(enc_data)

"""
# Check if there is any NaN in column
for col in df.columns:
	for item in df[col]:
		if math.isnan(item) == True:
			print(item)

"""
