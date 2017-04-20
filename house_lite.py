#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:30:27 2017

@author: NNK
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sb
import os

np.random.seed(10)
#%% 
os.chdir("/Users/NNK/Documents/project/kaggle/houseprices")

htrain = pd.read_csv('train.csv')
htest = pd.read_csv('test.csv')

htrain = htrain.drop('Id', axis=1)

###
##   Note: change "YearBuilt", "YearSold", "YearRM"
#       to "Years Since" vars..
##
###
htrain['YrsOld'] = 2017 - htrain['YearBuilt']
htrain['YrsRM'] = 2017 - htrain['YearRemodAdd']
htrain['YrsSS'] = 2017 - htrain['YrSold']

htrain = htrain.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1)

htrain['TotalSF'] = htrain.GrLivArea + htrain.TotalBsmtSF


# ------------------- DATA CLEANING ------------------------- #

#...............................................................
# Convert dataframe to all numeric for training models
#
#-------------------------
#htrain = htrain.dropna(axis=1)
# Fill NAs with 0 first step
htrain = htrain.fillna(value=0)
sd_cols = pd.get_dummies(htrain.SaleCondition, prefix='SaleCon')
bt_cols = pd.get_dummies(htrain.BldgType, prefix='BldgType')
style_cols = pd.get_dummies(htrain.HouseStyle, prefix='HouseStyle')
ms_cols = pd.get_dummies(htrain.Exterior1st, prefix='Exterior1st')
nb_cols = pd.get_dummies(htrain.GarageType, prefix='GarageType')
htrain = pd.concat([htrain,sd_cols, bt_cols, style_cols,
                    ms_cols, nb_cols], axis=1)

# Seperate numerical and non-numerical columns into dataframes
numht = htrain.select_dtypes(include = ['float64','int64'])
nonht = htrain.select_dtypes(exclude = ['float64','int64'])

# -- Get Category Codes --
#Create empty df and pass all columns from non-numeric df 
# converted to categorical
non2 = pd.DataFrame()
for column in nonht:
    non2[column] = pd.Categorical(nonht[column])

#Create another empty df and pass all columns from categorical df
# converted to codes    
nonC = pd.DataFrame()
for column in non2:
    nonC[column] = non2[column].cat.codes
    
#Combine the native numerical and newly converted dataframes
trNum = pd.concat([numht,nonC], axis=1)

#Create normalized dataframe with unadjusted SalePrice
temp = numht.drop('SalePrice',axis=1)
trnorm = (temp - temp.mean()) / (temp.max() - temp.min())
trNum_norm = pd.concat([trnorm,nonC,numht['SalePrice']], axis=1)

#Create normalized dataframe with normalized outcome var
normdf = (trNum - trNum.mean()) / (trNum.max() - trNum.min())



train, test = train_test_split(trNum, test_size = .30, random_state = 1010)

# Train outcome and predictors 
y = train.SalePrice
X = train.drop('SalePrice', axis=1)

# Test outcome and predictors
yt = test.SalePrice
Xt = test.drop('SalePrice', axis=1)

# Create normalized train and test sets

train, test = train_test_split(trNum_norm, test_size = .30, random_state = 1010)

ynorm = train.SalePrice
Xnorm = train.drop('SalePrice', axis=1)

ytnorm = test.SalePrice
Xtnorm = test.drop('SalePrice', axis=1)

#%%
#               ## ==== Model Training ==== ##
#
#               ## ==== Gradient Boosting Regressor ==== ##

#-----------------------------------------------------------------
#Set model parameters
gbfit = GradientBoostingRegressor(n_estimators=250, loss='ls', random_state=1010)

#Fit model
gbfit.fit(X=X, y=y)

#%% explore GB fit
accuracy = gbfit.score(Xt, yt)
predict = gbfit.predict(Xt)

#%% GB with Normalized variables
gbfit.fit(X=Xnorm, y=ynorm)
accuracy = gbfit.score(Xtnorm,ytnorm)
predict = gbfit.predict(Xtnorm)


#%%
# Show results of GBR with all variables

sb.set_style('darkgrid')
plt.rcParams['figure.figsize']=(10,8)
plt.scatter(predict, yt)
plt.suptitle('test title')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')

print('Gradient Boosting Accuracy %s' % '{0:.2%}'.format(accuracy))
#%%
# Model feature importances ranking
importances = gbfit.feature_importances_
indices = np.argsort(importances)[::-1]

print('Feature Importances')

for f in range(X.shape[1]):
    print("feature %s (%f)" % (list(X)[f], importances[indices[f]]))

#plot
feat_imp = pd.DataFrame({'Feature':list(X),
                         'Gini Importance':importances[indices]})

plt.rcParams['figure.figsize']=(8,12)
sb.set_style('whitegrid')
ax = sb.barplot(x='Gini Importance', y='Feature', data=feat_imp)
ax.set(xlabel='Gini Importance')
plt.show()    
#%%
#         --  Plotly --

plotly.tools.set_credentials_file(username='sampsonsimpson', 
                                  api_key='cxm2iF7KKBGZgXmDOU9S')

#%%
gbr_results = pd.DataFrame({'Predicted':predict,
                            'Ground Truth':yt})

    
gbr_results.iplot(kind='scatter',mode='markers',x='Predicted',y='Ground Truth',
                  title='GBR Prediction Results',
                  xTitle='Predicted', 
                  yTitle='Ground Truth',
                  filename='predicted-groundtruth')



    
