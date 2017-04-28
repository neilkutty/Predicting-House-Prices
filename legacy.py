#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:19:48 2017

@author: NNK
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn import linear_model
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import os


np.random.seed(10)
 
os.chdir("/Users/NNK/Documents/project/kaggle/houseprices")

static = pd.read_csv('train.csv')
htrain = pd.read_csv('train.csv')
htest = pd.read_csv('test.csv')

htrain = htrain.drop('Id', axis=1)
htrain = htrain[htrain.GrLivArea < 4000]

# ###
##   Note: change "YearBuilt", "YearSold", "YearRM"
#       to "Years Since" vars..
##   Drop original year variables
###  Create a TotalSF variable
htrain['YrsOld'] = 2017 - htrain['YearBuilt']
htrain['YrsRM'] = 2017 - htrain['YearRemodAdd']
htrain['YrsSS'] = 2017 - htrain['YrSold']

htrain = htrain.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1)

htrain['TotalSF'] = htrain.GrLivArea + htrain.TotalBsmtSF


# ------------------- DATA CLEANING ------------------------- #

#...............................................................


htrain = htrain.fillna(value=0)
#sd_cols = pd.get_dummies(htrain.MSZoning, prefix='MSZoning')
#bt_cols = pd.get_dummies(htrain.BldgType, prefix='BldgType')
#style_cols = pd.get_dummies(htrain.HouseStyle, prefix='HouseStyle')
#ms_cols = pd.get_dummies(htrain.Exterior1st, prefix='Exterior1st')
#nb_cols = pd.get_dummies(htrain.GarageType, prefix='GarageType')
#htrain = pd.concat([htrain,sd_cols],axis=1)

#htrain = pd.concat([htrain,sd_cols,bt_cols,style_cols,ms_cols,nb_cols], axis=1)
#%%
# Convert dataframe to all numeric for training models
#
#-------------------------

#1.) Seperate numerical and non-numerical columns into dataframes
numht = htrain.select_dtypes(include = ['float64','int64'])
nonht = htrain.select_dtypes(exclude = ['float64','int64'])
# -- Get Category Codes --
#2.) Create empty df and pass all columns from non-numeric df 
# converted to categorical
non2 = pd.DataFrame()
for column in nonht:
    non2[column] = pd.Categorical(nonht[column])
#3.) Create another empty df and pass all columns from categorical df
# converted to codes    
nonC = pd.DataFrame()
for column in non2:
    nonC[column] = non2[column].cat.codes
#Combine the native numerical and newly converted dataframes
trNum = pd.concat([numht,nonC], axis=1)
#Create normalized dataframe if needed
trNorm = (trNum - trNum.mean()) / (trNum.max() - trNum.min())

train, test = train_test_split(trNum, test_size = .30, random_state = 1010)
# Train outcome and predictors 
y = train.SalePrice
X = train.drop('SalePrice', axis=1)
# Test outcome and predictors
yt = test.SalePrice
Xt = test.drop('SalePrice', axis=1)

# Create normalized train and test sets
train, test = train_test_split(trNorm, test_size = .30, random_state = 1010)
ynorm = train.SalePrice
Xnorm = train.drop('SalePrice', axis=1)
#Normalized test set
ytnorm = test.SalePrice
Xtnorm = test.drop('SalePrice', axis=1)

#%% 
#<><><><><><><><><>---------------------------------------------------<><><><><><><><><><><>
#<><><><><><><><><>---------------------------------------------------<><><><><><><><><><><>    
#<><><><><><><><><>---------------------------------------------------<><><><><><><><><><><>    
#
#%%

# -- Re-address NaN values

trNorm = trNorm.fillna(0)