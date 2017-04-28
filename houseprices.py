#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:11:49 2017

# Implement unsupervised learning method!

@author: NNK
"""

#%%
import datetime
now = datetime.datetime.now()
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

#%%  Functionalize cleaning
def convert_data(data):
    data = data.fillna(0)
    data = data.drop('Id',axis=1)
    data['YrsOld'] = now.year - data['YearBuilt']
    data['YrsRM'] = now.year - data['YearRemodAdd']
    data['YrsSS'] = now.year - data['YrSold']
    data = data.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1)
    num = data.select_dtypes(include = ['float64','int64'])
    non = data.select_dtypes(exclude = ['float64','int64'])
    j = pd.DataFrame()
    k = pd.DataFrame()
    for column in non:
        j[column] = pd.Categorical(non[column])
        k[column] = j[column].cat.codes
    converted_data = pd.concat([num,k], axis=1)
    converted_data = converted_data.fillna(0)
    return converted_data
#%%
trNum = convert_data(htrain)
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
#...................................................................................
#
#                                <Spectral Clustering >
#...................................................................................

import sklearn.cluster as cluster
import time
sb.set_context('poster')
sb.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sb.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)


#%%
plot_clusters(trNorm.drop('SalePrice',axis=1).as_matrix(), cluster.KMeans, (), {'n_clusters':6})
#...................................................................................
#...................................................................................


#%%


#<><><><><><><><><>---------------------------------------------------<><><><><><><><><><><>  


#%%  
#|---||---||---||---||---||---||---||---||---||---||---||---||---||---||---|
#
#|---||---||---||---|    Functionalize Model Training   |---||---||---||---|
#
#|---||---||---||---||---||---||---||---||---||---||---||---||---||---||---|

def fit_model(model, X, y, Xtest, ytest):
    fit = model.fit(X, y)
    accuracy = fit.score(Xtest, ytest)
    predict = fit.predict(Xtest)
    return fit, accuracy, predict

#%%

models = [GradientBoostingRegressor(n_estimators=250,max_depth=2,loss='ls',random_state=1010),
          RandomForestRegressor(n_estimators=250)]

#%%
sup = fit_model(models[1], X,y,Xt,yt)
#%%
#<><><><><><><><><>---------------------------------------------------<><><><><><><><><><><>    
#<><><><><><><><><>---------------------------------------------------<><><><><><><><><><><>    

#%%
#               ## ==== Model Training ==== ##
#
#               ## ==== Random Forest Regressor ==== ##
rf_fit = RandomForestRegressor(n_estimators=250)
rf_fit.fit(X=Xnorm,y=ynorm)

#%%
rf_accuracy = rf_fit.score(Xtnorm, ytnorm)

#%%
#               ## ==== Gradient Boosting Regressor ==== ##

#-----------------------------------------------------------------
#Set model parameters
gbfit = GradientBoostingRegressor(n_estimators=250, max_depth=2, loss='ls', random_state=1010)

#Fit model
gbfit.fit(X=X, y=y)

#%% explore GB fit
accuracy = gbfit.score(Xt, yt)
predict = gbfit.predict(Xt)

#%% 
# Model feature importances ranking
importances = gbfit.feature_importances_
indices = np.argsort(importances)[::-1]

print('Feature Importances')

for f in range(X.shape[1]):
    print("feature %s (%f)" % (list(X)[f], importances[indices[f]]))


feat_imp = pd.DataFrame({'Feature':list(X),
                         'Gini Importance':importances[indices]})
plt.rcParams['figure.figsize']=(8,12)
sb.set_style('whitegrid')
ax = sb.barplot(x='Gini Importance', y='Feature', data=feat_imp)
ax.set(xlabel='Gini Importance')
plt.show()  
    
#%% 
#..............................................
# ----------- Exploration --------------------
#..............................................

static.isnull().sum().sum()
static.notnull().sum().sum()
#%%

df = htrain.sort_index(axis=1,ascending=True)
plt.rcParams['figure.figsize']=(8,14)
#...............................................................
#     ----------- Correlation Plot -----------
#...............................................................
 
corrdf = trNum.corr()
ycorr = corrdf[['SalePrice']]
ycorr = ycorr.sort_index(axis=0,ascending=False)
#ycorr = ycorr.sort_values(by=['SalePrice'])
ycorr.drop(['SalePrice']).plot(kind='barh')


    
#%%
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#       =========  #PCA  --- Principal Components ========== -------

#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Create a different normalized dataframe for PCA
trPCA = (trNum - trNum.mean()) / trNum.std()

i = np.identity(trPCA.drop('SalePrice', axis=1).shape[1])

pca = PCA(n_components=5, random_state=1010)
pca.fit_transform(trPCA.drop('SalePrice', axis=1).values)

coef = pca.transform(i)
pcp = pd.DataFrame(coef, columns = ['PC-1','PC-2','PC-3','PC-4','PC-5'],
                           index = trPCA.drop('SalePrice', axis=1).columns)

pcp['max'] = pcp.max(axis=1)
pcp['sum'] = pcp.drop('max',axis=1).abs().sum(axis=1)

pcp = pcp.sort_values(by=['max'], ascending=False)

plt.rcParams['figure.figsize']=(10,20)
sb.heatmap(pcp, annot=True, annot_kws={"size": 12})

#%% Combine PCA results and Pearson Correlation results ---<><><><><><><><><><>

# --------------------- <><><><>< * * * * * * * * * * * * * * * * * ><><><><>
corrdf = trNum.corr()
ycorr = corrdf[['SalePrice']]
ycorr = ycorr.rename(columns={'SalePrice':'YCorrelation'})
fsel = pd.concat([ycorr.drop('SalePrice',axis=0), pcp], axis=1)
fsel['Feature'] = fsel.index

comb = pd.merge(feat_imp,fsel,on='Feature')
comb.index = comb.Feature
comb = comb.drop('Feature',axis=1)
sb.heatmap(comb, annot=True, annot_kws={"size": 12})

#%%
# -----  -----  save top components

#conditional top
#top = list(pcp[(pcp['max'] > pcp['max'].mean())].index)

#sorted list
pclist = list(pcp.index)

#%%
#<> ----  Create new train and test sets based on top principal components ---
#<><>
#<><><>
#<><><><>
#Set up training and test sets

pcdata = pd.concat([trNum[pclist[0:45]],trNum['SalePrice']],axis=1)
nmdata = pd.concat([trNorm[pclist[0:45]],trNorm['SalePrice']],axis=1)
#nmonly = pd.concat([trNum_norm[list(numht.drop('SalePrice',axis=1))],trNum_norm['SalePrice']],axis=1)

train, test = train_test_split(pcdata, test_size = .30, random_state = 1010)

# Train outcome and predictors 
y = train.SalePrice
X = train.drop('SalePrice', axis=1)
# Test outcome and predictors
yt = test.SalePrice
Xt = test.drop('SalePrice', axis=1)

# Create normalized train and test sets
train, test = train_test_split(nmdata, test_size = .30, random_state = 1010)
#Norm train
ynorm = train.SalePrice
Xnorm = train.drop('SalePrice', axis=1)
#Norm test
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

#_________________________________________________________________________________>  
#_________________________________________________________________________________> 

# End Gradient Boosting .. 
#_________________________________________________________________________________>  
htrain = htrain[htrain.GrLivArea > 4000]
#_________________________________________________________________________________>  
#_________________________________________________________________________________>    
#########

#%%
#          ___ Combine All Feature Ranking data ___

# --------------------- <><><><>< * * * * * * * * * * * * * * * * * ><><><><>
corrdf = trNum.corr()
ycorr = corrdf[['SalePrice']]
ycorr = ycorr.rename(columns={'SalePrice':'YCorrelation'})
fsel = pd.concat([ycorr.drop('SalePrice',axis=0), pcp], axis=1)
fsel['Feature'] = fsel.index

comb = pd.merge(feat_imp,fsel,on='Feature')
comb.index = comb.Feature
comb = comb.drop('Feature',axis=1)
sb.heatmap(comb, annot=True, annot_kws={"size": 12})


#%%#   ------- Linear Models --------
reg = linear_model.Lasso(alpha=1.4, max_iter=2000)
reg.fit(X,y)
accuracy = reg.score(Xt,yt)


########
#######
#%%
#%%
#%%
#
# --------------------- MLP Regressor ---------------------------
#
######

mlpreg = MLPRegressor(solver='lbfgs', alpha=.5, hidden_layer_sizes=(100,), random_state=10)
mlpRfit = mlpreg.fit(Xnorm, ynorm)
accuracy = mlpRfit.score(Xtnorm, ytnorm)


#%%
#%%  
#%%  
#%%  
#%%  
#%%  



#----------------------------------------------------------------------------------------]\\
#----------------------------------------------------------------------------------------]\\
#----------------------------------------------------------------------------------------]\\
#----------------------------------------------------------------------------------------]\\
    #%%
# -------------- Create Bins for SalesPrice Variable --------------------
# ...
## ... for use with Classification Models


#bins = [0,100000,150000,200000,250000,300000,350000,400000,
##       450000,500000,550000,600000,650000,700000]

# .. need to create labels vector for 'mix type error (string and num)

bins=5
trNum['PriceRange'] = pd.cut(trNum.SalePrice, bins)
trNorm['PriceRange'] = pd.cut(trNum.SalePrice, bins)

#%%
#Set up training and test sets for classification

#cats = pd.concat([trNum_norm,trNum_norm['PriceRange']],axis=1)
cats = trNorm.drop('SalePrice',axis=1)
train, test = train_test_split(cats, test_size = .25, random_state=10)

# Train outcome and predictors 
y = train.PriceRange
X = train.drop('PriceRange', axis=1)

# Test outcome and predictors
yt = test.PriceRange
Xt = test.drop('PriceRange', axis=1)



#%%
# >>>>>>>>>>>>>>> Random Forest <<<<<<<<<<<<<<<<<<<<<

forest = RandomForestClassifier(n_estimators=1000, random_state=1010)
rfit = forest.fit(X,y)
accuracy = rfit.score(Xt,yt)



#%%
# >>>>>>>>>>>>>>> MLP Classifier <<<<<<<<<<<<<<<<<<<<<

mlp = MLPClassifier(solver='lbfgs', alpha=5, hidden_layer_sizes=(500,), random_state=10)

fit = mlp.fit(X,y)
accuracy = fit.score(Xt,yt)

#%%
