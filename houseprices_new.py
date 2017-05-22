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
from matplotlib.colors import ListedColormap
import seaborn as sb
import plotly
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn import linear_model
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
teNum = convert_data(htest)


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
#               ## ==== Gradient Boosting Regressor ==== ##

#-----------------------------------------------------------------
#Set model parameters
gbfit = GradientBoostingRegressor(n_estimators=350, max_depth=3, loss='ls', random_state=1010)

#Fit model
gbfit.fit(X=X, y=y)

#%% explore GB fit
accuracy = gbfit.score(Xt, yt)
predict = gbfit.predict(Xt)

print('Gradient Boosting Accuracy %s' % '{0:.2%}'.format(accuracy))    
# Cross_Validation
v = cross_val_score(gbfit, X, y, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))

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
plt.rcParams['figure.figsize']=(10,7)
#...............................................................
#     ----------- Correlation Plot -----------
#...............................................................
 
corrdf = trNum.corr()
ycorr = corrdf[['SalePrice']]
#ycorr = ycorr.sort_index(axis=0,ascending=False)
ycorr = ycorr.sort_values(by=['SalePrice'])
top_ycorr = ycorr[ycorr.SalePrice.abs()>.2]
top_ycorr.drop(['SalePrice']).plot(kind='barh')





#%%
#...................................................................................
#
#                                <Spectral Clustering >
#...................................................................................

#%%
from sklearn.cluster import SpectralClustering

def spectralFeatureAppend(clusters, data):
    sc = SpectralClustering(n_clusters=clusters,
                            eigen_solver='arpack',
                            affinity="rbf")
    sc.fit(np.array(data.drop('SalePrice',axis=1)))
    labels=sc.labels_.astype(np.int)
    return labels
#%%
spec6_labels = spectralFeatureAppend(6, trNum)
spec10_labels = spectralFeatureAppend(10, trNum)
spec12_labels = spectralFeatureAppend(12, trNum)
spec14_labels = spectralFeatureAppend(14, trNum)
spec18_labels = spectralFeatureAppend(18, trNum)
spec24_labels = spectralFeatureAppend(24, trNum)


cData = pd.concat([trNum,
                   pd.Series(spec6_labels, name="Spectral_6"),
                   pd.Series(spec10_labels, name="Spectral_10"),
                   pd.Series(spec12_labels, name="Spectral_12"),
                   pd.Series(spec14_labels, name="Spectral_14"),
                   pd.Series(spec18_labels, name="Spectral_18"),
                   pd.Series(spec24_labels, name="Spectral_24")],axis=1)

#%%
#sb.set(rc={"figure.figsize": (16, 16)})
sb.lmplot("GrLivArea","SalePrice",cData,hue="Spectral_24",
          fit_reg=False,size=10,aspect=1,scatter_kws={"s": 55, 'alpha':0.30})
plt.legend()

#<><><><><><><><><>---------------------------------------------------<><><><><><><><><><><>  

#%%  Create new arrays with spectral clustering results

train, test = train_test_split(cData, test_size = .30, random_state = 1010)
# Train outcome and predictors 
ySpec = train.SalePrice
XSpec = train.drop('SalePrice', axis=1)
# Test outcome and predictors
ytSpec = test.SalePrice
XtSpec = test.drop('SalePrice', axis=1)

    
#%%
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#       =========  #PCA  --- Principal Components ========== -------

#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Create a different normalized dataframe for PCA
#trPCA = (cData - cData.mean()) / cData.std()
trPCA = (trNum - trNum.mean()) / trNum.std()
i = np.identity(trPCA.drop('SalePrice', axis=1).shape[1])

pca = PCA(n_components=5, random_state=1010)
pca.fit_transform(trPCA.drop('SalePrice', axis=1).values)

coef = pca.transform(i)
pcp = pd.DataFrame(coef, columns = ['PC-1','PC-2','PC-3','PC-4','PC-5'],
                           index = trPCA.drop('SalePrice', axis=1).columns)

pcp['max'] = pcp.max(axis=1)
pcp['sum'] = pcp.drop('max',axis=1).abs().sum(axis=1)

pcp = pcp.sort_values(by=['sum'], ascending=False)

plt.rcParams['figure.figsize']=(12,15)
sb.heatmap(pcp.ix[:50,:], annot=True, annot_kws={"size": 12})

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
comb = comb.sort_values(by=['sum'], ascending=False)
sb.heatmap(comb, annot=True, annot_kws={"size": 12})

#%%

# Drop Spectral Predictors that rank low





#%%
#<> ----  Create new train and test sets based on top principal components ---
#<><>
#<><><>
#<><><><>

# -----  -----  save top components

#conditional top
#top = list(pcp[(pcp['max'] > pcp['max'].mean())].index)

#sorted list
pclist = list(pcp.index)
#Set up training and test sets
trNorm = (trNum - trNum.mean()) / (trNum.max() - trNum.min())
pcdata = pd.concat([trNum[pclist[0:55]],trNum['SalePrice']],axis=1)
nmdata = pd.concat([trNorm[pclist],trNorm['SalePrice']],axis=1)
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

gbr_results = pd.DataFrame({'Predicted':predict,
                            'Ground Truth':yt})

print('Gradient Boosting Accuracy %s' % '{0:.2%}'.format(accuracy))    
# Cross_Validation
v = cross_val_score(gbfit, X, y, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))
#%% GB with Normalized variables
gbfit.fit(X=Xnorm, y=ynorm)
accuracy = gbfit.score(Xtnorm,ytnorm)
predict = gbfit.predict(Xtnorm)



#%% 

#_________________________________________________________________________________>  
#_________________________________________________________________________________> 

# End Gradient Boosting .. 
#_________________________________________________________________________________>  

#%%
#               ## ==== Model Training ==== ##
#
#               ## ==== Random Forest Regressor ==== ##
rf_fit = RandomForestRegressor(n_estimators=1000)
rf_fit.fit(X=X,y=y)

rf_accuracy = rf_fit.score(Xt, yt)
print('Random Forest Regressor Accuracy %s' % '{0:.2%}'.format(rf_accuracy))    
# Cross_Validation
v = cross_val_score(rf_fit, X, y, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))
#%%#   ------- Linear Models --------
reg = linear_model.Lasso(alpha=500, max_iter=2000)
reg.fit(X,y)
lasso_accuracy = reg.score(Xt,yt)

print('Lasso Regression Accuracy %s' % '{0:.2%}'.format(lasso_accuracy))    
# Cross_Validation
v = cross_val_score(reg, X, y, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))
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
mlpRaccuracy = mlpRfit.score(Xtnorm, ytnorm)

print('MLP Regression Accuracy %s' % '{0:.2%}'.format(mlpRaccuracy))    
# Cross_Validation
v = cross_val_score(mlpRfit, X, y, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))


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
trNum['PriceRange'] = pd.cut(cData.SalePrice, bins)
trNorm['PriceRange'] = pd.cut(cData.SalePrice, bins)
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
rfc_accuracy = rfit.score(Xt,yt)


print('Random Forest Classifier %s' % '{0:.2%}'.format(rfc_accuracy))    
# Cross_Validation
v = cross_val_score(rfit, X, y, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))
#%%
# >>>>>>>>>>>>>>> MLP Classifier <<<<<<<<<<<<<<<<<<<<<

mlp = MLPClassifier(solver='lbfgs', alpha=5, hidden_layer_sizes=(500,), random_state=10)

mlpfit = mlp.fit(X,y)
mlp_accuracy = mlpfit.score(Xt,yt)

print('MLP Classifier %s' % '{0:.2%}'.format(mlp_accuracy))    
# Cross_Validation
v = cross_val_score(mlpfit, X, y, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))

#%%
