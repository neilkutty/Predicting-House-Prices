#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:34:16 2017

@author: NNK
"""
import pandas as pd
import numpy as np
import os
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

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
##
###
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

ytnorm = test.SalePrice
Xtnorm = test.drop('SalePrice', axis=1)


#%%

#...............................................................
#     ----------- Plotly - Correlation Plot -----------
#...............................................................

corrdf = trNum.corr()
ycorr = corrdf[['SalePrice']]
ycorr = ycorr.sort_values(by=['SalePrice'])

l = go.Layout(title='Correlation of All Variables to Sale Price',
#    xaxis=dict(
#        autorange=True,
#        showgrid=False,
#        zeroline=True,
#        showline=False,
#        autotick=True,
#        ticks='-',
#        showticklabels=False
#    ),
#    yaxis=dict(
#        autorange=True,
#        showgrid=False,
#        zeroline=False,
#        showline=False,
#        autotick=False,
#        ticks='-',
#        showticklabels=True
#    ),
      autosize=False,
     # width=500,
      height=2000,
      margin=go.Margin(
              l=200,
              r=50,
              #b=100,
              #t=100,
              pad=1
              )
      )
              
d = [go.Bar(y=ycorr[(ycorr.index <> 'SalePrice')].index,x=ycorr.SalePrice,
            marker=dict(                    
                    color = ycorr.SalePrice, #set color equal to a variable
                    colorscale='Viridis',
                    showscale=True
                    ),
            orientation='h')]

fig = go.Figure(data=d,layout=l)
#cf.iplot(fig, filename='correlation')
py.plot(fig, filename='correlation')

#%%
#...............................................................
#     ----------- Plotly - Correlation Heatmap -----------
#...............................................................
corrdf = corrdf.sort_index(axis=1,ascending=True).sort_index(axis=0,ascending=True)
l = go.Layout(title='Correlation of All Variables to Each Other',

      autosize=True,
              width=1000,
              height=1000,
              margin=go.Margin(
                  l=200,
                  r=50,
                  #b=100,
                  #t=100,
                  pad=1
                  )
              )
              
d = [go.Heatmap(
        z=corrdf.values.tolist(),
        x=corrdf.columns.tolist(),
        y=corrdf.index.tolist(),
        colorscale='RdBu')]

fig = go.Figure(data=d,layout=l)
#cf.iplot(fig, filename='correlation-heatmap')
py.plot(fig, filename='correlation-heatmap')


#%%  

#       =========  #Plotly - Exploratory  ==========



plotly.tools.set_credentials_file(username='sampsonsimpson', 
                                  api_key='cxm2iF7KKBGZgXmDOU9S')

buttons = []
for i in list(trNum):
    buttons.append(dict(args=[i],
             label=i, method='restyle'))

layout = go.Layout(
        title='House Price vs. Living Area',
        updatemenus=list([
        dict(x=-0.1, y=0.7,
             yanchor='middle',
             bgcolor='c7c7c7',
             borderwidth=2,
             bordercolor='#ffd6dc',
             buttons=list(buttons)),
             ]),
             )
        
#traces = []
#for j in list(trNum):
#    traces.append(
#            go.Scatter(
#            x=trNum[j],
#            y=trNum.SalePrice,
#            mode = 'markers',
#            marker= dict(size= 14,
#                    line= dict(width=1),
#                    color= trNum['SalePrice'],
#                    opacity= 0.7
#                        )
#                    )
#            )
d = [
     go.Scatter(
             x=trNum['GrLivArea'],
             y=trNum['SalePrice'],
             mode = "markers",
             marker= dict(size= 14,
                    line= dict(width=1),
                    color= trNum['SalePrice'],
                    opacity= 0.7
                   ))]


fig = go.Figure(data=d, layout=layout)

py.plot(fig, filename='house-prices')

# ---  Jupyter Notebook Version
#cf.iplot(fig, filename='house-prices')

#%% 

#           Plotly Gradient Boosting Results

gbr_results = pd.DataFrame({'Predicted':predict,
                            'Ground Truth':yt})

print('Gradient Boosting Accuracy %s' % '{0:.2%}'.format(accuracy))    
layout = go.Layout(
        title='Gradient Boosting Results - All Features'
        )

d = [
     go.Scatter(
             x=gbr_results.Predicted,
             y=gbr_results['Ground Truth'],
             mode = "markers",
             marker= dict(size= 14,
                    line= dict(width=1),
                    color= gbr_results['Ground Truth'] - gbr_results['Predicted'],
                    colorscale = 'Viridis',
                    opacity= 0.7
                   ))]


fig = go.Figure(data=d, layout=layout)

py.plot(fig, filename='GBR-Results')
#cf.iplot(fig, filename='GBR-Results')
