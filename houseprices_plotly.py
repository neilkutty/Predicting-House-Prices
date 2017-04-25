#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:34:16 2017

@author: NNK
"""

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
l = go.Layout(title='Correlation of All Variables to Sale Price',

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
        colorscale='Jet')]

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
