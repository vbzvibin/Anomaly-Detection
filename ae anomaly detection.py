#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:10:20 2019

@author: antonyvibin
"""


### Neural Networks for Anomaly (Outliers) Detection
# https://github.com/abelusha/AutoEncoders-for-Anomaly-Detection

#Data Exploration & Preprocessing
#Model Building
#Model Evaluation
#Model Interpretation

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)

import matplotlib.pyplot as plt
plt.rcdefaults()

from pylab import rcParams

import seaborn as sns
%matplotlib inline

import datetime

############################## Ploltly ########################################

import plotly
import plotly.plotly as py
import plotly.offline as pyo
import plotly.figure_factory as ff
from   plotly.tools import FigureFactory as FF, make_subplots
import plotly.graph_objs as go
from   plotly.graph_objs import *
from   plotly import tools
from   plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import cufflinks as cf

init_notebook_mode(connected=True)
cf.go_offline()
pyo.offline.init_notebook_mode()


########################## Deep learning libraries ############################

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from ann_visualizer.visualize import ann_viz
# 
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_auc_score, auc,
                             precision_score, recall_score, roc_curve, precision_recall_curve,
                             precision_recall_fscore_support, f1_score,
                             precision_recall_fscore_support)
#
from IPython.display import display, Math, Latex



############################### Load Data #####################################

df = pd.read_csv('/Users/antonyvibin/Desktop/Python/Anomaly Detection using Auto Encoder/AutoEncoders-for-Anomaly-Detection-master/JADS_CarrerDay_Data.cvs',index_col=0)

df.index

df.reset_index(inplace=True, drop= True)

df.index

df.head()

df.shape

df.Label.value_counts()

df.Label.value_counts(normalize=True)*100



### Highly Imbalanced Data

df.columns

numerical_cols = ['V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9']

df.Label.unique()

labels = df['Label'].astype(int)

labels[labels != 0] = 1

len(labels[labels !=0])


### Data Exploration

df.Label.value_counts().tolist()

df.Label.astype(str).unique().tolist()

def label_dist(df):    
    x = df.Label.astype(str).unique().tolist()
    y =  df.Label.value_counts().tolist()
    trace = {'type': 'bar', 
             'x'   : x,
             'y'   : y,
             'text' : [str(y[0]),str(y[1])],
             'textposition' : "outside", 'textfont' : {'size' : 20}
             }

    data = Data([trace])
    layout = {'title' : '<b>Data per Label', 'titlefont': {'size':   20},
              'xaxis' : {'title':'<b>Labels',   'titlefont':{'size' : 20}, 'tickfont': {'size':20}},
              'yaxis' : {'title':'<b>Frequency','titlefont':{'size' : 20}, 'tickfont': {'size':20}, 'range': [0,max(y) + 1000]},
            }
    fig = Figure(data = data, layout = layout)
    return py.iplot(fig)


label_dist(df)


def line_plot(df, col):
    
    x = df.index.tolist()
    y = df[col].tolist()
    
    trace = {'type':  'scatter', 
             'x'   :  x,
             'y'   :  y,
             'mode' : 'markers',
             'marker': {'colorscale':'red', 'opacity': 0.5}
            }
    data   = Data([trace])
    layout = {'title': 'Line plot of {}'.format(col), 'titlefont': {'size': 30},
              'xaxis' : {'title' :'Data Index', 'titlefont': {'size' : 20}},
              'yaxis' : {'title': col, 'titlefont' : {'size': 20}},
              'hovermode': 'closest'
             }
    fig = Figure(data = data, layout = layout)
    return fig

def kde_hist(df, col):
    name = [col]
    hist_data = [df[col]]    
    fig = ff.create_distplot(hist_data,name, show_hist=False, bin_size= 0.5)
    return fig



def subplot(df, is_line = True):
    fig_subplots = make_subplots(rows=10, cols=6, horizontal_spacing=0.2,vertical_spacing = 0.1, print_grid=False,
                        subplot_titles=(df.columns.tolist()),
                        specs = [[{'rowspan' : 2, 'colspan' : 3},  None,None, {'rowspan' : 2, 'colspan' : 3}, None,None  ],
                                 [None,None,None, None,None, None],
                                 [{'rowspan' : 2, 'colspan' : 3},  None,None,  {'rowspan' : 2,'colspan':3}, None,None ],
                                 [None,None,None, None,None,None ],
                                 [{'rowspan' : 2, 'colspan' : 3},  None,None, {'rowspan' : 2, 'colspan' : 3}, None,None  ],
                                 [None,None,None, None,None, None],
                                 [{'rowspan' : 2, 'colspan' : 3},  None,None,  {'rowspan' : 2,'colspan':3}, None,None ],
                                 [None,None,None, None,None,None ],
                                 [{'rowspan' : 2, 'colspan' : 3},  None,None,  {'rowspan' : 2,'colspan':3}, None,None ],
                                 [None,None,None, None,None,None ]
                                 ])

 
    figs = {}
    if(is_line):
        n = 1   
        title = 'Line Plots of Variables'
        for col in df.columns.tolist():
            figs[col] =  line_plot(df,col)
    else:
        n = 2
        title = 'Normilized Distribution of Variables'
        for col in df.columns.tolist():
            figs[col] =  kde_hist(df,col)     
        

    for i in range(n):
        fig_subplots.append_trace(figs['V_1']['data'][i]   ,  row = 1, col = 1)
        fig_subplots.append_trace(figs['V_2']['data'][i]   ,  row = 1, col = 4)
        fig_subplots.append_trace(figs['V_3']['data'][i]   ,  row = 3, col = 1)
        fig_subplots.append_trace(figs['V_4']['data'][i]   ,  row = 3, col = 4)
        fig_subplots.append_trace(figs['V_5']['data'][i]   ,  row = 5, col = 1)
        fig_subplots.append_trace(figs['V_6']['data'][i]   ,  row = 5, col = 4)
        fig_subplots.append_trace(figs['V_7']['data'][i]   ,  row = 7, col = 1)
        fig_subplots.append_trace(figs['V_8']['data'][i]   ,  row = 7, col = 4)
        fig_subplots.append_trace(figs['V_9']['data'][i]   ,  row = 9, col = 1)
        fig_subplots.append_trace(figs['Label']['data'][0] ,  row = 9, col = 4)

    fig_subplots['layout'].update({'title' : title, 'titlefont' : {'size' : 20}})
    fig_subplots['layout'].update({ 'height' : 1500, 'width' : 1000, 'showlegend' : False})

    return fig_subplots



lines_subplot = subplot(df, is_line = True)
kdes_subplot  = subplot(df, is_line = False)

pyo.iplot(lines_subplot)

pyo.iplot(kdes_subplot)


# =============================================================================
# AutoEncoders
# Feed Forward Neural Network model for unsupervised tasks (No Lables)
# Model the identity function f(x) ≈ x (It encodes itself)
# Simple to undersatnd!
# Compress data and learn some features
# =============================================================================

## Data Preprocessing

RANDOM_SEED = 101

X_train, X_test = train_test_split(df, test_size=0.2, random_state = RANDOM_SEED)

X_train = X_train[X_train['Label'] == 0]
X_train = X_train.drop(['Label'], axis=1)
y_test  = X_test['Label']
X_test  = X_test.drop(['Label'], axis=1)
X_train = X_train.values
X_test  = X_test.values
print('Training data size   :', X_train.shape)
print('Validation data size :', X_test.shape)


############################ Data Scaling ####################################

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

############################ Modeling ########################################

# No of Neurons in each Layer [9,6,3,2,3,6,9]
input_dim = X_train.shape[1]
encoding_dim = 6

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
encoder = Dense(int(2), activation="tanh")(encoder)
decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()



def train_validation_loss(df_history):
    
    trace = []
    
    for label, loss in zip(['Train', 'Validation'], ['loss', 'val_loss']):
        trace0 = {'type' : 'scatter', 
                  'x'    : df_history.index.tolist(),
                  'y'    : df_history[loss].tolist(),
                  'name' : label,
                  'mode' : 'lines'
                  }
        trace.append(trace0)
    data = Data(trace)
    
    layout = {'title' : 'Model train-vs-validation loss', 'titlefont':{'size' : 30},
              'xaxis' : {'title':  '<b> Epochs', 'titlefont':{ 'size' : 25}},
              'yaxis' : {'title':  '<b> Loss', 'titlefont':{ 'size' : 25}},
              }
    fig = Figure(data = data, layout = layout)
    
    return pyo.iplot(fig)  


nb_epoch = 100
batch_size = 50
autoencoder.compile(optimizer='adam', loss='mse' )

t_ini = datetime.datetime.now()
history = autoencoder.fit(X_train_scaled, X_train_scaled,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=0
                        )

t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))



df_history = pd.DataFrame(history.history)
# validation_data=(X_test_scaled, X_test_scaled)


train_validation_loss(df_history)

################### Predictions & Computing Reconstruction Error ##############

predictions = autoencoder.predict(X_test_scaled)

mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse, 'Label': y_test}, index=y_test.index)
df_error.describe()

fig = line_plot(df_error, 'reconstruction_error')
pyo.iplot(fig)


####################### Model Interpretability ################################

# change X_tes_scaled to pandas dataframe
data_n = pd.DataFrame(X_test_scaled, index= y_test.index, columns=numerical_cols)


def compute_error_per_dim(point):
    
    initial_pt = np.array(data_n.loc[point,:]).reshape(1,9)
    reconstrcuted_pt = autoencoder.predict(initial_pt)
    
    return abs(np.array(initial_pt  - reconstrcuted_pt)[0])


outliers = df_error.index[df_error.reconstruction_error > 0.1].tolist()

RE_per_dim = {}
for ind in outliers:
    RE_per_dim[ind] = compute_error_per_dim(ind)
    
RE_per_dim = pd.DataFrame(RE_per_dim, index= numerical_cols).T

RE_per_dim.head()


def bar_plot(df, data_pt):
    x = df.columns.tolist()
    y = df.loc[data_pt]
    
    trace = {'type': 'bar',
             'x'   : x,
             'y'   : y}
    data = Data([trace])
    layout = {'title' : "<b>Reconstruction error in each dimension for data poitn {}".format(data_pt),
              'titlefont':{'size' : 20},
              'xaxis' : {'title': '<b>Features',
                         'titlefont':{'size' : 20},
                         'tickangle': -45, 'tickfont': {'size':15} },
              'yaxis' : {'title': '<b>Reconstruction Error',
                         'titlefont':{'size' : 20},
                         'tickfont': {'size':15}},
              'margin' : {'l':100, 'r' : 1, 'b': 200, 't': 100, 'pad' : 1},
              'height' : 600, 'width' : 800,
             }
    
    fig = Figure(data = data, layout = layout)
    
    return pyo.iplot(fig)

for pt in outliers[:5]:
    bar_plot(RE_per_dim, pt)
    
    
    
    
# =============================================================================
# Conclusions¶
# There is a patten for V_1 & V_2 in contributing mainly to the reconstruction error!
# AEs can be good choice to reduce dimensions in high dimensional data!
# AEs are good for outlier detections in complex data!    
# =============================================================================
    

df_error.loc[1387]

data_n.loc[1387]

autoencoder.predict(np.array(data_n.loc[1387]).reshape(1,9))

initial_pt = np.array(data_n.loc[point,:]).reshape(1,9)
    reconstrcuted_pt = autoencoder.predict(initial_pt)
    
    
autoencoder.predict()

np.array(1)    







