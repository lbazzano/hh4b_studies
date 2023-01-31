import os
import uproot3 as uproot
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, plot_roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from numpy.random import seed
import math
import ROOT

vec_df = [pd.read_pickle("dataframes/clean_ordered_dataframe.pkl"),pd.read_pickle("dataframes/NOTclean_ordered_dataframe_012.pkl")]
vec_df_hh4b = [pd.read_pickle("dataframes/clean_ordered_dataframe_hh4b.pkl"), pd.read_pickle("dataframes/NOTclean_ordered_dataframe_hh4b.pkl")]
vec_sample = ["All Slices (clean)","JZ0 JZ1 JZ2 (raw)"]

vec_df = [df]
vec_df_hh4b = [df_hh4b]
vec_sample = ["All Slices"]

#vec_var = ["ratio10","ratio20","ratio30","ratio40","m0","m1","m2","m3","pt0","pt1","pt2","pt3"]
vec_var = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4', 'eta0','eta1', 'eta2', 'eta3', 'eta4', 'phi0', 'phi1', 'phi2', 'phi3', 'phi4','m0', 'm1', 'm2', 'm3', 'm4']
vec_var = ['ratio40', 'ratio41', 'ratio42', 'ratio43','ratio30', 'ratio31', 'ratio32', 'ratio20', 'ratio21', 'ratio10','dphi01', 'dphi02', 'dphi03', 'dphi04', 'dphi12', 'dphi13', 'dphi14','dphi23', 'dphi24', 'dphi34', 'dphi0X', 'dphiYZ', 'dphi0X5', 'dphiYZ5']
vec_var = ["pt3"]

vec_var = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4', 'eta0','eta1', 'eta2', 'eta3', 'eta4', 'phi0', 'phi1', 'phi2', 'phi3', 'phi4','m0', 'm1', 'm2', 'm3', 'm4']
sig = "s_or_b"
Dens = False
NBins = 1000

## other signal definitions
#df['s_or_b_4']=((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS2>=0)&(df.isHS3>=0)).astype(float)
#df['s_or_b_4_inOrder']=((df.isHS0==0)&(df.isHS1==1)&(df.isHS2==2)&(df.isHS3==3)).astype(float)
#df_hh4b['s_or_b_4']=((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)).astype(float)
#df_hh4b['s_or_b_4_inOrder']=((df_hh4b.isHS0==0)&(df_hh4b.isHS1==1)&(df_hh4b.isHS2==2)&(df_hh4b.isHS3==3)).astype(float)
for var in vec_var:
    for n in [0,1]:
        if n ==1:
            continue
        df = vec_df[n]
        df_hh4b = vec_df_hh4b[n]
        sample = vec_sample[n]
        b_min= min(df[var])
        b_max = max(df[var])    
        # get histos
        signal_df = df[df[sig]>0]
        back_df   = df[df[sig]==0]
        #signal_df = df[df.s_or_b_4>0]
        #back_df   = df[df.s_or_b_4==0]
        pt0_h ,         bins = np.histogram(df[var],         bins=np.linspace(b_min,b_max,NBins),density=Dens,weights=df['weight'])
        signal_pt0_h ,  bins = np.histogram(signal_df[var],  bins=np.linspace(b_min,b_max,NBins),density=Dens,weights=signal_df['weight'])
        back_pt0_h ,    bins = np.histogram(back_df[var],    bins=np.linspace(b_min,b_max,NBins),density=Dens,weights=back_df['weight'])
        hh4b_pt0_h ,    bins = np.histogram(df_hh4b[var],    bins=np.linspace(b_min,b_max,NBins),density=Dens,weights=df_hh4b['weight'])
        # get ratios
        ratio_signal = signal_pt0_h/hh4b_pt0_h
        ratio_back   = back_pt0_h/hh4b_pt0_h
        ratio_hh4b   = hh4b_pt0_h/hh4b_pt0_h
        ratio_backQCD   = back_pt0_h/pt0_h
        ratio_signalQCD   = signal_pt0_h/pt0_h
        # plot
        plt.figure(n+1,figsize=(8,10))
        plt.subplot(211)
        plt.hist(df[var],       bins=np.linspace(b_min,b_max,NBins),alpha=0.8,color="green",histtype = "step",label="QCD",       density=Dens,weights=df['weight'])
        plt.hist(signal_df[var],bins=np.linspace(b_min,b_max,NBins),alpha=0.5,label="QCD signal",density=Dens,weights=signal_df['weight'])
        plt.hist(back_df[var],  bins=np.linspace(b_min,b_max,NBins),alpha=0.5,label="QCD background",density=Dens,weights=back_df['weight'])
        #plt.hist(df_hh4b[var],  bins=np.linspace(b_min,b_max,NBins),alpha=1,  color="black",histtype="step",density=Dens,label="hh4b",weights=df_hh4b['weight'])
        plt.legend()
        plt.yscale('log')
        plt.xlim([b_min,b_max])
        plt.title("sample type: "+sample+"  |  var: "+var+"  |  signal: "+sig)
        plt.subplot(212)
        #plt.plot(bins[:-1],ratio_signal,color="blue",label="qcd signal")
        #plt.plot(bins[:-1],ratio_back,color="orange",label="qcd background")
        #plt.plot(bins[:-1],ratio_hh4b,color="black",label="hh4b")
        plt.plot(bins[:-1],ratio_backQCD,color="orange",label="qcd background")
        plt.plot(bins[:-1],ratio_signalQCD,color="blue",label="qcd signal")
        plt.legend()
        plt.xlim([b_min,b_max])
        plt.ylim([0,2])
        plt.show()
    

