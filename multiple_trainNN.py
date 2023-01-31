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
from scipy import interpolate

#import ROOT

import HelperFunctions
import importlib
importlib.reload(HelperFunctions)

# =========================================================================================================================================================================
# import trainable dataframe
#df = pd.read_pickle("dataframes/ready_jj.pkl")
df = pd.read_pickle("dataframes/ordered_dataframe.pkl")
#df = pd.read_pickle("dataframes/clean_ordered_dataframe.pkl")
#df_orig = pd.read_pickle("dataframes/clean_ordered_scaled_dataframe.pkl")
#df_orig = pd.read_pickle("dataframes/NOTclean_ordered_scaled_dataframe.pkl")
#df_orig = pd.read_pickle("dataframes/NOTclean_ordered_scaled_dataframe_012.pkl")
#df = df_orig

# import hh4b sample
#df_orig_hh4b = pd.read_pickle("dataframes/clean_ordered_scaled_dataframe_hh4b.pkl")
#df_orig_hh4b = pd.read_pickle("dataframes/NOTclean_ordered_scaled_dataframe_hh4b.pkl")
#df_hh4b = pd.read_pickle("dataframes/ready_hh4b.pkl")
#df_hh4b = pd.read_pickle("dataframes/clean_ordered_dataframe_hh4b.pkl")
df_hh4b = pd.read_pickle("dataframes/ordered_dataframe_hh4b.pkl")

# =========================================================================================================================================================================
# clean 
def clean_df(df,var_str,max_val,min_val):
    df = df[df[var_str] <= max_val]
    df = df[df[var_str] >= min_val]
    return df


LUP = 400  # set lowest unprescale, sólo relevante para pt0 porque estan ordenados
var_str_vec = ['pt0','pt1','pt2','pt3','pt4','pt5']
min_val = 30 # esto ya esta por default en la muestra, sólo relevante para el menos energético porque estan ordenados
for var_str in var_str_vec:
    df = clean_df(df,var_str,LUP,min_val)
    df_hh4b = clean_df(df_hh4b,var_str,LUP,min_val)


# =========================================================================================================================================================================
# add new signal definitions
def add_signal(df):
    j0 = (df.isHS0>=0).astype(float)
    j1 = (df.isHS1>=0).astype(float)
    j2 = (df.isHS2>=0).astype(float)
    j3 = (df.isHS3>=0).astype(float)
    j4 = (df.isHS4>=0).astype(float)
    j5 = (df.isHS5>=0).astype(float)
    sum_4 = j0+j1+j2+j3
    sum_5 = j0+j1+j2+j3+j4
    sum_6 = j0+j1+j2+j3+j4+j5
    df['s_or_b_4']    = pd.Series(sum_4).between(3.5,10).astype(float)
    df['s_or_b_4in5'] = pd.Series(sum_5).between(3.5,10).astype(float)
    df['s_or_b_4in6'] = pd.Series(sum_6).between(3.5,10).astype(float)
    return df

df = add_signal(df)
df_hh4b = add_signal(df_hh4b)
#df['s_or_b_4']=((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS2>=0)&(df.isHS3>=0)).astype(float)
#df['s_or_b_4_inOrder']=((df.isHS0==0)&(df.isHS1==1)&(df.isHS2==2)&(df.isHS3==3)).astype(float)
#df_hh4b['s_or_b_4']=((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)).astype(float)
#df_hh4b['s_or_b_4_inOrder']=((df_hh4b.isHS0==0)&(df_hh4b.isHS1==1)&(df_hh4b.isHS2==2)&(df_hh4b.isHS3==3)).astype(float)
#df['s_or_b_4in5']=(((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS2>=0)&(df.isHS3>=0)) | ((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS2>=0)&(df.isHS4>=0)) | ((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS3>=0)&(df.isHS4>=0)) | ((df.isHS0>=0)&(df.isHS2>=0)&(df.isHS3>=0)&(df.isHS4>=0)) | ((df.isHS1>=0)&(df.isHS2>=0)&(df.isHS3>=0)&(df.isHS4>=0))).astype(float)
#df_hh4b['s_or_b_4in5']=(((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)) | ((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS4>=0)) | ((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS3>=0)&(df_hh4b.isHS4>=0)) | ((df_hh4b.isHS0>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)&(df_hh4b.isHS4>=0)) | ((df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)&(df_hh4b.isHS4>=0))).astype(float)
#df['s_or_b_4in6']=(((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS2>=0)&(df.isHS3>=0)) | ((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS2>=0)&(df.isHS4>=0)) | ((df.isHS0>=0)&(df.isHS1>=0)&(df.isHS3>=0)&(df.isHS4>=0)) | ((df.isHS0>=0)&(df.isHS2>=0)&(df.isHS3>=0)&(df.isHS4>=0)) | ((df.isHS1>=0)&(df.isHS2>=0)&(df.isHS3>=0)&(df.isHS4>=0))).astype(float)
#df_hh4b['s_or_b_4in6']=(((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)) | ((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS4>=0)) | ((df_hh4b.isHS0>=0)&(df_hh4b.isHS1>=0)&(df_hh4b.isHS3>=0)&(df_hh4b.isHS4>=0)) | ((df_hh4b.isHS0>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)&(df_hh4b.isHS4>=0)) | ((df_hh4b.isHS1>=0)&(df_hh4b.isHS2>=0)&(df_hh4b.isHS3>=0)&(df_hh4b.isHS4>=0))).astype(float)

# =========================================================================================================================================================================
# take a fraction for tests
# df = df.sample(frac=0.2)
# df_hh4b = df_hh4b.sample(frac=0.2)

# =========================================================================================================================================================================
# set NN inputs
array_of_vector_string = [ 
                           ["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3","m0","m1","m2","m3","pt4","eta4","phi4","m4","pt5","eta5","phi5","m5"],
                           ["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3","m0","m1","m2","m3","pt4","eta4","phi4","m4"],
                           ["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3","m0","m1","m2","m3"],
                           #["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3","m0","m1","m2","m3","HT_pt30"],
                           #["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3","m0","m1","m2","m3","HT_n8"],
                           #["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3","m0","m1","m2","m3"],
                           #["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3","m0","m1","m2","m3"],
                           #["pt0","pt1","pt2","pt3","pt4"],
                           #["pt0","pt1","pt2","pt3"],
                           #["ratio40", "ratio41", "ratio42", "ratio43", "ratio30", "ratio31", "ratio32", "ratio20", "ratio21", "ratio10", "dphi01", "dphi02", "dphi03", "dphi04", "dphi12", "dphi13", "dphi14","dphi23", "dphi24","dphi34"],
                           #["ratio30", "ratio31", "ratio32", "ratio20", "ratio21", "ratio10", "dphi01", "dphi02", "dphi03", "dphi12", "dphi13","dphi23"],
                           #
                           #["pt0_uncalib","pt1_uncalib","pt2_uncalib","pt3_uncalib","eta0_uncalib","eta1_uncalib","eta2_uncalib","eta3_uncalib","phi0","phi1","phi2","phi3","m0","m1","m2","m3"] ,
                           #["pt0_uncalib","pt1_uncalib","pt2_uncalib","pt3_uncalib"] ,
                           #["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3"],
                           #["pt0","pt1","pt2","pt3","eta0","eta1","eta2","eta3"],
                           #["eta0","eta1","eta2","eta3","phi0","phi1","phi2","phi3"],
                           #["pt0"],
                           #["ratio30", "dphi03"],
                           #['pt3','pt4','m3','m4','ratio43','ratio32','ratio21','ratio10'],
                         ]

array_of_input_id =      [      
                           "6_jets_pt_eta_phi_m_calib",  
                           "5_jets_pt_eta_phi_m_calib",  
                           "4_jets_pt_eta_phi_m_calib",
                           #"4_jets_pt_eta_phi_m_HT_pt30_calib",
                           #"4_jets_pt_eta_phi_m_HT_n8_calib",
                           #"4_jets_pt_eta_phi_m_calib",
                           #"4_jets_pt_eta_phi_m_calib",
                           #"5_jets_pt_calib", 
                           #"4_jets_pt_calib", 
                           #"5_jets_ratio_XY_dphi_XY",
                           #"4_jets_ratio_XY_dphi_XY",
                           #
                           #"4_jets_pt_eta_phi_m_uncalib",
                           #"4_jets_pt_eta_phi_calib",
                           #"4_jets_pt_eta_calib",
                           #"4_jets_eta_phi_calib", 
                           #"4_jets_pt_uncalib", 
                           #"1_jets_pt_calib",
                           #"2_jets_ratio_03_dphi_03"
                           #"pt_m_3_4_ratio_43_32_21_10"
                         ]

array_of_input_labels =  [
                           "6 jets (pt,eta,phi,m)",
                           "5 jets (pt,eta,phi,m)",
                           "4 jets (pt,eta,phi,m)",
                           #"4 jets (pt,eta,phi,m), HT ( pt > 30 GeV )",
                           #"4 jets (pt,eta,phi,m), HT ( 8 jets)",
                           #"4 jets (pt,eta,phi,m)",
                           #"4 jets (pt,eta,phi,m)",
                           #"5 jets (pt)",
                           #"4 jets (pt)",
                           #"5 jets ratio XY, dphi XY",
                           #"4 jets ratio XY, dphi XY",
                           #
                           #"4 jets (pt,eta,phi,m) uncalibrated",
                           #"4 jets (pt) uncalibrated",
                           #"4 jets (pt,eta,phi) calibrated",
                           #"4 jets (pt,eta) calibrated",
                           #"4 jets (eta,phi) calibrated",
                           #"1 jets (pt) calibrated ",
                           #"2 jets ratio 03, dphi 03",
                           #"pt m 3 4, ratio 43 32 21 10"
                         ]

epochs_vec    = [50 ,50 ,50 ,50 ,50, 50 ,50 ,50 ,50 ,50]
batchSize_vec = [128,128,128,128,128,128,128,128,128,128]
layers_vec    = [3  ,3  ,3  ,3  ,3  ,3  ,3  ,3  ,3  ,3  ]
nodes_vec     = [8  ,8  ,8  ,8  ,8  ,8  ,8  ,8  ,8  ,8  ]

signal_def    = ["s_or_b_4in6","s_or_b_4in5","s_or_b_4"]
signal_labels    = ["4 HS in 6","4 HS in 5","4 HS in 4"]
signal_ROC    = ["s_or_b","s_or_b","s_or_b","s_or_b","s_or_b","s_or_b","s_or_b","s_or_b","s_or_b","s_or_b"]

# ========================================================================================================================
# settings
uploadModel_vec = [False,False,False,False,False,False,False,False,False,False]
#uploadModel_vec = [True,True,True,False]
normalize_NNoutput_hists = True

# ========================================================================================================================
# train
def equalize(df_,s_or_b_column):
    fraction = (float( len(df_[df_[s_or_b_column]> 0]) )  /  float( len(df_[df_[s_or_b_column]<=0]) )  )
    if fraction < 1: 
        df_ = df_.drop(df_[df_[s_or_b_column] < 0.5].sample(frac=     1.-    fraction).index)
    else: 
        df_ = df_.drop(df_[df_[s_or_b_column] > 0.5].sample(frac=     1.- 1./fraction).index)
    return df_

evaluate_pTcut = 400.0
importlib.reload(HelperFunctions)
result_matrix = []

for s_or_b_column,s_or_b_ROC,vector_string,input_id,epochs,batchSize,layers,nodes, uploadModel in zip(signal_def,signal_ROC,array_of_vector_string,array_of_input_id,epochs_vec,batchSize_vec,layers_vec,nodes_vec,uploadModel_vec):
    df_equal = equalize(df,s_or_b_column)
    X_train, X_test, y_train, y_predict_train, w_train, y_test, y_predict_test, w_test, pred_signal_train, pred_back_train, pred_signal_w_train, pred_back_w_train, pred_signal_test, pred_back_test, pred_signal_w_test, pred_back_w_test, X_hh4b, y_hh4b, y_predict_hh4b, w_hh4b, pred_signal_hh4b, pred_back_hh4b, pred_signal_w_hh4b, pred_back_w_hh4b , pred_hh4b, pred_hh4b_w,X_QCD,y_QCD, y_predict_QCD,w_QCD, pred_signal_QCD, pred_back_QCD, pred_signal_w_QCD, pred_back_w_QCD, pred_QCD, pred_QCD_w = HelperFunctions.trainAndPlot(df_equal,df,vector_string,input_id,epochs,batchSize,layers,nodes,df_hh4b,uploadModel,s_or_b_column,s_or_b_ROC,evaluate_pTcut)
    HelperFunctions.plotNNdist_addHH4B(pred_signal_train, pred_back_train, pred_signal_w_train, pred_back_w_train, pred_signal_test, pred_back_test, pred_signal_w_test, pred_back_w_test,input_id,pred_signal_hh4b, pred_back_hh4b, pred_signal_w_hh4b, pred_back_w_hh4b,pred_hh4b, pred_hh4b_w, pred_signal_QCD, pred_back_QCD, pred_signal_w_QCD, pred_back_w_QCD, pred_QCD, pred_QCD_w, normalize_NNoutput_hists,s_or_b_column,s_or_b_ROC)
    result_matrix.append( [y_train, y_predict_train, w_train, y_test, y_predict_test, w_test, X_hh4b, y_hh4b, y_predict_hh4b, w_hh4b,X_train, X_test,y_QCD,y_predict_QCD,w_QCD,X_QCD ] )
    del df_equal


# ========================================================================================================================
# plot roc curves
importlib.reload(HelperFunctions)
doFancyPlot = True
zoom = True
threshold = 0.6
acceptance = 0.9
fig = plt.figure(figsize=(9,7),constrained_layout=True) # default is (8,6) 
ax1 = fig.add_subplot()           # Add the primary axis
ax2 = ax1.twinx()
best_thresholds_signal = []
fixAcc_thresholds_signal = []
best_thresholds = []
fixAcc_thresholds = []
colors = ["tab:orange","tab:purple","tab:red"]#,"tab:green","tab:blue","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
#colors = ["tab:red","tab:green","tab:blue"]
for result,input_label,signalTag,color in zip(result_matrix,array_of_input_labels,signal_labels,colors):
    y_train, y_predict_train, w_train, y_test, y_predict_test, w_test, X_hh4b, y_hh4b, y_predict_hh4b, w_hh4b,X_train, X_test,y_QCD,y_predict_QCD,w_QCD,X_QCD  = result
    frac_hh4b_pass = HelperFunctions.get_pass_fraction(y_predict_hh4b,w_hh4b,threshold)
    best_threshold,fixAcc_threshold = HelperFunctions.plot_roc(fig,ax1,ax2,y_QCD,y_predict_QCD,w_QCD,threshold,acceptance,frac_hh4b_pass,y_predict_hh4b,input_label+' - NN signal: '+signalTag,2,color=color,alpha_i=0.8,linestyle='solid',fancy_plot=doFancyPlot,do_zoom=zoom,plotError=True)
    #best_threshold,fixAcc_threshold = HelperFunctions.plot_roc(fig,ax1,ax2,y_train,y_predict_train,w_train,threshold,acceptance,frac_hh4b_pass,y_predict_hh4b,input_label+' - signal: '+signalTag,2,color=color,alpha_i=0.8,linestyle='dotted',fancy_plot=doFancyPlot,do_zoom=zoom)
    best_thresholds.append(best_threshold)
    fixAcc_thresholds.append(fixAcc_threshold)


ax1.set_title('Signal: 4 HS jets in the event')
ax1.legend(loc='upper right',prop={'size': 9})#, bbox_to_anchor=(1.1, 0.5))
ax1.grid()
ax1.grid(visible=True, which='minor',color='grey', linestyle='--',alpha=0.3)
#ax2.yaxis.set_major_locator(ticker.NullLocator())
plt.minorticks_on()
plt.savefig('ROC.png')
plt.clf()
plt.close('all')
'''
# ========================================================================================================================
# asym triggers
LUP = 1
colors = ['black','gray','black','gray','yellow','orange','orange']
#colors = ['red']
sizes = [110,110,40,40,100,100,40]
#sizes = [100]
#labels = ['(130,85,55,45) GeV cut (0.5 Eff)','(145,85,55,45) GeV cut (0.5 Eff)','(95,65,50,40) GeV cut (0.2 Eff)','(105,70,55,40) GeV cut (0.2 Eff)']
labels = ['(150,85,60,45) GeV cut (0.5 Eff)','(135,85,60,45) GeV cut (0.5 Eff)','(100,70,50,40) GeV cut (0.2 Eff)','(90,60,50,40) GeV cut (0.2 Eff)','(135,85,60,45,35) GeV cut (0.5 Eff)','(75,65,60,50) GeV cut (0.5 Eff hh4b based)','(65,60,50,45) GeV cut (0.2 Eff hh4b based)']
#labels = ['(135,85,60,45,35) GeV cut (0.5 Eff)']
#pt_threshs =[ [130, 85, 55, 45], [145, 85, 55, 45], [95, 65, 50, 40], [105, 70, 55, 40]]
pt_threshs =[ [150, 85, 60, 45], [135, 85, 60, 45], [100, 70, 50, 40], [90, 60, 50, 40],[35,85,60,45,35],[75,65,60,50],[65,60,50,45]]
#pt_threshs =[ [135, 85, 60, 45, 35]]
for pt_thresh,label,size,color in zip(pt_threshs,labels,sizes,colors):
    pt0_thresh = pt_thresh[0]
    pt1_thresh = pt_thresh[1]
    pt2_thresh = pt_thresh[2]
    pt3_thresh = pt_thresh[3]
    df_equal = equalize(df,s_or_b_column)
    if len(pt_thresh) < 5:
        x_,y_ = HelperFunctions.getPoint(df_equal,pt0_thresh,pt1_thresh,pt2_thresh,pt3_thresh,LUP,is_pt4_thresh=False)
    elif len(pt_thresh) == 5:
        x_,y_ = HelperFunctions.getPoint(df_equal,pt0_thresh,pt1_thresh,pt2_thresh,pt3_thresh,LUP,is_pt4_thresh=True,pt4_thresh=pt_thresh[4])    
    if doFancyPlot:
        ax1.scatter(y_,1.0/x_,label=label,s=size,color=color,edgecolors='black')
    else:
        ax1.scatter(y_,x_,label=label,s=size,color=color,edgecolors='black')
# ========================================================================================================================
'''


# ========================================================================================================================
# plot pt spectrums
def Extract(lst):
    return [[item[0],item[1],item[2],item[3]] for item in lst]

for result,input_label,signalTag,color in zip(result_matrix,array_of_input_labels,signal_labels,colors):
    y_train, y_predict_train, w_train, y_test, y_predict_test, w_test, X_hh4b, y_hh4b, y_predict_hh4b, w_hh4b,X_train, X_test,y_QCD,y_predict_QCD,w_QCD,X_QCD = result
    if "ratio" in input_id:
        continue
    pt_hist = []
    weight_hist = []
    pt_hist_pass = []
    weight_hist_pass = []
    for NN_QCD_X,NN_QCD_y,NN_predict_QCD,NN_weight_QCD in zip(X_QCD,y_QCD,y_predict_QCD,w_QCD):
        pt_hist.append([NN_QCD_X[0],NN_QCD_X[1],NN_QCD_X[2],NN_QCD_X[3]])
        weight_hist.append(NN_weight_QCD)
        if NN_predict_QCD > best_threshold:
            pt_hist_pass.append([NN_QCD_X[0],NN_QCD_X[1],NN_QCD_X[2],NN_QCD_X[3]])
            weight_hist_pass.append(NN_weight_QCD)
    hh4b_pt_hist = []
    hh4b_weight_hist = []
    for NN_hh4b_X,NN_hh4b_y,NN_hh4b_weight in zip(X_hh4b,y_hh4b,w_hh4b):
        hh4b_pt_hist.append([NN_hh4b_X[0],NN_hh4b_X[1],NN_hh4b_X[2],NN_hh4b_X[3]])
        hh4b_weight_hist.append(NN_hh4b_weight)
    Bins = np.linspace(0,450,100)
    y,x,p = plt.hist(Extract(pt_hist),bins=Bins,alpha=0.1,label=["jet 0","jet 1","jet 2","jet 3"],weights=[weight_hist,weight_hist,weight_hist,weight_hist])
    Y_p,X_p,p_p = plt.hist(Extract(pt_hist_pass),bins=Bins,label=["jet 0 pass","jet 1 pass","jet 2 pass","jet 3 pass"],histtype="step",color="blue"  ,weights=[weight_hist_pass,weight_hist_pass,weight_hist_pass,weight_hist_pass], linewidth=1.5)
    plt.hist(Extract(hh4b_pt_hist),bins=Bins,alpha=0.6,label=["jet 0 hh4b","jet 1 hh4b","jet 2 hh4b","jet 3 hh4b"],weights=[hh4b_weight_hist,hh4b_weight_hist,hh4b_weight_hist,hh4b_weight_hist],histtype="step",color="blue"  ,linestyle = "dotted")
    plt.title("NN Inputs: "+ input_label + "  Signal: "+signalTag)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.ylabel('Weighted Entries')
    plt.xlabel('$p_{T}$ [GeV]')
    plt.show()




    x = (x0[:-1]+x0[1:])/2
    plt.plot(x,Y/y,label="jet 0 efficiency",color="blue")
    plt.xlim([20,200])
    plt.ylim([-0.1,1.1])
    plt.ylabel('Efficiency')
    plt.xlabel('$p_{T}$ [GeV]')
    plt.title("NN Inputs: "+ input_label + "  Signal: "+signalTag)
    plt.legend()
    plt.grid()
    plt.show()

        '''
        # and a fully hh4b derived asymm trigger (using qcd weighted numerator):
        Y0,X0,p0 = plt.hist(hh4b_pt0_hist,bins=Bins,alpha=0.6,label="jet 0 hh4b",weights=hh4b_weight_hist,histtype="step",color="blue"  ,linestyle = "dotted")
        Y1,X1,p0 = plt.hist(hh4b_pt1_hist,bins=Bins,alpha=0.6,label="jet 1 hh4b",weights=hh4b_weight_hist,histtype="step",color="orange",linestyle = "dotted")
        Y2,X2,p0 = plt.hist(hh4b_pt2_hist,bins=Bins,alpha=0.6,label="jet 2 hh4b",weights=hh4b_weight_hist,histtype="step",color="green" ,linestyle = "dotted")
        Y3,X3,p0 = plt.hist(hh4b_pt3_hist,bins=Bins,alpha=0.6,label="jet 3 hh4b",weights=hh4b_weight_hist,histtype="step",color="red"   ,linestyle = "dotted")
        #Y4,X4,p0 = plt.hist(hh4b_pt4_hist,bins=Bins,alpha=0.6,label="jet 4 hh4b",weights=hh4b_weight_hist,histtype="step",color="violet"   ,linestyle = "dotted")
        plt.title("NN Inputs: "+ input_label + "  Signal: "+signalTag)
        plt.legend()
        plt.grid()
        plt.yscale("log")
        plt.ylabel('Weighted Entries')
        plt.xlabel('$p_{T}$ [GeV]')
        plt.show()
        x = (x0[:-1]+x0[1:])/2
        plt.plot(x,Y0/y0,label="jet 0 efficiency",color="blue")
        plt.plot(x,Y1/y1,label="jet 1 efficiency",color="orange")
        plt.plot(x,Y2/y2,label="jet 2 efficiency",color="green")
        plt.plot(x,Y3/y3,label="jet 3 efficiency",color="red")
        #plt.plot(x,Y4/y4,label="jet 4 efficiency",color="violet")
        plt.xlim([20,200])
        plt.ylim([-0.1,1.1])
        plt.ylabel('Efficiency')
        plt.xlabel('$p_{T}$ [GeV]')
        plt.title("NN Inputs: "+ input_label + "  Signal: "+signalTag)
        plt.legend()
        plt.grid()
        plt.show()
        '''

        '''
        f0 = interpolate.interp1d(Y0[:-10]/y0[:-10],x[:-10])
        f1 = interpolate.interp1d(Y1[:-10]/y1[:-10],x[:-10])
        f2 = interpolate.interp1d(Y2[:-10]/y2[:-10],x[:-10])
        f3 = interpolate.interp1d(Y3[:-10]/y3[:-10],x[:-10])
        x0_50 = f0(0.5)
        x1_50 = f1(0.5)
        x2_50 = f2(0.5)
        x3_50 = f3(0.5)
        print(x0_50,x1_50,x2_50,x3_50)
        x0_20 = f0(0.2)
        x1_20 = f1(0.2)
        x2_20 = f2(0.2)
        x3_20 = f3(0.2)
        print(x0_20,x1_20,x2_20,x3_20)
        '''



