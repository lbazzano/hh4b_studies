import os
import uproot3 as uproot
import pandas as pd
import numpy as np
from tqdm import tqdm
#import tensorflow as tf
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from keras.models import Sequential, Model
#from keras.layers import Input, Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn import preprocessing
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import roc_curve, plot_roc_curve, auc, roc_auc_score
#from sklearn.model_selection import train_test_split
from numpy.random import seed
import math
import ROOT


# ========================================================================================================================
# prepare data

#slice_folder = "../samples/"
slice_folder = "../calib_jj_samples/"

slice_list = [
    "user.lbazzano.gtp_output_ntuple_JZ0.root",
    "user.lbazzano.gtp_output_ntuple_JZ1.root",
    "user.lbazzano.gtp_output_ntuple_JZ2.root",
    "user.lbazzano.gtp_output_ntuple_JZ3.root",
    "user.lbazzano.gtp_output_ntuple_JZ4.root",
    "user.lbazzano.gtp_output_ntuple_JZ5.root",
    "user.lbazzano.gtp_output_ntuple_JZ6.root",
    "user.lbazzano.gtp_output_ntuple_JZ7.root",
    "user.lbazzano.gtp_output_ntuple_JZ8.root",
]
xSec_list = [
    0.07893 * 0.97308,
    0.07893 * 0.026889,
    0.002684 * 0.010154,
    0.0026843 * 0.00013459,
    2.9772000000000005e-7 * 0.014015,
    5.5408000000000006e-9 * 0.015524,
    3.2602e-10 * 0.010583,
    2.1738000000000003e-11 * 0.013063,
    9.2954e-13 * 0.012908,
    ]

#slice_folder = "../calib_hh4b_samples/"
#slice_list = ["user.lbazzano.gtp_output_ntuple_hh4b.root"]
#xSec_list = [1]

inputMoments = ["s_or_b","weight",
                "pt0","pt1","pt2","pt3","pt4","pt5","pt6","pt7","pt8","pt9","pt10","pt11","pt12","pt13","pt14","pt15","pt16","pt17","pt18","pt19",
                "isHS0","isHS1","isHS2","isHS3","isHS4","isHS5","isHS6","isHS7","isHS8","isHS9","isHS10","isHS11","isHS12","isHS13","isHS14","isHS15","isHS16","isHS17","isHS18","isHS19",
                ]

GeV = 1000  # scaling
nEventsPerSlice = 10000

# =========================================================================================================================================================================
# weight stuff
print("calculating weights")
slice_weights = []
for slice_name,xSec in zip(slice_list,xSec_list):
    nEvents = 0
    print("slice: "+slice_name)
    fFile = ROOT.TFile(slice_folder+slice_name, "READ")
    fTree = fFile.Get("tree")
    sum_w = 0
    for entry in fTree:
        sum_w += entry.weight
        nEvents += 1
        if nEvents >= nEventsPerSlice and nEventsPerSlice != -1:
            break
    slice_weights.append( xSec / sum_w )


# ========================================================================================================================
# fill event vector (with weights also)
print("creating events dataframe")
events = []
for slice_name,xSec,slice_weight in zip(slice_list,xSec_list,slice_weights):
    nEvents = 0
    print("slice: "+slice_name)
    fFile = ROOT.TFile(slice_folder+slice_name, "READ")
    fTree = fFile.Get("tree")
    for entry in fTree:
        event = [entry.s_or_b if entry.s_or_b > 0 else 0, slice_weight,
                entry.pt[0]/GeV,entry.pt[1]/GeV,entry.pt[2]/GeV,entry.pt[3]/GeV,entry.pt[4]/GeV,entry.pt[5]/GeV,entry.pt[6]/GeV,entry.pt[7]/GeV,entry.pt[8]/GeV,entry.pt[9]/GeV,entry.pt[10]/GeV,entry.pt[11]/GeV,entry.pt[12]/GeV,entry.pt[13]/GeV,entry.pt[14]/GeV,entry.pt[15]/GeV,entry.pt[16]/GeV,entry.pt[17]/GeV,entry.pt[18]/GeV,entry.pt[19]/GeV,
                entry.isHS[0],entry.isHS[1],entry.isHS[2],entry.isHS[3],entry.isHS[4],entry.isHS[5],entry.isHS[6],entry.isHS[7],entry.isHS[8],entry.isHS[9],entry.isHS[10],entry.isHS[11],entry.isHS[12],entry.isHS[13],entry.isHS[14],entry.isHS[15],entry.isHS[16],entry.isHS[17],entry.isHS[18],entry.isHS[19]
                ]
        events.append(event)
        nEvents += 1
        if nEvents >= nEventsPerSlice and nEventsPerSlice != -1:
            break
df = pd.DataFrame(events, columns = inputMoments)

# ORDENAR DATAFRAMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

minpt = 1
maxpt = 800
atLeast = 100 # at least these events to plot, we want to tran a NN so we need many events

Ns_vec = [4,5,6,10,20] # cannot be greater than 10
pts_vec = np.logspace(np.log10(minpt),np.log10(maxpt),100)

efficiency_vec = []
passed_vec = []
pt_cut_vec = []

for pt_cut in pts_vec:
    efficiency = []
    passed = []
    for N in Ns_vec:
        # for all events I look to the first N
        N_pass = 0
        N_ptcut = 0
        N_tot = 0
        print("pt cut: "+str(pt_cut)+ "\t N: "+str(N))
        for event in range(len(df)):
            matches = 0
            N_tot += 1
            if df.iloc[event,1+N] < pt_cut:
                continue
            N_ptcut += 1
            for jet in range(N):
                if df.iloc[event,22+jet] > -0.5 :# starts from isHS0
                    matches += 1
            if matches >= 4.0 :
                N_pass += 1
        if N_ptcut > atLeast:
            efficiency.append(float(float(N_pass) / float(N_ptcut) ))
            passed.append(float(N_pass))
        else:
            efficiency.append(np.nan)
            passed.append(float(N_pass))
    efficiency_vec.append(efficiency)
    passed_vec.append(passed)
    pt_cut_vec.append(pt_cut)

plt.figure(1)
plt.plot(pt_cut_vec,passed_vec,label=Ns_vec)
plt.title(str(len(df))+" total events considered (only selections with >"+str(atLeast)+" events)")
#plt.yscale('log')
plt.xscale('log')
plt.xlim(minpt,maxpt)
plt.ylabel("Signal Events (pass N & pt cuts)")
plt.xlabel("pt cut on jet N [GeV]")
plt.legend(title="N jets considered")
plt.grid()

plt.figure(2)
plt.plot(pt_cut_vec,efficiency_vec,label=Ns_vec)
plt.title(str(len(df))+" total events considered (only selections with >"+str(atLeast)+" events)")
#plt.yscale('log')
plt.xscale('log')
plt.ylim(0,1)
plt.xlim(minpt,maxpt)
plt.ylabel("Signal Efficiency (pass N & pt cuts / pass pt cut)")
plt.xlabel("pt cut on jet N [GeV]")
plt.legend(title="N jets considered")
plt.grid()
plt.show()

