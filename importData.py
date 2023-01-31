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

# testing commit
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
    #"user.lbazzano.gtp_output_ntuple_JZ8.root",
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
    #9.2954e-13 * 0.012908,
    ]

'''
slice_folder = "../calib_hh4b_samples/"
slice_list = ["user.lbazzano.gtp_output_ntuple_hh4b.root"]
xSec_list = [1]
'''

inputMoments = ["s_or_b","weight","mu","HT_pt30", "HT_n8",
                "pt0","pt1","pt2","pt3","pt4","pt5",
                "eta0","eta1","eta2","eta3","eta4","eta5",
                "phi0","phi1","phi2","phi3","phi4","phi5",
                "m0","m1","m2","m3","m4","m5",
                #"nConstituents0","nConstituents1","nConstituents2","nConstituents3","nConstituents4","nConstituents5",
                "pt0_uncalib","pt1_uncalib","pt2_uncalib","pt3_uncalib","pt4_uncalib","pt5_uncalib",
                "eta0_uncalib","eta1_uncalib","eta2_uncalib","eta3_uncalib","eta4_uncalib","eta5_uncalib",
                "isHS0","isHS1","isHS2","isHS3","isHS4","isHS5",
                "pt0_t","pt1_t","pt2_t","pt3_t","pt4_t","pt5_t",
                #"eta0_t","eta1_t","eta2_t","eta3_t","eta4_t","eta5_t",
                ]

GeV = 1000  # scaling
nEventsPerSlice = -1

# ========================================================================================================================
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
counter_1 = 0
counter_minpt = 0
counter_minptuncalib = 0
counter_minpttrue = 0
counter_Z = 0
for slice_name,xSec,slice_weight in zip(slice_list,xSec_list,slice_weights):
    nEvents = 0
    print("slice: "+slice_name)
    fFile = ROOT.TFile(slice_folder+slice_name, "READ")
    fTree = fFile.Get("tree")
    for entry in fTree:
        # if no truth jets in event, don't save
        #if int(entry.pt_t[0]) == -1:
        #    counter_1 += 1
        #    continue
        # don't save events of low pt (reco, uncalib and true)
        if abs(entry.eta[0]) > 4 or abs(entry.eta[1]) > 4  or abs(entry.eta[2]) > 4 or abs(entry.eta[3]) > 4  :# or entry.pt[1]/GeV < minPt1 or entry.pt[2]/GeV < minPt2 or entry.pt[3]/GeV < minPt3:
            continue
        if entry.pt[0]/GeV < 30:# or entry.pt[1]/GeV < minPt1 or entry.pt[2]/GeV < minPt2 or entry.pt[3]/GeV < minPt3:
            continue
        #    counter_minpt += 1
        #if no truth jets matched, just write -1 in truth properties
        aMatch=False
        for isM in entry.isMatch:
            if isM !=-1:
                aMatch = True
                continue
        pt_t_0 = -1
        pt_t_1 = -1
        pt_t_2 = -1
        pt_t_3 = -1
        pt_t_4 = -1
        pt_t_5 = -1
        eta_t_0 = -1
        eta_t_1 = -1
        eta_t_2 = -1
        eta_t_3 = -1
        eta_t_4 = -1
        eta_t_5 = -1
        if aMatch:
            p = 0
            for isM in entry.isMatch:
                if isM ==0:
                    pt_t_0 = entry.pt_t[p]/GeV
                    eta_t_0 = entry.eta_t[p]
                elif isM ==1:
                    pt_t_1 = entry.pt_t[p]/GeV
                    eta_t_1 = entry.eta_t[p]
                elif isM ==2:
                    pt_t_2 = entry.pt_t[p]/GeV
                    eta_t_2 = entry.eta_t[p]
                elif isM ==3:
                    pt_t_3 = entry.pt_t[p]/GeV
                    eta_t_3 = entry.eta_t[p]
                elif isM ==4:
                    pt_t_4 = entry.pt_t[p]/GeV
                    eta_t_4 = entry.eta_t[p]
                elif isM ==5:
                    pt_t_5 = entry.pt_t[p]/GeV
                    eta_t_5 = entry.eta_t[p]
                p += 1
        else:
            counter_Z += 1
        HT_pt30 = 0.0
        HT_n8 = 0.0
        print(range(len(entry.pt)))
        for jet_n in range(len(entry.pt)):
            jet_pt_n = entry.pt[jet_n]/GeV
            if jet_pt_n > 30:
                HT_pt30 = HT_pt30 + jet_pt_n
                print(HT_pt30)
            if jet_n < 8:
                HT_n8 = HT_n8 + jet_pt_n
                print(HT_n8)

        event = [entry.s_or_b if entry.s_or_b > 0 else 0, slice_weight, entry.mu, HT_pt30, HT_n8,
                entry.pt[0]/GeV,entry.pt[1]/GeV,entry.pt[2]/GeV,entry.pt[3]/GeV,entry.pt[4]/GeV,entry.pt[5]/GeV,
                entry.eta[0],entry.eta[1],entry.eta[2],entry.eta[3],entry.eta[4],entry.eta[5],
                entry.phi[0],entry.phi[1],entry.phi[2],entry.phi[3],entry.phi[4],entry.phi[5],
                entry.m[0]/GeV,entry.m[1]/GeV,entry.m[2]/GeV,entry.m[3]/GeV,entry.m[4]/GeV,entry.m[5]/GeV,
                entry.pt_uncalib[0]/GeV,entry.pt_uncalib[1]/GeV,entry.pt_uncalib[2]/GeV,entry.pt_uncalib[3]/GeV,entry.pt_uncalib[4]/GeV,entry.pt_uncalib[5]/GeV,
                entry.eta_uncalib[0],entry.eta_uncalib[1],entry.eta_uncalib[2],entry.eta_uncalib[3],entry.eta_uncalib[4],entry.eta_uncalib[5],
                entry.isHS[0],entry.isHS[1],entry.isHS[2],entry.isHS[3],entry.isHS[4],entry.isHS[5],
                pt_t_0, pt_t_1,pt_t_2, pt_t_3, pt_t_4, pt_t_5,
                #eta_t_0, eta_t_1, eta_t_2, eta_t_3, eta_t_4, eta_t_5,
                ]
        events.append(event) 
        nEvents += 1
        if nEvents >= nEventsPerSlice and nEventsPerSlice != -1:
            break
 
print(counter_Z)
df = pd.DataFrame(events, columns = inputMoments)
print(df)

# ========================================================================================================================

# save
#df.to_pickle("dataframes/raw_dataframe_012.pkl")
df.to_pickle("dataframes/raw_dataframe.pkl")
#df.to_pickle("dataframes/raw_dataframe_hh4b.pkl")
# load
df = pd.read_pickle("dataframes/raw_dataframe.pkl")

# ========================================================================================================================
# response plots
print("plotting response")

def plot_response(rp_pt0, rp_responses, Bins, rp_weights, xaxis, nameVar,nameResp="pt"):
    h, xe, ye, im = plt.hist2d(rp_pt0,rp_responses,bins = Bins, weights = rp_weights, norm = mpl.colors.LogNorm())#,range=[[min(rp_pt0),max(rp_pt0)],[0,2]]
    cbar=plt.colorbar()
    cbar.set_label("Weighted Entries")
    y_profile = []
    err_profile = []
    responses_x = []
    for bin_x_inf,bin_x_sup in zip(xe[:-1],xe[1:]):
        responses_bin = []
        responses_bin_w = []
        for p0,response,resp_w in zip(rp_pt0,rp_responses,rp_weights):
            if p0 < bin_x_sup and p0 > bin_x_inf:
                responses_bin.append(response)
                responses_bin_w.append(resp_w)
        if sum(responses_bin_w) == 0.:
            continue
        avg = np.average( responses_bin, weights = responses_bin_w )
        y_profile.append( avg )
        err = math.sqrt( np.average( ( responses_bin - avg )**2, weights = responses_bin_w ) ) 
        err_profile.append( err )
        responses_x.append((bin_x_inf+bin_x_sup)/2)
    plt.plot(responses_x, y_profile, color='k')
    plt.errorbar(responses_x, y_profile, err_profile,fmt='_', ecolor='k')
    if "pt" in nameVar:
        plt.xscale('log')
    plt.xlabel(xaxis)
    if "pt" in nameResp:
    plt.ylabel('pt Response (reco/true)')
if "eta" in nameResp:
        plt.ylabel('eta Bias (reco-true)')
    plt.savefig('response/response_vs_'+nameVar+'_'+nameResp+'Response.png')

rp_df =    df[df.pt0_t > 0]
rp_pt0 = list(rp_df['pt0_t'])
rp_eta0 = list(rp_df['eta0_t'])
rp_responses = list(rp_df['pt0']/rp_df['pt0_t'])
rp_responses_uncalib = list(rp_df['pt0_uncalib']/rp_df['pt0_t'])
rp_responses_eta = list(rp_df['eta0']-rp_df['eta0_t'])
rp_responses_eta_uncalib = list(rp_df['eta0_uncalib']-rp_df['eta0_t'])
rp_weights = list(rp_df['weight'])

y_space = np.linspace(0,2, 500)
x_space = np.logspace(np.log10(min(rp_pt0)), np.log10(max(rp_pt0)), 30)
Bins = (x_space, y_space)
plt.clf()
plot_response(rp_pt0, rp_responses, Bins, rp_weights, "leading jet true pt [GeV]","pt0")
plt.clf()
plot_response(rp_pt0, rp_responses_uncalib, Bins, rp_weights, "leading jet true pt [GeV]","pt0_uncalib")
plt.clf()

y_space = np.linspace(-0.2,0.2, 500)
Bins = (x_space, y_space)
plot_response(rp_pt0, rp_responses_eta, Bins, rp_weights, "leading jet true pt [GeV]","pt0","eta")
plt.clf()
plot_response(rp_pt0, rp_responses_eta_uncalib, Bins, rp_weights, "leading jet true pt [GeV]","pt0_uncalib","eta")
plt.clf()

y_space = np.linspace(0,2, 500)
x_space = np.linspace(min(rp_eta0), max(rp_eta0), 30)
Bins = (x_space, y_space)
plot_response(rp_eta0,rp_responses, Bins, rp_weights, "leading jet true eta [GeV]","eta0")
plt.clf()
plot_response(rp_eta0,rp_responses_uncalib, Bins, rp_weights, "leading jet true eta [GeV]","eta0_uncalib")
plt.clf()

y_space = np.linspace(-0.2,0.2, 500)
x_space = np.linspace(min(rp_eta0), max(rp_eta0), 30)
Bins = (x_space, y_space)
plot_response(rp_eta0,rp_responses_eta, Bins, rp_weights, "leading jet true eta [GeV]","eta0","eta")
plt.clf()
plot_response(rp_eta0,rp_responses_eta_uncalib, Bins, rp_weights, "leading jet true eta [GeV]","eta0_uncalib","eta")
plt.clf()
# ========================================================================================================================

# clean df
print("cleaning dataframe")
# remove events 

# reco #####################################################
# min values
min_reco_pt = 30
minPt0 = min_reco_pt 
minPt1 = min_reco_pt 
minPt2 = min_reco_pt 
minPt3 = min_reco_pt 
#minPt4 = min_reco_pt 
# calib
df = df[df.pt0 >= minPt0]
df = df[df.pt1 >= minPt1]
df = df[df.pt2 >= minPt2]
df = df[df.pt3 >= minPt3]
#df = df[df.pt4 >= minPt4]
# uncalib
df = df[df.pt0_uncalib >= minPt0]
df = df[df.pt1_uncalib >= minPt1]
df = df[df.pt2_uncalib >= minPt2]
df = df[df.pt3_uncalib >= minPt3]
#df = df[df.pt4_uncalib >= minPt4]
# max values
LUP = 400   # set lowest unprescale
maxPt0 = LUP
maxPt1 = LUP
maxPt2 = LUP
maxPt3 = LUP
#maxPt4 = LUP
# calib
df = df[df.pt0 <= maxPt0]
df = df[df.pt1 <= maxPt1]
df = df[df.pt2 <= maxPt2]
df = df[df.pt3 <= maxPt3]
#df = df[df.pt4 <= maxPt4]
# uncalib
df = df[df.pt0_uncalib <= maxPt0]
df = df[df.pt1_uncalib <= maxPt1]
df = df[df.pt2_uncalib <= maxPt2]
df = df[df.pt3_uncalib <= maxPt3]
#df = df[df.pt4_uncalib <= maxPt4]

# true #####################################################
#min_true_pt = 20
#minPt0_t = min_true_pt 
#minPt1_t = min_true_pt 
#minPt2_t = min_true_pt 
#minPt3_t = min_true_pt 
#minPt4_t = min_true_pt
#df = df[df.pt0_t >= minPt0_t]
#df = df[df.pt1_t >= minPt1_t]
#df = df[df.pt2_t >= minPt2_t]
#df = df[df.pt3_t >= minPt3_t]
#df = df[df.pt4_t >= minPt4_t]

#print(df)
# remove weird events with jets that have 0 < m <~ 1 GeV
#minM = 1
#df = df[df.m0 >= minM]
#df = df[df.m1 >= minM]
#df = df[df.m2 >= minM]
#df = df[df.m3 >= minM]
#df = df[df.m4 >= minM]

#print(df)
# remove events with any eta > 4
minEta = -4
df = df[df.eta0 >= minEta]
df = df[df.eta1 >= minEta]
df = df[df.eta2 >= minEta]
df = df[df.eta3 >= minEta]
df = df[df.eta4 >= minEta]
maxEta = 4
df = df[df.eta0 <= maxEta]
df = df[df.eta1 <= maxEta]
df = df[df.eta2 <= maxEta]
df = df[df.eta3 <= maxEta]
df = df[df.eta4 <= maxEta]

df = df[df.eta0_uncalib >= minEta]
df = df[df.eta1_uncalib >= minEta]
df = df[df.eta2_uncalib >= minEta]
df = df[df.eta3_uncalib >= minEta]
df = df[df.eta4_uncalib >= minEta]
df = df[df.eta0_uncalib <= maxEta]
df = df[df.eta1_uncalib <= maxEta]
df = df[df.eta2_uncalib <= maxEta]
df = df[df.eta3_uncalib <= maxEta]
df = df[df.eta4_uncalib <= maxEta]

print(df)
# ========================================================================================================================

# save
df.to_pickle("dataframes/clean_dataframe.pkl")
#df.to_pickle("dataframes/clean_dataframe_hh4b.pkl")
# load
df = pd.read_pickle("dataframes/clean_dataframe.pkl")

# ========================================================================================================================
# reorganize because calibration ruins everything
print("reorganizing events")
m = df.pt0 < df.pt1
df.loc[m, ['pt0', 'pt1']]  = ( df.loc[m, ['pt1',  'pt0']].values)
df.loc[m, ['eta0','eta1']] = ( df.loc[m, ['eta1', 'eta0']].values)
df.loc[m, ['phi0','phi1']] = ( df.loc[m, ['phi1', 'phi0']].values)
df.loc[m, ['m0',  'm1']]   = ( df.loc[m, ['m1',   'm0']].values)
df.loc[m, ['isHS0',  'isHS1']]   = ( df.loc[m, ['isHS1',   'isHS0']].values)
m = df.pt0 < df.pt2
df.loc[m, ['pt0', 'pt2']]  = ( df.loc[m, ['pt2',  'pt0']].values)
df.loc[m, ['eta0','eta2']] = ( df.loc[m, ['eta2', 'eta0']].values)
df.loc[m, ['phi0','phi2']] = ( df.loc[m, ['phi2', 'phi0']].values)
df.loc[m, ['m0',  'm2']]   = ( df.loc[m, ['m2',   'm0']].values)
df.loc[m, ['isHS0',  'isHS2']]   = ( df.loc[m, ['isHS2',   'isHS0']].values)
m = df.pt0 < df.pt3
df.loc[m, ['pt0', 'pt3']]  = ( df.loc[m, ['pt3',  'pt0']].values)
df.loc[m, ['eta0','eta3']] = ( df.loc[m, ['eta3', 'eta0']].values)
df.loc[m, ['phi0','phi3']] = ( df.loc[m, ['phi3', 'phi0']].values)
df.loc[m, ['m0',  'm3']]   = ( df.loc[m, ['m3',   'm0']].values)
df.loc[m, ['isHS0',  'isHS3']]   = ( df.loc[m, ['isHS3',   'isHS0']].values)
m = df.pt0 < df.pt4
df.loc[m, ['pt0', 'pt4']]  = ( df.loc[m, ['pt4',  'pt0']].values)
df.loc[m, ['eta0','eta4']] = ( df.loc[m, ['eta4', 'eta0']].values)
df.loc[m, ['phi0','phi4']] = ( df.loc[m, ['phi4', 'phi0']].values)
df.loc[m, ['m0',  'm4']]   = ( df.loc[m, ['m4',   'm0']].values)
df.loc[m, ['isHS0',  'isHS4']]   = ( df.loc[m, ['isHS4',   'isHS0']].values)
m = df.pt0 < df.pt5
df.loc[m, ['pt0', 'pt5']]  = ( df.loc[m, ['pt5',  'pt0']].values)
df.loc[m, ['eta0','eta5']] = ( df.loc[m, ['eta5', 'eta0']].values)
df.loc[m, ['phi0','phi5']] = ( df.loc[m, ['phi5', 'phi0']].values)
df.loc[m, ['m0',  'm5']]   = ( df.loc[m, ['m5',   'm0']].values)
df.loc[m, ['isHS0',  'isHS5']]   = ( df.loc[m, ['isHS5',   'isHS0']].values)

m = df.pt1 < df.pt2
df.loc[m, ['pt1', 'pt2']]  = ( df.loc[m, ['pt2', 'pt1']].values)
df.loc[m, ['eta1','eta2']] = ( df.loc[m, ['eta2', 'eta1']].values)
df.loc[m, ['phi1','phi2']] = ( df.loc[m, ['phi2', 'phi1']].values)
df.loc[m, ['m1',  'm2']]   = ( df.loc[m, ['m2',   'm1']].values)
df.loc[m, ['isHS1',  'isHS2']]   = ( df.loc[m, ['isHS2',   'isHS1']].values)
m = df.pt1 < df.pt3
df.loc[m, ['pt1', 'pt3']]  = ( df.loc[m, ['pt3',  'pt1']].values)
df.loc[m, ['eta1','eta3']] = ( df.loc[m, ['eta3', 'eta1']].values)
df.loc[m, ['phi1','phi3']] = ( df.loc[m, ['phi3', 'phi1']].values)
df.loc[m, ['m1',  'm3']]   = ( df.loc[m, ['m3',   'm1']].values)
df.loc[m, ['isHS1',  'isHS3']]   = ( df.loc[m, ['isHS2',   'isHS1']].values)
m = df.pt1 < df.pt4
df.loc[m, ['pt1', 'pt4']]  = ( df.loc[m, ['pt4',  'pt1']].values)
df.loc[m, ['eta1','eta4']] = ( df.loc[m, ['eta4', 'eta1']].values)
df.loc[m, ['phi1','phi4']] = ( df.loc[m, ['phi4', 'phi1']].values)
df.loc[m, ['m1',  'm4']]   = ( df.loc[m, ['m4',   'm1']].values)
df.loc[m, ['isHS1',  'isHS4']]   = ( df.loc[m, ['isHS4',   'isHS1']].values)
m = df.pt1 < df.pt5
df.loc[m, ['pt1', 'pt5']]  = ( df.loc[m, ['pt5',  'pt1']].values)
df.loc[m, ['eta1','eta5']] = ( df.loc[m, ['eta5', 'eta1']].values)
df.loc[m, ['phi1','phi5']] = ( df.loc[m, ['phi5', 'phi1']].values)
df.loc[m, ['m1',  'm5']]   = ( df.loc[m, ['m5',   'm1']].values)
df.loc[m, ['isHS1',  'isHS5']]   = ( df.loc[m, ['isHS5',   'isHS1']].values)

m = df.pt2 < df.pt3
df.loc[m, ['pt2', 'pt3']]  = ( df.loc[m, ['pt3',  'pt2']].values)
df.loc[m, ['eta2','eta3']] = ( df.loc[m, ['eta3', 'eta2']].values)
df.loc[m, ['phi2','phi3']] = ( df.loc[m, ['phi3', 'phi2']].values)
df.loc[m, ['m2',  'm3']]   = ( df.loc[m, ['m3',   'm2']].values)
df.loc[m, ['isHS2',  'isHS3']]   = ( df.loc[m, ['isHS3',   'isHS2']].values)
m = df.pt2 < df.pt4
df.loc[m, ['pt2', 'pt4']]  = ( df.loc[m, ['pt4',  'pt2']].values)
df.loc[m, ['eta2','eta4']] = ( df.loc[m, ['eta4', 'eta2']].values)
df.loc[m, ['phi2','phi4']] = ( df.loc[m, ['phi4', 'phi2']].values)
df.loc[m, ['m2',  'm4']]   = ( df.loc[m, ['m4',   'm2']].values)
df.loc[m, ['isHS2',  'isHS4']]   = ( df.loc[m, ['isHS4',   'isHS2']].values)
m = df.pt2 < df.pt5
df.loc[m, ['pt2', 'pt5']]  = ( df.loc[m, ['pt5',  'pt2']].values)
df.loc[m, ['eta2','eta5']] = ( df.loc[m, ['eta5', 'eta2']].values)
df.loc[m, ['phi2','phi5']] = ( df.loc[m, ['phi5', 'phi2']].values)
df.loc[m, ['m2',  'm5']]   = ( df.loc[m, ['m5',   'm2']].values)
df.loc[m, ['isHS2',  'isHS5']]   = ( df.loc[m, ['isHS5',   'isHS2']].values)

m = df.pt3 < df.pt4
df.loc[m, ['pt3', 'pt4']]  = ( df.loc[m, ['pt4',  'pt3']].values)
df.loc[m, ['eta3','eta4']] = ( df.loc[m, ['eta4', 'eta3']].values)
df.loc[m, ['phi3','phi4']] = ( df.loc[m, ['phi4', 'phi3']].values)
df.loc[m, ['m3',  'm4']]   = ( df.loc[m, ['m4',   'm3']].values)
df.loc[m, ['isHS3',  'isHS4']]   = ( df.loc[m, ['isHS4',   'isHS3']].values)
m = df.pt3 < df.pt5
df.loc[m, ['pt3', 'pt5']]  = ( df.loc[m, ['pt5',  'pt3']].values)
df.loc[m, ['eta3','eta5']] = ( df.loc[m, ['eta5', 'eta3']].values)
df.loc[m, ['phi3','phi5']] = ( df.loc[m, ['phi5', 'phi3']].values)
df.loc[m, ['m3',  'm5']]   = ( df.loc[m, ['m5',   'm3']].values)
df.loc[m, ['isHS3',  'isHS5']]   = ( df.loc[m, ['isHS5',   'isHS3']].values)

m = df.pt4 < df.pt5
df.loc[m, ['pt4', 'pt5']]  = ( df.loc[m, ['pt5',  'pt4']].values)
df.loc[m, ['eta4','eta5']] = ( df.loc[m, ['eta5', 'eta4']].values)
df.loc[m, ['phi4','phi5']] = ( df.loc[m, ['phi5', 'phi4']].values)
df.loc[m, ['m4',  'm5']]   = ( df.loc[m, ['m5',   'm4']].values)
df.loc[m, ['isHS4',  'isHS5']]   = ( df.loc[m, ['isHS5',   'isHS4']].values)
# ========================================================================================================================
# normalize weights ( mean(weights) = 1 )
Weight_norm = sum(df['weight'])/len(df['weight'])
df['weight'] = df['weight'].div(Weight_norm)

# ========================================================================================================================

# save
#df.to_pickle("dataframes/clean_ordered_dataframe_012.pkl")
df.to_pickle("dataframes/clean_ordered_dataframe.pkl")
#df.to_pickle("dataframes/clean_ordered_dataframe_hh4b.pkl")
# load
df = pd.read_pickle("dataframes/clean_ordered_dataframe.pkl")

# ========================================================================================================================
# variables plots
print("plotting variables")

def hist_var(df, df_s, df_b, s_or_b_weights, varName = "pt0", nBins = 100,type_var = ""):
    plt.clf()
    df[varName].plot(kind="hist",bins=nBins, weights=df["weight"],histtype='step',color='k',label=varName)
    df_s_or_b_pt0 = [df_s[varName],df_b[varName]]
    plt.hist(df_s_or_b_pt0,bins=nBins, weights=s_or_b_weights,histtype='stepfilled',stacked=False,alpha=0.5,label=['signal','background'])
    plt.yscale('log')
    units = " GeV" if type_var=="" and "pt" in varName or "m" in varName else ""
    plt.xlabel(type_var+" "+varName + units)
    plt.ylabel('Weighted Entries')
    #plt.ylim(1,1e6)
    plt.legend(framealpha=0.2,loc='best', fancybox=True)
    plt.grid()
    plt.savefig('dist/original/'+type_var+"/"+varName+'.png')
    plt.clf()
    #df[varName].plot(kind="hist",bins=nBins, weights=df["weight"],histtype='step',color='k',label=varName,density=True)
    #df_s_or_b_pt0 = [df_s[varName],df_b[varName]]
    plt.hist(df_s_or_b_pt0,bins=nBins, weights=s_or_b_weights,histtype='stepfilled',stacked=False,alpha=0.5,label=['signal','background'],density=True)
    if 'phi' not in varName:
        plt.yscale('log')
    plt.xlabel(type_var+" "+varName + units)
    plt.ylabel('Weighted Normalized Entries')
    #plt.ylim(1e-6,1e2)
    plt.legend(framealpha=0.2,loc='best', fancybox=True)
    plt.grid()
    #plt.show()
    plt.savefig('dist/normalized/'+type_var+"/"+varName+'.png')
    plt.clf()
    return

# s or b
df_s = df.drop(df[(df.s_or_b < 1)].index)
df_b = df.drop(df[(df.s_or_b > 0)].index)
s_or_b_weights= [df_s["weight"],df_b["weight"]]

for var_name in inputMoments[3:]:
    hist_var(df, df_s, df_b, s_or_b_weights, varName = var_name, type_var = "unscaled")

# ========================================================================================================================
# add already scaled variables to dataframe:
# ratio
print("calculating extra variables")
df['ratio50'] = df['pt5']/df['pt0']
df['ratio51'] = df['pt5']/df['pt1']
df['ratio52'] = df['pt5']/df['pt2']
df['ratio53'] = df['pt5']/df['pt3']
df['ratio54'] = df['pt5']/df['pt4']
df['ratio40'] = df['pt4']/df['pt0']
df['ratio41'] = df['pt4']/df['pt1']
df['ratio42'] = df['pt4']/df['pt2']
df['ratio43'] = df['pt4']/df['pt3']
df['ratio30'] = df['pt3']/df['pt0']
df['ratio31'] = df['pt3']/df['pt1']
df['ratio32'] = df['pt3']/df['pt2']
df['ratio20'] = df['pt2']/df['pt0']
df['ratio21'] = df['pt2']/df['pt1']
df['ratio10'] = df['pt1']/df['pt0']

def deltaPhi( p1, p2):
    return abs(abs(abs(p1-p2)-np.pi)-np.pi)

import tqdm

# dphi 0x yz (4 and 5)
dphi01_vec = []
dphi02_vec = []
dphi03_vec = []
dphi04_vec = []
dphi05_vec = []
dphi12_vec = []
dphi13_vec = []
dphi14_vec = []
dphi15_vec = []
dphi23_vec = []
dphi24_vec = []
dphi25_vec = []
dphi34_vec = []
dphi35_vec = []
dphi45_vec = []
dphi0X_vec = []
dphiYZ_vec = []
#dphi0X5_vec = []
#dphiYZ5_vec = []
print("calculating delta phis")
for event_n in tqdm.tqdm(range(len(df))):
    phi0 = df.iloc[event_n]['phi0']
    phi1 = df.iloc[event_n]['phi1']
    phi2 = df.iloc[event_n]['phi2']
    phi3 = df.iloc[event_n]['phi3']
    phi4 = df.iloc[event_n]['phi4']
    phi5 = df.iloc[event_n]['phi5']
    dphi01 = deltaPhi(phi0,phi1)
    dphi02 = deltaPhi(phi0,phi2)
    dphi03 = deltaPhi(phi0,phi3)
    dphi04 = deltaPhi(phi0,phi4)
    dphi05 = deltaPhi(phi0,phi5)
    dphi12 = deltaPhi(phi1,phi2)
    dphi13 = deltaPhi(phi1,phi3)
    dphi14 = deltaPhi(phi1,phi4)
    dphi15 = deltaPhi(phi1,phi5)
    dphi23 = deltaPhi(phi2,phi3)
    dphi24 = deltaPhi(phi2,phi4)
    dphi25 = deltaPhi(phi2,phi5)
    dphi34 = deltaPhi(phi3,phi4)
    dphi35 = deltaPhi(phi3,phi5)
    dphi45 = deltaPhi(phi4,phi5)
    dphi01_vec.append(dphi01)
    dphi02_vec.append(dphi02)
    dphi03_vec.append(dphi03)
    dphi04_vec.append(dphi04)
    dphi05_vec.append(dphi05)
    dphi12_vec.append(dphi12)
    dphi13_vec.append(dphi13)
    dphi14_vec.append(dphi14)
    dphi15_vec.append(dphi15)
    dphi23_vec.append(dphi23)
    dphi24_vec.append(dphi24)
    dphi25_vec.append(dphi25)
    dphi34_vec.append(dphi34)
    dphi35_vec.append(dphi35)
    dphi45_vec.append(dphi45)
    # dphi 0 123
    temp_dphi0X = [dphi01,dphi02,dphi03]
    max_ind = np.argmax(temp_dphi0X)
    dphi0X_vec.append(temp_dphi0X[max_ind])
    others = ['phi1','phi2','phi3']
    others.remove(others[max_ind])
    phiY = others[0]
    phiZ = others[1]
    dphiYZ_vec.append(deltaPhi(df.iloc[event_n][phiY],df.iloc[event_n][phiZ]))
    # dphi 0 1234
    #temp_dphi0X5 = [dphi01,dphi02,dphi03,dphi04]
    #max_ind = np.argmax(temp_dphi0X5)
    #dphi0X5_vec.append(temp_dphi0X5[max_ind])
    #others = ['phi1','phi2','phi3','phi4']
    #others.remove(others[max_ind])
    #temp_dphiYZ5 = [deltaPhi(df.iloc[event_n][others[0]],df.iloc[event_n][others[1]]), deltaPhi(df.iloc[event_n][others[0]],df.iloc[event_n][others[2]]), deltaPhi(df.iloc[event_n][others[1]],df.iloc[event_n][others[2]])]
    #max_ind = np.argmax(temp_dphiYZ5)
    #dphiYZ5_vec.append(temp_dphiYZ5[max_ind])


df['dphi01'] = dphi01_vec
df['dphi02'] = dphi02_vec
df['dphi03'] = dphi03_vec
df['dphi04'] = dphi04_vec
df['dphi12'] = dphi12_vec
df['dphi13'] = dphi13_vec
df['dphi14'] = dphi14_vec
df['dphi23'] = dphi23_vec
df['dphi24'] = dphi24_vec
df['dphi34'] = dphi34_vec
df['dphi0X'] = dphi0X_vec
df['dphiYZ'] = dphiYZ_vec
df['dphi0X5'] = dphi0X5_vec
#df['dphiYZ5'] = dphiYZ5_vec

# ========================================================================================================================
# scale df variables
print("scaling dataframe variables")

# pt
Pt_norm = LUP
df['pt0'] = df['pt0'].div(Pt_norm)
df['pt1'] = df['pt1'].div(Pt_norm)
df['pt2'] = df['pt2'].div(Pt_norm)
df['pt3'] = df['pt3'].div(Pt_norm)
df['pt4'] = df['pt4'].div(Pt_norm)

# m
M_norm = max([max( df['m0']), max(df['m1']),max(df['m2']),max(df['m3']),max(df['m4'])])
df['m0'] = df['m0'].div(M_norm)
df['m1'] = df['m1'].div(M_norm)
df['m2'] = df['m2'].div(M_norm)
df['m3'] = df['m3'].div(M_norm)
df['m4'] = df['m4'].div(M_norm)

# m
NC_norm = max([max( df['nConstituents0']), max(df['nConstituents1']),max(df['nConstituents2']),max(df['nConstituents3']),max(df['nConstituents4'])])
df['nConstituents0'] = df['nConstituents0'].div(NC_norm)
df['nConstituents1'] = df['nConstituents1'].div(NC_norm)
df['nConstituents2'] = df['nConstituents2'].div(NC_norm)
df['nConstituents3'] = df['nConstituents3'].div(NC_norm)
df['nConstituents4'] = df['nConstituents4'].div(NC_norm)

# phi
Phi_norm = 2*np.pi
df['phi0'] = df['phi0'].div(Phi_norm)+1/2
df['phi1'] = df['phi1'].div(Phi_norm)+1/2
df['phi2'] = df['phi2'].div(Phi_norm)+1/2
df['phi3'] = df['phi3'].div(Phi_norm)+1/2
df['phi4'] = df['phi4'].div(Phi_norm)+1/2

#dphis
df['dphi01'] = df['dphi01'].div(np.pi)
df['dphi02'] = df['dphi02'].div(np.pi) 
df['dphi03'] = df['dphi03'].div(np.pi) 
df['dphi04'] = df['dphi04'].div(np.pi) 
df['dphi0X'] = df['dphi0X'].div(np.pi) 
df['dphiYZ'] = df['dphiYZ'].div(np.pi) 
df['dphi0X5']=df['dphi0X5'].div(np.pi)
df['dphiYZ5']=df['dphiYZ5'].div(np.pi)

# eta
#Eta_norm = 2 * max([max(abs( df['eta0'])), max(abs(df['eta1'])),max(abs(df['eta2'])),max(abs(df['eta3'])),max(abs(df['eta4'])),max(abs( df['eta0_uncalib'])), max(abs(df['eta1_uncalib'])),max(abs(df['eta2_uncalib'])),max(abs(df['eta3_uncalib'])),max(abs(df['eta4_uncalib']))])
Eta_norm = 2 * 4
df['eta0'] = df['eta0'].div(Eta_norm)+1/2
df['eta1'] = df['eta1'].div(Eta_norm)+1/2
df['eta2'] = df['eta2'].div(Eta_norm)+1/2
df['eta3'] = df['eta3'].div(Eta_norm)+1/2
df['eta4'] = df['eta4'].div(Eta_norm)+1/2

# pt uncalib
df['pt0_uncalib'] = df['pt0_uncalib'].div(Pt_norm)
df['pt1_uncalib'] = df['pt1_uncalib'].div(Pt_norm)
df['pt2_uncalib'] = df['pt2_uncalib'].div(Pt_norm)
df['pt3_uncalib'] = df['pt3_uncalib'].div(Pt_norm)
df['pt4_uncalib'] = df['pt4_uncalib'].div(Pt_norm)

# eta uncalib
df['eta0_uncalib'] = df['eta0_uncalib'].div(Eta_norm)+1/2
df['eta1_uncalib'] = df['eta1_uncalib'].div(Eta_norm)+1/2
df['eta2_uncalib'] = df['eta2_uncalib'].div(Eta_norm)+1/2
df['eta3_uncalib'] = df['eta3_uncalib'].div(Eta_norm)+1/2
df['eta4_uncalib'] = df['eta4_uncalib'].div(Eta_norm)+1/2

# pt t
df['pt0_t'] = df['pt0_t'].div(Pt_norm)
# eta 
df['eta0_t'] = df['eta0_t'].div(Eta_norm)+1/2


print(df)

# ========================================================================================================================

# save
#df.to_pickle("dataframes/clean_ordered_scaled_dataframe_012.pkl")
df.to_pickle("dataframes/clean_ordered_scaled_dataframe.pkl")
#df.to_pickle("dataframes/clean_ordered_scaled_dataframe_hh4b.pkl")
# load
df = pd.read_pickle("dataframes/clean_ordered_scaled_dataframe.pkl")

# ========================================================================================================================
# scaled variables plots
print("plotting scaled variables")


inputMoments_NN_plots = ["pt0","pt1","pt2","pt3","pt4",
                "eta0","eta1","eta2","eta3","eta4",
                "phi0","phi1","phi2","phi3","phi4",
                "m0","m1","m2","m3","m4",
                "nConstituents0","nConstituents1","nConstituents2","nConstituents3","nConstituents4",
                "pt0_uncalib","pt1_uncalib","pt2_uncalib","pt3_uncalib","pt4_uncalib",
                "eta0_uncalib","eta1_uncalib","eta2_uncalib","eta3_uncalib","eta4_uncalib",
                ]
# s or b
df_s = df.drop(df[(df.s_or_b < 1)].index)
df_b = df.drop(df[(df.s_or_b > 0)].index)
s_or_b_weights= [df_s["weight"],df_b["weight"]]
for var_name in inputMoments_NN_plots:
    hist_var(df, df_s, df_b, s_or_b_weights, varName = var_name, type_var = "scaled")


inputMoments_NN_special_plots=['ratio40','ratio41','ratio42','ratio43','ratio30','ratio31','ratio32','ratio20','ratio21','ratio10','dphi0X','dphiYZ','dphi0X5','dphiYZ5','dphi01','dphi02','dphi03','dphi04','dphi12','dphi13','dphi14','dphi23','dphi24','dphi34']
for var_name in inputMoments_NN_special_plots:
    hist_var(df, df_s, df_b, s_or_b_weights, varName = var_name, type_var = "special")

