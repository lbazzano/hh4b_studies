import os
import uproot3 as uproot
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, plot_roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from numpy.random import seed
import math
#import ROOT

def getPoint(df,pt0_thresh,pt1_thresh,pt2_thresh,pt3_thresh,LUP,is_pt4_thresh=False,pt4_thresh=30):
    if not is_pt4_thresh:
        df_signal = df.loc[(df.pt0 >= pt0_thresh/LUP) & (df.pt1 >= pt1_thresh/LUP) & (df.pt2 >= pt2_thresh/LUP) & (df.pt3 >= pt3_thresh/LUP)]
        df_back   = df.loc[(df.pt0 < pt0_thresh/LUP) | (df.pt1 < pt1_thresh/LUP) | (df.pt2 < pt2_thresh/LUP) | (df.pt3 < pt3_thresh/LUP)]
    elif is_pt4_thresh:
        df_signal = df.loc[(df.pt0 >= pt0_thresh/LUP) & (df.pt1 >= pt1_thresh/LUP) & (df.pt2 >= pt2_thresh/LUP) & (df.pt3 >= pt3_thresh/LUP) & (df.pt4 >= pt4_thresh/LUP)]
        df_back   = df.loc[(df.pt0 < pt0_thresh/LUP) | (df.pt1 < pt1_thresh/LUP) | (df.pt2 < pt2_thresh/LUP) | (df.pt3 < pt3_thresh/LUP) | (df.pt4 < pt4_thresh/LUP)]
    true_positive      = len(df_signal[df_signal.s_or_b>0.5])
    true_negative      = len(df_back[df_back.s_or_b<0.5])
    false_positive     = len(df_signal[df_signal.s_or_b<0.5])
    false_negative     = len(df_back[df_back.s_or_b>0.5])
    print(true_positive,false_positive,true_negative,false_negative)
    false_positive_rate = false_positive / (true_negative+false_positive)
    true_positive_rate = true_positive / (false_negative + true_positive)
    return false_positive_rate, true_positive_rate

def getHistosPred(X_,y_,w_,model):
    y_predict = model.predict(X_)
    pred_signal = []
    pred_back = []
    pred_signal_w = []
    pred_back_w = []
    pred_ = []
    pred_w = []
    for y_i,y_predict_i,w in zip(y_,y_predict,w_):
        if y_i > 0.5:
            pred_signal.append(float(y_predict_i))
            pred_signal_w.append(float(w))
        elif y_i < 0.5:
            pred_back.append(float(y_predict_i))
            pred_back_w.append(float(w))
        pred_.append(float(y_predict_i))
        pred_w.append(float(w))
    return y_predict, pred_signal, pred_back, pred_signal_w, pred_back_w, pred_, pred_w

def trainAndPlot(df,df_QCD,vector_string,input_id,epochs,batchSize,layers,nodes,df_hh4b,uploadModel,s_or_b_column,s_or_b_ROC,evaluate_pTcut):
    # separate train and test sample
    X_train,X_test,y_train,y_test,w_train,w_test = train_test_split(
            df,               # input variables
            df[s_or_b_column],            # signal or backgound
            df.weight,            # weights
            test_size=0.2,
            shuffle= True,
            #random_state = 1
            )
    #
    pt0_train = X_train[["pt0"]]
    pt0_train.to_numpy()
    #
    X_train = X_train[vector_string]
    X_train.to_numpy()
    #
    pt0_test = X_test[["pt0"]]
    X_test = X_test[vector_string]
    X_test.to_numpy()
    #
    y_train.to_numpy()
    y_test.to_numpy()
    #
    w_train.to_numpy()
    w_test.to_numpy()

    if not uploadModel:
        # ========================================================================================================================
        # define the keras model
        model = Sequential()
        for layer in range(layers):
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(1, activation= 'sigmoid' ))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'], weighted_metrics = [])
        nEpochs      = epochs
        batchSize    = batchSize#256#512
        # ========================================================================================================================
        # fit the keras model on the dataset
        history = model.fit(X_train, y_train,
                  #validation_data  = [X_test,y_test],
                  sample_weight    = w_train,
                  epochs           = nEpochs,
                  batch_size       = batchSize,
                  validation_split = 0.2)
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']
        # ========================================================================================================================
        # evaluate the keras model train
        _, accuracy_total = model.evaluate(X_train, y_train)
        print('Accuracy: %.2f' % (accuracy_total*100))
        # ========================================================================================================================
        # evaluate the keras model test
        _, accuracy_total = model.evaluate(X_test, y_test)
        print('Accuracy: %.2f' % (accuracy_total*100))
        # ========================================================================================================================
        # plot loss and accuracy
        fig, [ax1, ax2] = plt.subplots(2, 1)
        ax1.plot(loss,color='green',label="train")
        ax1.plot(val_loss,color='red',label="validation")
        ax1.set(xlabel="", ylabel="Loss")
        ax1.legend(framealpha=0.2,loc='best')
        ax1.grid()
        #ax1.text(nEpochs, loss[-1],'loss', fontdict={'color':'green'})
        #ax1.text(nEpochs, val_loss[-1],'val_loss', fontdict={'color':'red'})
        ax2.plot(accuracy,color='green',label="train")
        ax2.plot(val_accuracy,color='red',label="validation")
        ax2.set(xlabel="#Epoch", ylabel="Accuracy")
        ax2.legend(framealpha=0.2,loc='best')
        ax2.grid()
        #ax2.text(nEpochs, accuracy[-1],'accuracy', fontdict={'color':'green'})
        #ax2.text(nEpochs, val_accuracy[-1],'val_accuracy', fontdict={'color':'red'})
        ax1.title.set_text(input_id)
        plt.savefig('loss_accuracy/loss_and_accuracy_weighted_'+input_id+'_NNsignal_'+s_or_b_column+'_ROCsignal_'+s_or_b_ROC+'.png')
        plt.close('all')
        # ========================================================================================================================
        # save keras model
        model.save("models/model_"+input_id+'_NNsignal_'+s_or_b_column+'_ROCsignal_'+s_or_b_ROC+".h5")
    elif uploadModel:
        # to upload:
        model = load_model("models/model_"+input_id+'_NNsignal_'+s_or_b_column+'_ROCsignal_'+s_or_b_ROC+".h5")
    # ========================================================================================================================
    # make class predictions
    if evaluate_pTcut > 0.0:
        boolist = pt0_train.pt0 < evaluate_pTcut
        X_train = X_train[boolist].to_numpy()
        y_train = y_train[boolist].to_numpy()
        w_train = w_train[boolist].to_numpy()
        boolist = pt0_test.pt0 < evaluate_pTcut
        X_test = X_test[boolist].to_numpy()
        y_test = y_test[boolist].to_numpy()
        w_test = w_test[boolist].to_numpy()
    y_predict_train, pred_signal_train, pred_back_train, pred_signal_w_train, pred_back_w_train, tot_train, tot_train_w = getHistosPred(X_train,y_train,w_train,model)
    y_predict_test,  pred_signal_test,  pred_back_test,  pred_signal_w_test,  pred_back_w_test,  tot_test,  tot_test_w  = getHistosPred(X_test,y_test,w_test,model)
    df_hh4b_columns = df_hh4b[vector_string].columns
    pt0_hh4b = df_hh4b.loc[:,["pt0"]]
    X_hh4b = df_hh4b.loc[:,df_hh4b_columns]
    y_hh4b = df_hh4b[s_or_b_column]
    w_hh4b = df_hh4b.weight
    pt0_hh4b.to_numpy()
    X_hh4b.to_numpy()
    y_hh4b.to_numpy()
    w_hh4b.to_numpy()
    if evaluate_pTcut > 0.0:
        boolist = pt0_hh4b.pt0 < evaluate_pTcut
        X_hh4b = X_hh4b[boolist].to_numpy()
        y_hh4b = y_hh4b[boolist].to_numpy()
        w_hh4b = w_hh4b[boolist].to_numpy()
    y_predict_hh4b, pred_signal_hh4b, pred_back_hh4b, pred_signal_w_hh4b, pred_back_w_hh4b, pred_hh4b, pred_hh4b_w  = getHistosPred(X_hh4b,y_hh4b,w_hh4b,model)
    # QCD
    y_QCD = df_QCD[s_or_b_ROC]
    df_QCD_columns = df_QCD[vector_string].columns
    pt0_QCD = df_QCD.loc[:,["pt0"]]
    X_QCD = df_QCD.loc[:,df_QCD_columns]
    w_QCD = df_QCD.weight
    X_QCD.to_numpy()
    y_QCD.to_numpy()
    w_QCD.to_numpy()
    if evaluate_pTcut > 0.0:
        boolist = pt0_QCD.pt0 < evaluate_pTcut
        X_QCD = X_QCD[boolist].to_numpy()
        y_QCD = y_QCD[boolist].to_numpy()
        w_QCD = w_QCD[boolist].to_numpy()
    y_predict_QCD, pred_signal_QCD, pred_back_QCD, pred_signal_w_QCD, pred_back_w_QCD, pred_QCD, pred_QCD_w  = getHistosPred(X_QCD,y_QCD,w_QCD,model)
    return X_train, X_test, y_train, y_predict_train, w_train, y_test, y_predict_test, w_test, pred_signal_train, pred_back_train, pred_signal_w_train, pred_back_w_train, pred_signal_test, pred_back_test, pred_signal_w_test, pred_back_w_test, X_hh4b, y_hh4b, y_predict_hh4b, w_hh4b, pred_signal_hh4b, pred_back_hh4b, pred_signal_w_hh4b, pred_back_w_hh4b, pred_hh4b, pred_hh4b_w,X_QCD, y_QCD,y_predict_QCD,w_QCD, pred_signal_QCD, pred_back_QCD, pred_signal_w_QCD, pred_back_w_QCD, pred_QCD, pred_QCD_w


# plot NN output dists
def plotNNdist_addHH4B(pred_signal_train, pred_back_train, pred_signal_w_train, pred_back_w_train, pred_signal_test, pred_back_test, pred_signal_w_test, pred_back_w_test,input_id,pred_signal_hh4b, pred_back_hh4b, pred_signal_w_hh4b, pred_back_w_hh4b,pred_hh4b, pred_hh4b_w,pred_signal_QCD, pred_back_QCD, pred_signal_w_QCD, pred_back_w_QCD, pred_QCD, pred_QCD_w,Density, s_or_b_column, s_or_b_ROC):
    plt.clf()
    nBins = np.linspace(0,1,30)
    plt.hist(pred_signal_train,alpha=0.7, density=Density, histtype='stepfilled', label='QCD NN signal train ('+str(len(pred_signal_train))+' entries)',     bins=nBins, weights=pred_signal_w_train)
    plt.hist(pred_back_train,  alpha=0.3, density=Density, histtype='stepfilled', label='QCD NN background train ('+str(len(pred_back_train))+' entries)', bins=nBins, weights=pred_back_w_train  )
    plt.hist(pred_signal_test, alpha=1, density=Density, histtype='step', label='QCD NN signal test ('+str(len(pred_signal_test))+' entries)',     bins=nBins, weights=pred_signal_w_test,color="blue",linestyle=':')
    plt.hist(pred_back_test,   alpha=1, density=Density, histtype='step', label='QCD NN background test ('+str(len(pred_back_test))+' entries)', bins=nBins, weights=pred_back_w_test ,color="orange" ,linestyle=':') 

    plt.yscale('log')
    plt.xlabel("NN output")
    if Density:
        plt.ylabel('Weighted Normalized Entries')
    else:
        plt.ylabel('Weighted Entries (not normalized)')
    plt.title('NNinput: '+input_id+' NNsignal: '+s_or_b_column+' ROCsignal:'+s_or_b_ROC)
    plt.grid()
    plt.ylim((0.0005,11))
    #plt.legend(framealpha=0.5,loc='lower center',prop={'size': 8})
    #plt.savefig('dist/NNoutput_normalized_'+input_id+'_NNsignal_'+s_or_b_column+'_ROCsignal_'+s_or_b_ROC+'_1.png')
    #plt.legend('', frameon=False)
    plt.hist(pred_signal_QCD,  alpha=1,   density=Density, histtype='step', label='QCD truth signal ('+str(len(pred_signal_QCD))+' entries)', bins=nBins, weights=pred_signal_w_QCD,  color = 'tab:green' ,linewidth=2)
    plt.hist(pred_back_QCD,    alpha=1,   density=Density, histtype='step', label='QCD truth background ('+str(len(pred_back_QCD))+' entries)', bins=nBins, weights=pred_back_w_QCD,  color = 'tab:red' ,linewidth=2)
    #plt.legend(framealpha=0.5,loc='lower center',prop={'size': 8})
    #plt.savefig('dist/NNoutput_normalized_'+input_id+'_NNsignal_'+s_or_b_column+'_ROCsignal_'+s_or_b_ROC+'_2.png')
    #plt.legend('', frameon=False)
    plt.hist(pred_hh4b,        alpha=1,   density=Density, histtype='step', label='HH4b ('+str(len(pred_hh4b))+' entries)', bins=nBins, weights=pred_hh4b_w,  color = 'black' ,linewidth=2)
    plt.legend(framealpha=0.5,loc='lower center',prop={'size': 8})
    #plt.hist(pred_signal_hh4b, alpha=1,   density=Density, histtype='step', label='HH4b signal',     bins=nBins, weights=pred_signal_w_hh4b,color = 'black',linestyle='dashed')
    #plt.hist(pred_back_hh4b,   alpha=1,   density=Density, histtype='step', label='HH4b background', bins=nBins, weights=pred_back_w_hh4b,  color = 'gray' ,linestyle='dashed')
    if Density:
        plt.savefig('dist/NNoutput_normalized_'+input_id+'_NNsignal_'+s_or_b_column+'_ROCsignal_'+s_or_b_ROC+'.png')
    else:
        plt.savefig('dist/NNoutput_not_normalized_'+input_id+'_NNsignal_'+s_or_b_column+'_ROCsignal_'+s_or_b_ROC+'.png')
    plt.clf()
    plt.close('all')

# get hh4b pass fraction
def get_pass_fraction(pred_hh4b,pred_hh4b_w,threshold):
    sorter = np.argsort(pred_hh4b,axis=0)
    values = pred_hh4b[sorter]
    sample_weight = pred_hh4b_w[sorter]
    pass_sum=0
    total_sum=0
    for w,v in zip(sample_weight,values):
        if v > threshold:
            pass_sum += w
        total_sum += w
    return pass_sum/total_sum


# plot NN output dists
def plotNNdist_addHH4B_ratios(pred_signal_train_vec, pred_back_train_vec, pred_signal_w_train_vec, pred_back_w_train_vec, pred_signal_test_vec, pred_back_test_vec, pred_signal_w_test_vec, pred_back_w_test_vec,input_id_vec,pred_signal_hh4b_vec, pred_back_hh4b_vec, pred_signal_w_hh4b_vec, pred_back_w_hh4b_vec,pred_hh4b_vec, pred_hh4b_w_vec,Density,plotHH4b,extraTag=""):
    for pred_signal_train,pred_back_train,pred_signal_test,pred_back_test,pred_signal_w_train,pred_back_w_train,pred_signal_w_test,pred_back_w_test,input_id,pred_signal_hh4b,pred_back_hh4b,pred_signal_w_hh4b,pred_back_w_hh4b,pred_hh4b, pred_hh4b_w in zip(pred_signal_train_vec, pred_back_train_vec,pred_signal_test_vec, pred_back_test_vec,pred_signal_w_train_vec, pred_back_w_train_vec,pred_signal_w_test_vec, pred_back_w_test_vec,input_id_vec,pred_signal_hh4b_vec, pred_back_hh4b_vec, pred_signal_w_hh4b_vec, pred_back_w_hh4b_vec,pred_hh4b_vec, pred_hh4b_w_vec):
        plt.clf()
        nBins = np.linspace(0,1,30)
        signal_pt0_h ,  bins = np.histogram(pred_signal_train, density=Density,bins=nBins, weights=pred_signal_w_train)
        back_pt0_h ,    bins = np.histogram(pred_back_train,   density=Density,bins=nBins, weights=pred_back_w_train)
        hh4b_pt0_h ,    bins = np.histogram(pred_hh4b,         density=Density,bins=nBins, weights=pred_hh4b_w)

        ratio_signal = signal_pt0_h/hh4b_pt0_h
        ratio_back   = back_pt0_h/hh4b_pt0_h
        ratio_hh4b   = hh4b_pt0_h/hh4b_pt0_h
        # plot
        plt.plot(bins[:-1],ratio_signal,color="blue",label="qcd signal")
        plt.plot(bins[:-1],ratio_back,color="orange",label="qcd background")
        plt.plot(bins[:-1],ratio_hh4b,color="black",label="hh4b")
        #plt.yscale('log')
        #plt.xlim([b_min,b_max])
        #plt.title("sample type: "+sample+"  |  var: "+var+"  |  signal: "+sig)
        #plt.subplot(212)
        #plt.legend()
        #plt.xlim([b_min,b_max])
        plt.ylim([0,2])
        #plt.show()

        #plt.yscale('log')
        plt.xlabel("NN output")
        if Density:
            plt.ylabel('Weighted Normalized Entries')
        else:
            plt.ylabel('Weighted Entries (not normalized)')
        plt.title(input_id)
        plt.legend(framealpha=0.1,loc='upper left')
        plt.grid()
        if Density:
            plt.savefig('dist/ratios_NNoutput_normalized_'+input_id+extraTag+'.png')
        else:
            plt.savefig('dist/ratios_NNoutput_not_normalized_'+input_id+extraTag+'.png')
        plt.clf()
        plt.close('all')

# roc curves
def plot_roc(fig,ax1,ax2,y_true, y_score, y_weight,threshold,acceptance,frac_hh4b_pass, y_hh4b, estimatorName='train',lw=1,color="black",linestyle='solid',alpha_i=1,fancy_plot = False,do_zoom=False,plotError=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,sample_weight=y_weight)
    idx = (np.abs(thresholds - threshold)).argmin()
    idx_fixAcc = (np.abs(tpr - acceptance)).argmin()
    print(1/fpr[idx_fixAcc])
    threshold_fixAcc = thresholds[idx_fixAcc]
    idx_best = np.sqrt( (1-tpr)*(1-tpr) + (fpr)*(fpr) ).argmin()
    threshold_best = thresholds[idx_best]
    frac_hh4b_pass_best = float(len(y_hh4b [ y_hh4b > threshold_best ] )) / float(len(y_hh4b))
    frac_hh4b_pass_fixAcc = float(len(y_hh4b [ y_hh4b > threshold_fixAcc ] )) / float(len(y_hh4b))
    if fancy_plot == False:
        ax1.plot(tpr, fpr, label = estimatorName,lw=lw,alpha=alpha_i,linestyle=linestyle,color=color)
        #ax1.scatter(tpr[idx], fpr[idx], label = "fixed threshold: %s  pass: %.2f" % (threshold, round(frac_hh4b_pass[0], 2)) ,color=color)
        ax1.scatter(tpr[idx_fixAcc], fpr[idx_fixAcc], label = str(acceptance)+" acceptance threshold: %s hh4b pass: %.2f" % (round(threshold_fixAcc,2), round(frac_hh4b_pass_fixAcc, 2)) ,color=color, marker='s')
        ax1.scatter(tpr[idx_best], fpr[idx_best], label = "optimal threshold: %s hh4b pass: %.2f" % (round(thresholds[idx_best],2), round(frac_hh4b_pass_best, 2)) ,color=color, facecolors='none',s=50)

        ax1.set_xlabel('True Positive Rate (signal efficiency)')
        ax1.set_ylabel('False Positive Rate (1-background rejection)')
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax2.set_xlim([0.0, 1.0])

        if do_zoom:
            ax1.set_xlim([0.7, 1.0])
            ax1.set_ylim([0.0, 0.4])
            ax2.set_xlim([0.7, 1.0])
    else:
        tpr_noZeros = []
        fpr_noZeros = []
        thresholds_noZeros = []
        for tpr_i,fpr_i,threshold_i in zip(tpr,fpr,thresholds):
            if fpr_i > 0:
                tpr_noZeros.append(tpr_i)
                fpr_noZeros.append(fpr_i)
        div = [1/y for y in fpr_noZeros]
        ax1.plot(tpr_noZeros, div , label = estimatorName,lw=lw,alpha=alpha_i,linestyle=linestyle,color=color)
        if plotError:
            div_up   = [i+i*0.03 for i in div]
            div_down = [i-i*0.03 for i in div]
            ax1.fill_between(tpr_noZeros,div_down,div_up, alpha=0.1,color = color)
        #ax1.scatter(tpr_noZeros[idx], div[idx], label = "fixed threshold: %s  pass: %.2f" % (threshold, round(frac_hh4b_pass[0], 2)) ,color=color)
        ax1.scatter(tpr_noZeros[idx_fixAcc], div[idx_fixAcc], label = str(acceptance)+" acceptance threshold: %s  hh4b pass: %.2f" % (round(threshold_fixAcc,2), round(frac_hh4b_pass_fixAcc, 2)) ,color=color,marker='s')
        ax1.scatter(tpr_noZeros[idx_best], div[idx_best], label = "optimal threshold: %s  hh4b pass: %.2f" % (round(thresholds[idx_best],2), round(frac_hh4b_pass_best, 2)) ,color=color, facecolors='none',s=100)
        ax1.set_xlabel('Signal Efficiency')
        ax1.set_ylabel('1 / Background Efficiency')
        
        ax1.set_xlim([0.0, 1.0])
        #ax1.set_ylim([1., 10000.])
        ax1.set_yscale('log')
        ax2.set_xlim([0.0, 1.0])
        #ax2.set_xlim([0.2, 1.0])
        if do_zoom:
            ax1.set_xlim([0.7, 1.0])
            ax1.set_ylim([1., 20.])
            ax2.set_xlim([0.7, 1.0])
            
            ax1.set_xlim([0.85, 0.95])
            ax1.set_ylim([2., 3])
            ax2.set_xlim([0.85, 0.95])
    #ax2.plot(tpr,  thresholds , lw=0.8,alpha=0.5,color=color)
    #ax2.set_ylabel('NN output threshold',alpha=0.5)
    #ax2.set_ylim([0.0, 1.0])
    return thresholds[idx_best],thresholds[idx_fixAcc]

