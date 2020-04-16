# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:28:53 2020
A package of VJ fucntions.
@author: Stuber_Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io as sio
import os
import subprocess
import bisect
import errno
import time
import pandas
import pickle
import num2word
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
import scipy.stats as stats
from sklearn.metrics import roc_auc_score as auROC
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import (ModelDesc, EvalEnvironment, Term, EvalFactor, LookupFactor, dmatrices, INTERCEPT)
from statsmodels.distributions.empirical_distribution import ECDF
from shapely.geometry import MultiPolygon, Polygon, Point
import PIL
from itertools import product
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from sima.ROI import poly2mask, _reformat_polygons, ROIList
import h5py
import sima
import sys
import seaborn as sns

def ismembertol(x, y, tol=1E-6):
    # Are elements of x in y within tolerance of tol?
    # x and y must be 1d numpy arrays
    sortx = np.sort(x)
    orderofx = np.argsort(x)
    sorty = np.sort(y)
    current_y_idx = 0
    result = np.nan*np.zeros(x.shape)
    for i, elt in enumerate(sortx):
        temp = sorty[current_y_idx:]
        if np.any(np.abs(temp-elt)<=tol):
            result[orderofx[i]]=1
        else:
            result[orderofx[i]]=0
        temp = np.argwhere(sorty>elt)
        if temp.size>0:
            current_y_idx = temp[0][0]
    return result

def firstlick_after_event(events, licks):
    result = np.nan*np.ones(events.shape)
    for i in range(events.shape[0]):    
        temp = bisect.bisect(licks,events[i])
        if temp<licks.shape[0]:
            result[i] = licks[temp]
    return result

def fix_any_dropped_frames(frames):
    dropped_frames = []
    diff_frames = np.diff(frames)
    inter_frame_interval = np.amin(diff_frames)
    frame_drop_idx = np.where(diff_frames>1.5*inter_frame_interval)[0]
    for idx in frame_drop_idx:
        numframesdropped = int(np.round((frames[idx+1]-frames[idx])/(inter_frame_interval+0.0))-1)
        temp = [frames[idx]+a*inter_frame_interval for a in range(1,numframesdropped+1)]
        dropped_frames.extend(temp)
    corrected_frames = np.sort(np.concatenate((frames, np.array(dropped_frames))))
    return corrected_frames


def calculate_lick_bout_onsets(licks, min_ili_forbout=500):
    # http://jn.physiology.org/content/jn/95/1/119.full.pdf is the reference for 500ms
    lick_bout_onsets = np.insert(licks[np.where(np.diff(licks)>min_ili_forbout)[0]+1], 0, licks[0])    
    return lick_bout_onsets

def fit_regression(x, y):
    lm = sm.OLS(y, sm.add_constant(x)).fit()
    x_range = sm.add_constant(np.array([x.min(), x.max()]))
    x_range_pred = lm.predict(x_range)
    return lm.pvalues[1], lm.params[1], x_range[:,1], x_range_pred, lm.rsquared

def CDFplot(x, ax, color=None, label='', linetype='-'):
    x = np.array(x)
    ix=np.argsort(x)
    ax.plot(x[ix], ECDF(x)(x)[ix], linetype, color=color, label=label)
    return ax

def fit_regression_and_plot(x, y, ax, plot_label='', color='k', linecolor='r', markersize=10):
    #linetype is a string like 'bo'
    pvalue, slope, temp, temppred, R2 = fit_regression(x, y)    
    ax.scatter(x, y, color=color, label='%s p=%.3e\nR$^2$=%.3f'% (plot_label, pvalue, R2), s=markersize)
    ax.plot(temp, temppred, color=linecolor)
    return ax, slope, pvalue, R2

def align_traces_around_cues(signalsT, framenumberforcues_dict, frame_of_next_cue_dict, t_fxd, framerate):
    # window_size: How many frames do you want to plot around the origin?
    
    if len(signalsT.shape)==1:
        signalsT = np.expand_dims(signalsT, axis=1)
    
    numrois = signalsT.shape[1]
    numframesforcue = int(t_fxd*framerate*1E-3)
    pre_window_size = numframesforcue  # How many frames per trial before origin to be plotted?
#     post_window_size = window_size-pre_window_size
    post_window_size = 2*numframesforcue + 70
    window_size = pre_window_size + post_window_size
    align = {}
    align_baselinesubtracted = {}

    for key in framenumberforcues_dict.keys():
        align[key] = np.NAN*np.zeros([framenumberforcues_dict[key].shape[0],
                                     window_size,
                                     numrois])
        align_baselinesubtracted[key] = np.NAN*np.zeros([framenumberforcues_dict[key].shape[0],
                                                         window_size,
                                                         numrois])
        tempbaseline = np.NAN*np.zeros([framenumberforcues_dict[key].shape[0], numrois])
        for f in range(framenumberforcues_dict[key].shape[0]):
            tempframe = int(framenumberforcues_dict[key][f])
            tempbaseline[f,:] = np.nanmean(signalsT[tempframe-pre_window_size:tempframe, :], axis=0)
            if tempframe + post_window_size >= signalsT.shape[0]:
                tempend = signalsT.shape[0]
            elif post_window_size >= frame_of_next_cue_dict[key][f]:
                tempend = tempframe + int(frame_of_next_cue_dict[key][f])
            else:
                tempend = tempframe + post_window_size
            align[key][f,:(tempend-(tempframe-pre_window_size)),:] = signalsT[tempframe-pre_window_size:tempend, :]
        align_baselinesubtracted[key] = align[key] - np.nanmean(tempbaseline, axis=0)
        align[key] = np.squeeze(align[key])
        align_baselinesubtracted[key] = np.squeeze(align_baselinesubtracted[key])
    return align, align_baselinesubtracted, pre_window_size, window_size

def plot_rasters_PSTH(rois_of_interest, keys_withtrials, align_baselinesubtracted,
                  window_size, pre_window_size, t_fxd, framerate, trials_of_interest_dict,
                  colors_for_key, indir, ylabel, savedir, numframesforcue,rasters=True, PSTH=True, sortby=None):
    
#     if len(align_baselinesubtracted[keys_withtrials[0]].shape)<3:
    if rois_of_interest is None:
        cmap='gray_r'
        rois_of_interest = [0]
    else:
        cmap = 'coolwarm'
    for roi in rois_of_interest:
        cmin = np.nan*np.ones((len(keys_withtrials),))
        cmax = np.nan*np.ones((len(keys_withtrials),))
        if rasters and PSTH:
            fig, axs = plt.subplots(len(keys_withtrials)+1, figsize=(5,3*(len(keys_withtrials)+1)))
        elif PSTH:
            fig, axs = plt.subplots(1, figsize=(5,3))
        for k, key in enumerate(keys_withtrials):
            if len(align_baselinesubtracted[key].shape)<3:
                align_baselinesubtracted[key] = np.expand_dims(align_baselinesubtracted[key], axis=2)
                
            if rasters:
                if sortby is None:
                    sns.heatmap(align_baselinesubtracted[key][trials_of_interest_dict[key],:,roi], ax=axs[k], linewidth=0,
                                cmap=plt.get_cmap(cmap))
                else:
                    sns.heatmap(align_baselinesubtracted[key][sortby[key],:,roi], ax=axs[k], linewidth=0,
                                cmap=plt.get_cmap(cmap))
                axs[k].set_title('CS%s trials'%key)
                axs[k].set_ylabel('Trial number')
                axs[k].set_xlabel('Time from cue (s)')
                axs[k].set_xticks(range(0, window_size, window_size/10))
                axs[k].set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size, window_size/10)])
                axs[k].set_yticks(range(0, trials_of_interest_dict[key].shape[0], 10))
                axs[k].set_yticklabels([str(a) for a in range(trials_of_interest_dict[key].shape[0], 0, -10)])
                axs[k].plot([pre_window_size, pre_window_size],
                            [0, trials_of_interest_dict[key].shape[0]], '--k', linewidth=1)
                axs[k].plot([pre_window_size+t_fxd*framerate*1E-3, pre_window_size+t_fxd*framerate*1E-3],
                            [0, trials_of_interest_dict[key].shape[0]], '--k', linewidth=1)
            if PSTH:
                if not rasters:
                    ax = axs
                else:
                    ax = axs[-1]
                sns.tsplot(align_baselinesubtracted[key][trials_of_interest_dict[key],:,roi],
                           ax=ax, estimator=np.nanmean, color=colors_for_key[key], condition=key)
            temp = np.nanmean(align_baselinesubtracted[key][trials_of_interest_dict[key],:,roi], axis=0)
            cmin[k] = np.amin(temp[np.isfinite(temp)])
            cmax[k] = np.amax(temp[np.isfinite(temp)])
        if PSTH:
            if not rasters:
                ax = axs
            else:
                ax = axs[-1]
            ax.set_xticks(range(0, window_size, window_size/10))
            ax.set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size, window_size/10)])

            ax.plot([pre_window_size, pre_window_size], 
                     [np.amin(cmin)-0.1,
                      np.amax(cmax)+0.1],
                     '--k', linewidth=1)  
            ax.plot([pre_window_size+numframesforcue, pre_window_size+numframesforcue],
                     [np.amin(cmin)-0.1,
                      np.amax(cmax)+0.1],
                     '--k', linewidth=1)
            ax.set_xlabel('Time from cue (s)')
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right') 
        fig.tight_layout()
        if len(rois_of_interest)==1:
            fig.savefig(os.path.join(savedir, 'PSTH_%s.png'%(ylabel.replace('\n', ' '))), format='png', dpi=300)
        else:#
            fig.savefig(os.path.join(savedir, 'ROI%d_PSTH_%s.png'%(roi+1, ylabel.replace('\n', ' '))), format='png', dpi=300)
            fig.clf()

def calculate_centraltendency_for_rois(signal, baseline_frame, centraltendency='auROC'):
    #signal has shape numtrials x numtimepoints x numrois
    #auROCmat has shape numrois x numtimepoints
    
    if centraltendency=='auROC':
        baseline = signal[:,baseline_frame,:]
        signal = signal[np.isfinite(baseline[:,0]),:,:]
        baseline = baseline[np.isfinite(baseline[:,0]),:]
        (numtrials, numtimepoints, numrois) = signal.shape
#         print baseline.shape
        #print np.mean(baseline)
        auROCmat = np.nan*np.ones((numrois, numtimepoints))
        for roi in range(numrois):
            #print roi
            for t in range(numtimepoints):
                temp = signal[:,t,roi]#; temp = temp[np.isfinite(temp)]
#                 print temp.shape, roi, t
#                 print temp
                data = np.concatenate((baseline[np.isfinite(temp),roi], temp[np.isfinite(temp)]))
                labels = np.concatenate((np.zeros((np.sum(np.isfinite(temp)),)), np.ones((np.sum(np.isfinite(temp)),))))
                auROCmat[roi,t] = 2*auROC(labels, data)-1
            #print roi, np.mean(auROCmat[roi,:])
        #print np.mean(auROCmat[:,baseline_epoch[0]:baseline_epoch[1]])
        #raise Exception()
        return auROCmat
    elif centraltendency=='baseline subtracted mean':
        return np.nanmean(signal, axis=0).T
    
def align_around_event(signals, framenumberforevent, frames, window_size, pre_window_size):
    numrois = signals.shape[0]
    post_window_size = window_size-pre_window_size

    numtrials = framenumberforevent.shape[0]

    align = np.NAN*np.zeros([numtrials,window_size,numrois])
    align_to_plot = np.NAN*np.zeros([numtrials,window_size,numrois])

    temp = signals.T
    prevendindex = 0
    tempbaseline = np.NAN*np.zeros([numtrials, numrois])
    for i in range(numtrials):
        tempindex = framenumberforevent[i]
        if np.isfinite(tempindex):
            tempindex = int(tempindex)
            tempstartindex = np.amin([pre_window_size, tempindex]).astype(int)
            startindex = np.amin([tempstartindex, tempindex-prevendindex]).astype(int)
            tempendindex = np.amin([len(frames)-tempindex, post_window_size])
            if i<(numtrials-1) and np.isfinite(framenumberforevent[i+1]):
                endindex = np.amin([framenumberforevent[i+1]-tempindex, tempendindex]).astype(int)
            else:
                endindex = tempendindex.astype(int)
            prevendindex = tempindex+endindex
            #print tempindex, temp.shape
            #print tempindex-startindex, tempindex+endindex
            align_to_plot[i,pre_window_size-startindex:pre_window_size+endindex,:] = temp[tempindex-startindex:tempindex+endindex,:]
            align[i,pre_window_size-tempstartindex:pre_window_size+endindex,:] = temp[tempindex-tempstartindex:tempindex+endindex,:]
            tempbaseline[i,:] = np.nanmean(temp[tempindex-startindex:tempindex, :], axis=0)
    align_to_plot = align_to_plot[np.where(np.isfinite(align_to_plot[:,0,0]))[0],:,:]
    align_baselinesubtracted = align_to_plot - np.nanmean(tempbaseline, axis=0)
            
    return align, align_to_plot, align_baselinesubtracted

def plot_average_PSTH_around_event(signals, framenumberforevent, framerate, frames, savedir,
                                   window_size=30, pre_window_size=10, trialsofinterest=None,
                                   sortby='response', eventname='first lick after unpredicted reward',
                                   centraltendency='baseline subtracted mean'):
    numrois = signals.shape[0]
    _,_,align_baselinesubtracted = align_around_event(signals, framenumberforevent, frames, window_size, pre_window_size)
    
    #populationdata = np.nanmean(align_baselinesubtracted[trialsofinterest,:,:], axis=0).T-1
    populationdata = calculate_centraltendency_for_rois(align_baselinesubtracted,
                                                        pre_window_size-1)
    print populationdata.shape
    #raise Exception()
    
    temp = {}
    temp[eventname] = align_baselinesubtracted
    with open(os.path.join(indir, 'Alignedtotrial_%s.pickle'%(eventname)), 'wb') as f:
        pickle.dump(temp, f)

    if sortby == 'response':
        tempresponse = np.nanmean(populationdata[:,pre_window_size:], axis=1)
        #temp=np.divide(1, (np.arange(1,post_window_size)+0.0)**0.5)
        #tempresponse = np.sum(populationdata[:,pre_window_size+1:]*np.tile(np.expand_dims(temp,axis=0), (populationdata.shape[0],1)),
        #                     axis=1)
        sortresponse = np.argsort(tempresponse)[::-1]
    elif sortby =='':
        sortresponse = np.arange(populationdata.shape[0])[::-1]

    fig, axs = plt.subplots(2, figsize=(5, 2*5))
    cmin = np.amin(populationdata)
    cmax = np.amax(populationdata)
    cax = sns.heatmap(populationdata[sortresponse,:],
                    ax=axs[0],
                    cmap=plt.get_cmap('coolwarm'),
                    vmin=-cmax,
                    vmax=cmax)

    axs[0].grid(False)
    axs[0].set_title('Response to %s of all ROIs'%(eventname))
    axs[0].set_ylabel('Sorted ROI number')
    axs[0].set_xlabel('Time from %s (s)'%(eventname))
    axs[0].set_xticks(range(0, window_size+1, 8))
    axs[0].set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size+1, 8)])
    axs[0].set_yticks(range(0, numrois, numrois/5))
    axs[0].set_yticklabels([str(a+1) for a in range(0, numrois, numrois/5)])

    axs[0].axvline(pre_window_size, color='k', linestyle='--')

    cbar = cax.collections[0].colorbar
    #cbar.set_ticks([-0.2, 0, 0.2])
    #cbar.set_ticklabels(['-0.2', '0', '0.2'])
    cbar.set_label('%s fluorescence'%(centraltendency), rotation='270', labelpad=10)#, fontsize='5', labelpad=10)
    
    sns.tsplot(populationdata, ax=axs[-1], color=(0,1,1), condition=eventname)
    axs[-1].set_xticks(range(0, window_size, window_size/5))
    axs[-1].set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size, window_size/5)])

    axs[-1].axvline(pre_window_size, linestyle='--', color='k', linewidth=1)
    #axs[-1].set_ylim([-0.12, 0.12])
    axs[-1].set_xlabel('Time from %s (s)'%(eventname))
    axs[-1].set_ylabel('Mean %s fluorescence\nacross cells'%(centraltendency))
    
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, 'Response to %s of all ROIs.png'%(eventname)), format='png', dpi=300)
    
def mkdir_p(path):
    #makes a new directory if it doesn't exist
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def framenumberforevent(event, frames):
    framenumber = np.nan*np.zeros(event.shape)
    for ie, e in enumerate(event):
        if np.isnan(e):
            framenumber[ie] = np.nan
        else:
            temp = np.nonzero(frames<=e)[0]
            if temp.shape[0]>0:
                framenumber[ie] = np.nonzero(frames<=e)[0][-1]
            else:
                framenumber[ie] = 0
    return framenumber

def calculate_num_licks_for_each_frame(framenumberforlicks, numframes):
    numlicksperframe = np.nan*np.ones((numframes,))
    for i in range(numframes):
        numlicksperframe[i] = np.sum(framenumberforlicks==i)
    return numlicksperframe

def Benjamini_Hochberg_pvalcorrection(vector_of_pvals):
    # This function implements the BH FDR correction
    
    # Parameters:
    # Vector of p values from the different tests
    
    # Returns: Corrected p values.
    
    sortedpvals = np.sort(vector_of_pvals)
    orderofpvals = np.argsort(vector_of_pvals)
    m = sortedpvals[np.isfinite(sortedpvals)].size #Total number of hypotheses
    corrected_sortedpvals = np.nan*np.ones((sortedpvals.size,))
    corrected_sortedpvals[m-1] = sortedpvals[m-1]
    for i in range(m-2, -1, -1):
        corrected_sortedpvals[i] = np.amin([corrected_sortedpvals[i+1], sortedpvals[i]*m/(i+1)])
    correctedpvals = np.nan*np.ones((vector_of_pvals.size,))
    correctedpvals[orderofpvals] = corrected_sortedpvals
    return correctedpvals

def align_traces_around_cues(signalsT, framenumberforcues_dict, frame_of_next_cue_dict, t_fxd, framerate):
    # window_size: How many frames do you want to plot around the origin?
    
    if len(signalsT.shape)==1:
        signalsT = np.expand_dims(signalsT, axis=1)
    
    numrois = signalsT.shape[1]
    numframesforcue = int(t_fxd*framerate*1E-3)
    pre_window_size = numframesforcue  # How many frames per trial before origin to be plotted?
#     post_window_size = window_size-pre_window_size
    post_window_size = 2*numframesforcue + 70
    window_size = pre_window_size + post_window_size
    align = {}
    align_baselinesubtracted = {}

    for key in framenumberforcues_dict.keys():
        align[key] = np.NAN*np.zeros([framenumberforcues_dict[key].shape[0],
                                     window_size,
                                     numrois])
        align_baselinesubtracted[key] = np.NAN*np.zeros([framenumberforcues_dict[key].shape[0],
                                                         window_size,
                                                         numrois])
        tempbaseline = np.NAN*np.zeros([framenumberforcues_dict[key].shape[0], numrois])
        for f in range(framenumberforcues_dict[key].shape[0]):
            tempframe = int(framenumberforcues_dict[key][f])
            tempbaseline[f,:] = np.nanmean(signalsT[tempframe-pre_window_size:tempframe, :], axis=0)
            if tempframe + post_window_size >= signalsT.shape[0]:
                tempend = signalsT.shape[0]
            elif post_window_size >= frame_of_next_cue_dict[key][f]:
                tempend = tempframe + int(frame_of_next_cue_dict[key][f])
            else:
                tempend = tempframe + post_window_size
            align[key][f,:(tempend-(tempframe-pre_window_size)),:] = signalsT[tempframe-pre_window_size:tempend, :]
        align_baselinesubtracted[key] = align[key] - np.nanmean(tempbaseline, axis=0)
        align[key] = np.squeeze(align[key])
        align_baselinesubtracted[key] = np.squeeze(align_baselinesubtracted[key])
    return align, align_baselinesubtracted, pre_window_size, window_size

def calculate_centraltendency_for_rois(signal, baseline_frame, centraltendency='auROC'):
    #signal has shape numtrials x numtimepoints x numrois
    #auROCmat has shape numrois x numtimepoints
    
    if centraltendency=='auROC':
        baseline = signal[:,baseline_frame,:]
        signal = signal[np.isfinite(baseline[:,0]),:,:]
        baseline = baseline[np.isfinite(baseline[:,0]),:]
        (numtrials, numtimepoints, numrois) = signal.shape
#         print baseline.shape
        #print np.mean(baseline)
        auROCmat = np.nan*np.ones((numrois, numtimepoints))
        for roi in range(numrois):
            #print roi
            for t in range(numtimepoints):
                temp = signal[:,t,roi]#; temp = temp[np.isfinite(temp)]
#                 print temp.shape, roi, t
#                 print temp
                data = np.concatenate((baseline[np.isfinite(temp),roi], temp[np.isfinite(temp)]))
                labels = np.concatenate((np.zeros((np.sum(np.isfinite(temp)),)), np.ones((np.sum(np.isfinite(temp)),))))
                auROCmat[roi,t] = 2*auROC(labels, data)-1
            #print roi, np.mean(auROCmat[roi,:])
        #print np.mean(auROCmat[:,baseline_epoch[0]:baseline_epoch[1]])
        #raise Exception()
        return auROCmat
    elif centraltendency=='baseline subtracted mean':
        return np.nanmean(signal, axis=0).T
    
def align_around_event(signals, framenumberforevent, frames, window_size, pre_window_size):
    numrois = signals.shape[0]
    post_window_size = window_size-pre_window_size

    numtrials = framenumberforevent.shape[0]

    align = np.NAN*np.zeros([numtrials,window_size,numrois])
    align_to_plot = np.NAN*np.zeros([numtrials,window_size,numrois])

    temp = signals.T
    prevendindex = 0
    tempbaseline = np.NAN*np.zeros([numtrials, numrois])
    for i in range(numtrials):
        tempindex = framenumberforevent[i]
        if np.isfinite(tempindex):
            tempindex = int(tempindex)
            tempstartindex = np.amin([pre_window_size, tempindex]).astype(int)
            startindex = np.amin([tempstartindex, tempindex-prevendindex]).astype(int)
            tempendindex = np.amin([len(frames)-tempindex, post_window_size])
            if i<(numtrials-1) and np.isfinite(framenumberforevent[i+1]):
                endindex = np.amin([framenumberforevent[i+1]-tempindex, tempendindex]).astype(int)
            else:
                endindex = tempendindex.astype(int)
            prevendindex = tempindex+endindex
            #print tempindex, temp.shape
            #print tempindex-startindex, tempindex+endindex
            align_to_plot[i,pre_window_size-startindex:pre_window_size+endindex,:] = temp[tempindex-startindex:tempindex+endindex,:]
            align[i,pre_window_size-tempstartindex:pre_window_size+endindex,:] = temp[tempindex-tempstartindex:tempindex+endindex,:]
            tempbaseline[i,:] = np.nanmean(temp[tempindex-startindex:tempindex, :], axis=0)
    align_to_plot = align_to_plot[np.where(np.isfinite(align_to_plot[:,0,0]))[0],:,:]
    align_baselinesubtracted = align_to_plot - np.nanmean(tempbaseline, axis=0)
            
    return align, align_to_plot, align_baselinesubtracted

def plot_average_PSTH_around_event(signals, framenumberforevent, framerate, frames, savedir,
                                   window_size=30, pre_window_size=10, trialsofinterest=None,
                                   sortby='response', eventname='first lick after unpredicted reward',
                                   centraltendency='baseline subtracted mean'):
    numrois = signals.shape[0]
    _,_,align_baselinesubtracted = align_around_event(signals, framenumberforevent, frames, window_size, pre_window_size)
    
    #populationdata = np.nanmean(align_baselinesubtracted[trialsofinterest,:,:], axis=0).T-1
    populationdata = calculate_centraltendency_for_rois(align_baselinesubtracted,
                                                        pre_window_size-1)
    print populationdata.shape
    #raise Exception()
    
    temp = {}
    temp[eventname] = align_baselinesubtracted
    with open(os.path.join(indir, 'Alignedtotrial_%s.pickle'%(eventname)), 'wb') as f:
        pickle.dump(temp, f)

    if sortby == 'response':
        tempresponse = np.nanmean(populationdata[:,pre_window_size:], axis=1)
        #temp=np.divide(1, (np.arange(1,post_window_size)+0.0)**0.5)
        #tempresponse = np.sum(populationdata[:,pre_window_size+1:]*np.tile(np.expand_dims(temp,axis=0), (populationdata.shape[0],1)),
        #                     axis=1)
        sortresponse = np.argsort(tempresponse)[::-1]
    elif sortby =='':
        sortresponse = np.arange(populationdata.shape[0])[::-1]

    fig, axs = plt.subplots(2, figsize=(5, 2*5))
    cmin = np.amin(populationdata)
    cmax = np.amax(populationdata)
    cax = sns.heatmap(populationdata[sortresponse,:],
                    ax=axs[0],
                    cmap=plt.get_cmap('coolwarm'),
                    vmin=-cmax,
                    vmax=cmax)

    axs[0].grid(False)
    axs[0].set_title('Response to %s of all ROIs'%(eventname))
    axs[0].set_ylabel('Sorted ROI number')
    axs[0].set_xlabel('Time from %s (s)'%(eventname))
    axs[0].set_xticks(range(0, window_size+1, 8))
    axs[0].set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size+1, 8)])
    axs[0].set_yticks(range(0, numrois, numrois/5))
    axs[0].set_yticklabels([str(a+1) for a in range(0, numrois, numrois/5)])

    axs[0].axvline(pre_window_size, color='k', linestyle='--')

    cbar = cax.collections[0].colorbar
    #cbar.set_ticks([-0.2, 0, 0.2])
    #cbar.set_ticklabels(['-0.2', '0', '0.2'])
    cbar.set_label('%s fluorescence'%(centraltendency), rotation='270', labelpad=10)#, fontsize='5', labelpad=10)
    
    sns.tsplot(populationdata, ax=axs[-1], color=(0,1,1), condition=eventname)
    axs[-1].set_xticks(range(0, window_size, window_size/5))
    axs[-1].set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size, window_size/5)])

    axs[-1].axvline(pre_window_size, linestyle='--', color='k', linewidth=1)
    #axs[-1].set_ylim([-0.12, 0.12])
    axs[-1].set_xlabel('Time from %s (s)'%(eventname))
    axs[-1].set_ylabel('Mean %s fluorescence\nacross cells'%(centraltendency))
    
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, 'Response to %s of all ROIs.png'%(eventname)), format='png', dpi=300)
def firstlick_after_event(events, licks):
    # First lick after a given event should only be counted
    # if this lick happens before the next event. Otherwise,
    # the first licks will be double counted for multiple
    # instances of the event
    result = np.nan*np.ones(events.shape)
    for i in range(events.shape[0]):    
        temp = bisect.bisect(licks,events[i])
        if i<events.shape[0]-1:
            tempend = events[i+1]
        else:
            tempend = licks[-1]
        if temp<licks.shape[0]:
            if licks[temp] <= tempend:
                result[i] = licks[temp]
    return result

def fit_regression(x, y):
    lm = sm.OLS(y, sm.add_constant(x)).fit()
    x_range = sm.add_constant(np.array([x.min(), x.max()]))
    x_range_pred = lm.predict(x_range)
    return lm.pvalues[1], lm.params[1], x_range[:,1], x_range_pred, lm.rsquared

def CDFplot(x, ax, color=None, label='', linetype='-'):
    x = np.array(x)
    ix=np.argsort(x)
    ax.plot(x[ix], ECDF(x)(x)[ix], linetype, color=color, label=label)
    return ax

def fit_regression_and_plot(x, y, ax, plot_label='', color='k', markersize=3):
    #linetype is a string like 'bo'
    pvalue, slope, temp, temppred, R2 = fit_regression(x, y)    
    ax.scatter(x, y, color=color, label='%s p=%.3f\nR$^2$=%.3f'% (plot_label, pvalue, R2), s=markersize)
    ax.plot(temp, temppred, color=color)
    return ax, slope, pvalue, R2

def fix_any_dropped_frames(frames):
    dropped_frames = []
    diff_frames = np.diff(frames)
    inter_frame_interval = diff_frames[np.argmax(np.bincount(diff_frames))]
    frame_drop_idx = np.where(diff_frames>1.5*inter_frame_interval)[0]
    for idx in frame_drop_idx:
        numframesdropped = int(np.round((frames[idx+1]-frames[idx])/(inter_frame_interval+0.0))-1)
        temp = [frames[idx]+a*inter_frame_interval for a in range(1,numframesdropped+1)]
        dropped_frames.extend(temp)
    corrected_frames = np.sort(np.concatenate((frames, np.array(dropped_frames))))
    return corrected_frames

def calculate_lick_bout_onsets(licks, min_ili_forbout=500):
    # http://jn.physiology.org/content/jn/95/1/119.full.pdf is the reference for 500ms
    lick_bout_onsets = np.insert(licks[np.where(np.diff(licks)>min_ili_forbout)[0]+1], 0, licks[0])    
    return lick_bout_onsets

def ismembertol(x, y, tol=1E-6):
    # Are elements of x in y within tolerance of tol?
    # x and y must be 1d numpy arrays
    sortx = np.sort(x)
    orderofx = np.argsort(x)
    sorty = np.sort(y)
    current_y_idx = 0
    result = np.nan*np.zeros(x.shape)
    for i, elt in enumerate(sortx):
        temp = sorty[current_y_idx:]
        if np.any(np.abs(temp-elt)<=tol):
            result[orderofx[i]]=1
        else:
            result[orderofx[i]]=0
        temp = np.argwhere(sorty>elt)
        if temp.size>0:
            current_y_idx = temp[0][0]
    return result

def mkdir_p(path):
    #makes a new directory if it doesn't exist
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def framenumberforevent(event, frames):
    framenumber = np.nan*np.zeros(event.shape)
    for ie, e in enumerate(event):
        if np.isnan(e):
            framenumber[ie] = np.nan
        else:
            temp = np.nonzero(frames<=e)[0]
            if temp.shape[0]>0:
                framenumber[ie] = np.nonzero(frames<=e)[0][-1]
            else:
                framenumber[ie] = 0
    return framenumber

def calculatepastrewrate(timesofinterest,
                         rewardtimes,
                         T_ime,
                         rewardmagnitudes):
    numtimes = timesofinterest.shape[0]
    pastrewrate = np.nan*np.ones((numtimes,))
    tau = T_ime/2
    for i in range(numtimes):
        if i == 0:
            prevpastrewrate = 0
            prevpastrewratetime = 0
        else:
            prevpastrewrate = pastrewrate[i-1]
            prevpastrewratetime = timesofinterest[i-1]
            
        temp = rewardtimes[np.logical_and(rewardtimes>=timesofinterest[i-1],
                                          rewardtimes<timesofinterest[i])]-timesofinterest[i]
        tempmag = rewardmagnitudes[np.logical_and(rewardtimes>=timesofinterest[i-1],
                                          rewardtimes<timesofinterest[i])]
        
        if temp.shape[0]>0:
            temp1 = prevpastrewrate*np.exp(-(timesofinterest[i]-prevpastrewratetime)/tau)
            temp2 = np.sum(tempmag*np.exp(temp/tau))/tau
        else:
            temp1 = 0
            temp2 = 0
        
        pastrewrate[i] = temp2 
        
        
    return pastrewrate

def calculate_num_licks_for_each_frame(framenumberforlicks, numframes):
    numlicksperframe = np.nan*np.ones((numframes,))
    for i in range(numframes):
        numlicksperframe[i] = np.sum(framenumberforlicks==i)
    return numlicksperframe

def align_traces_around_events(signalsT, framenumberforevents_dict,
                               frame_of_next_event_dict, baseline_duration, framerate):
    # window_size: How many frames do you want to plot around the origin?
    
    if len(signalsT.shape)==1:
        signalsT = np.expand_dims(signalsT, axis=1)
    
    numrois = signalsT.shape[1]
    numframesforbaseline = int(baseline_duration*framerate*1E-3)
    pre_window_size = numframesforbaseline  
    # How many frames per trial before origin to be plotted?
    post_window_size = 5*numframesforbaseline
    window_size = pre_window_size + post_window_size
    print('Pre window size = %d, window size = %d'%(pre_window_size, window_size))
    align = {}
    align_baselinesubtracted = {}

    for key in framenumberforevents_dict.keys():
        align[key] = np.NAN*np.zeros([framenumberforevents_dict[key].shape[0],
                                     window_size,
                                     numrois])
        align_baselinesubtracted[key] = np.NAN*np.zeros([framenumberforevents_dict[key].shape[0],
                                                         window_size,
                                                         numrois])
        tempbaseline = np.NAN*np.zeros([framenumberforevents_dict[key].shape[0], numrois])
        for f in range(framenumberforevents_dict[key].shape[0]):
            if np.isfinite(framenumberforevents_dict[key][f]):
                tempframe = int(framenumberforevents_dict[key][f])
                tempbaseline[f,:] = np.nanmean(signalsT[tempframe-pre_window_size:tempframe, :],
                                               axis=0)
                if tempframe + post_window_size >= signalsT.shape[0]:
                    tempend = signalsT.shape[0]
                elif post_window_size >= frame_of_next_event_dict[key][f]:
                    tempend = tempframe + int(frame_of_next_event_dict[key][f])
                else:
                    tempend = tempframe + post_window_size
                align[key][f,:(tempend-(tempframe-pre_window_size)),:] = \
                signalsT[tempframe-pre_window_size:tempend, :]
        align_baselinesubtracted[key] = align[key] - np.nanmean(tempbaseline, axis=0)
        align[key] = np.squeeze(align[key])
        align_baselinesubtracted[key] = np.squeeze(align_baselinesubtracted[key])
    return align, align_baselinesubtracted, pre_window_size, window_size

def threshold_align_by_numtrials(align_baselinesubtracted, numtrialthreshold=5):
    # During optogenetic inhibition, many frames get blanked out. The exact blanking of
    # this can be slightly misaligned on some trials due to the resonant scanning framerate
    # not being exactly 30 Hz. So just a few trials can sometimes have an additional
    # non-blanked-out frame when data are aligned to the cue. To prevent this, we will remove
    # any frame lag wrt cue onset in which the number of trials containing valid frames
    # at that lag is less than numtrialthreshold
    for key in align_baselinesubtracted.keys():
        temp = align_baselinesubtracted[key]
        tempmean = np.nanmean(temp, axis=2) # average all ROIs
        numvalidtrials_perlag = np.sum(np.isfinite(tempmean), axis=0)
        tempmask1 = (numvalidtrials_perlag < numtrialthreshold)
        tempmask = np.ones(tempmask1.shape)
        tempmask[tempmask1] = np.nan
        tempmask = np.expand_dims(np.expand_dims(tempmask, axis=0), axis=2)
        thresholdmask = np.tile(tempmask, (temp.shape[0], 1, temp.shape[2]))
        align_baselinesubtracted[key] *= thresholdmask
      
    return align_baselinesubtracted

def calculate_centraltendency_for_rois(signal, baseline_frame, centraltendency='auROC'):
    #signal has shape numtrials x numtimepoints x numrois
    #auROCmat has shape numrois x numtimepoints
    
    if centraltendency=='auROC':
        baseline = signal[:,baseline_frame,:]
        signal = signal[np.isfinite(baseline[:,0]),:,:]
        baseline = baseline[np.isfinite(baseline[:,0]),:]
        (numtrials, numtimepoints, numrois) = signal.shape
#         print baseline.shape
        #print np.mean(baseline)
        auROCmat = np.nan*np.ones((numrois, numtimepoints))
        for roi in range(numrois):
            #print roi
            for t in range(numtimepoints):
                temp = signal[:,t,roi]#; temp = temp[np.isfinite(temp)]
#                 print temp.shape, roi, t
#                 print temp
                data = np.concatenate((baseline[np.isfinite(temp),roi], temp[np.isfinite(temp)]))
                labels = np.concatenate((np.zeros((np.sum(np.isfinite(temp)),)), np.ones((np.sum(np.isfinite(temp)),))))
                auROCmat[roi,t] = 2*auROC(labels, data)-1
            #print roi, np.mean(auROCmat[roi,:])
        #print np.mean(auROCmat[:,baseline_epoch[0]:baseline_epoch[1]])
        #raise Exception()
        return auROCmat
    elif centraltendency=='baseline subtracted mean':
        return np.nanmean(signal, axis=0).T
    
def align_around_event(signals, framenumberforevent, frames, window_size, pre_window_size):
    numrois = signals.shape[0]
    post_window_size = window_size-pre_window_size

    numtrials = framenumberforevent.shape[0]

    align = np.NAN*np.zeros([numtrials,window_size,numrois])
    align_to_plot = np.NAN*np.zeros([numtrials,window_size,numrois])

    temp = signals.T
    prevendindex = 0
    tempbaseline = np.NAN*np.zeros([numtrials, numrois])
    for i in range(numtrials):
        tempindex = framenumberforevent[i]
        if np.isfinite(tempindex):
            tempindex = int(tempindex)
            tempstartindex = np.amin([pre_window_size, tempindex]).astype(int)
            startindex = np.amin([tempstartindex, tempindex-prevendindex]).astype(int)
            tempendindex = np.amin([len(frames)-tempindex, post_window_size])
            if i<(numtrials-1) and np.isfinite(framenumberforevent[i+1]):
                endindex = np.amin([framenumberforevent[i+1]-tempindex, tempendindex]).astype(int)
            else:
                endindex = tempendindex.astype(int)
            prevendindex = tempindex+endindex
            #print tempindex, temp.shape
            #print tempindex-startindex, tempindex+endindex
            align_to_plot[i,pre_window_size-startindex:pre_window_size+endindex,:] = temp[tempindex-startindex:tempindex+endindex,:]
            align[i,pre_window_size-tempstartindex:pre_window_size+endindex,:] = temp[tempindex-tempstartindex:tempindex+endindex,:]
            tempbaseline[i,:] = np.nanmean(temp[tempindex-startindex:tempindex, :], axis=0)
    align_to_plot = align_to_plot[np.where(np.isfinite(align_to_plot[:,0,0]))[0],:,:]
    align_baselinesubtracted = align_to_plot - np.nanmean(tempbaseline, axis=0)
            
    return align, align_to_plot, align_baselinesubtracted

def plot_average_PSTH_around_event(signals, framenumberforevent, framerate, frames, savedir,
                                   window_size=30, pre_window_size=10, trialsofinterest=None,
                                   sortby='response', eventname='first lick after unpredicted reward',
                                   centraltendency='baseline subtracted mean'):
    numrois = signals.shape[0]
    _,_,align_baselinesubtracted = align_around_event(signals, framenumberforevent, frames, window_size, pre_window_size)
    
    #populationdata = np.nanmean(align_baselinesubtracted[trialsofinterest,:,:], axis=0).T-1
    populationdata = calculate_centraltendency_for_rois(align_baselinesubtracted,
                                                        pre_window_size-1)
    print populationdata.shape
    #raise Exception()
    
    temp = {}
    temp[eventname] = align_baselinesubtracted
    with open(os.path.join(indir, 'Alignedtotrial_%s.pickle'%(eventname)), 'wb') as f:
        pickle.dump(temp, f)

    if sortby == 'response':
        tempresponse = np.nanmean(populationdata[:,pre_window_size:], axis=1)
        #temp=np.divide(1, (np.arange(1,post_window_size)+0.0)**0.5)
        #tempresponse = np.sum(populationdata[:,pre_window_size+1:]*np.tile(np.expand_dims(temp,axis=0), (populationdata.shape[0],1)),
        #                     axis=1)
        sortresponse = np.argsort(tempresponse)[::-1]
    elif sortby =='':
        sortresponse = np.arange(populationdata.shape[0])[::-1]

    fig, axs = plt.subplots(2, figsize=(5, 2*5))
    cmin = np.amin(populationdata)
    cmax = np.amax(populationdata)
    cax = sns.heatmap(populationdata[sortresponse,:],
                    ax=axs[0],
                    cmap=plt.get_cmap('coolwarm'),
                    vmin=-cmax,
                    vmax=cmax)

    axs[0].grid(False)
    axs[0].set_title('Response to %s of all ROIs'%(eventname))
    axs[0].set_ylabel('Sorted ROI number')
    axs[0].set_xlabel('Time from %s (s)'%(eventname))
    axs[0].set_xticks(range(0, window_size+1, 8))
    axs[0].set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size+1, 8)])
    axs[0].set_yticks(range(0, numrois, numrois/5))
    axs[0].set_yticklabels([str(a+1) for a in range(0, numrois, numrois/5)])

    axs[0].axvline(pre_window_size, color='k', linestyle='--')

    cbar = cax.collections[0].colorbar
    #cbar.set_ticks([-0.2, 0, 0.2])
    #cbar.set_ticklabels(['-0.2', '0', '0.2'])
    cbar.set_label('%s fluorescence'%(centraltendency), rotation='270', labelpad=10)#, fontsize='5', labelpad=10)
    
    sns.tsplot(populationdata, ax=axs[-1], color=(0,1,1), condition=eventname)
    axs[-1].set_xticks(range(0, window_size, window_size/5))
    axs[-1].set_xticklabels([str(((a-pre_window_size+0.0)/framerate)) for a in range(0, window_size, window_size/5)])

    axs[-1].axvline(pre_window_size, linestyle='--', color='k', linewidth=1)
    #axs[-1].set_ylim([-0.12, 0.12])
    axs[-1].set_xlabel('Time from %s (s)'%(eventname))
    axs[-1].set_ylabel('Mean %s fluorescence\nacross cells'%(centraltendency))
    
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, 'Response to %s of all ROIs.png'%(eventname)), format='png', dpi=300)