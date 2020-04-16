# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:50:06 2020
neuropil correction code designed by VJ
@author: Stuber_Lab
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import os
import subprocess
import bisect
import errno
import time
import pandas
import pickle
#import num2word
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
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
from sima.ROI import poly2mask, _reformat_polygons
import h5py
import sima
import sys
from scipy import optimize
from multiprocessing import Pool, cpu_count

try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        try:
            clear_output()
        except Exception:
            # terminal IPython has no clear_output
            pass
        print '\r', self,
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) / 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

def correct_sima_paths(h5filepath, savedir, simadir, dual_channel, masked=False):
    # This is a function that corrects the paths to datafiles in all files
    # within the .sima directory. It assumes that the original data file is
    # a .h5 file in the same directory as the .sima directory, and that
    # the name of this file is such that if its name is "data.h5", the 
    # .sima directory has the name "data_mc.sima". So there should be an
    # "_mc.sima" at the end of the .sima directory
    if not os.path.isdir(os.path.join(savedir, simadir)):
        raise Exception('%s does not exist in %s'%(simadir, savedir))
    sequencesdict = pickle.load(open(os.path.join(savedir, simadir, 'sequences.pkl'), 'rb'))
    datasetdict = pickle.load(open(os.path.join(savedir, simadir, 'dataset.pkl'), 'rb'))
    #print sequencesdict[0]['base']['base']['sequences'][0].keys()
    print datasetdict
    
    if dual_channel:
        abspath=sequencesdict[0]['base']['base']['sequences'][0]['_abspath']
    elif masked:
        abspath=sequencesdict[0]['base']['base']['base']['_abspath']
    else:
        abspath=sequencesdict[0]['base']['base']['_abspath']
    correctabspath = h5filepath
    #print correctabspath, abspath
    if abspath != correctabspath:
        print('Paths not appropriate in the .sima directory. Correcting them..')
        sequencesdict[0]['base']['base']['_abspath'] = correctabspath
        datasetdict['savedir'] = os.path.join(savedir, simadir)
        with open(os.path.join(savedir, simadir, 'sequences.pkl'), 'wb') as out1:
            pickle.dump(sequencesdict, out1)
        with open(os.path.join(savedir, simadir, 'dataset.pkl'), 'wb') as out2:
            pickle.dump(datasetdict, out2)
            
def load_rois_for_session(session_folder):
    temp = os.walk(session_folder).next()[1]
    sima_folder = [a for a in temp if '.sima' in a][0]
    with open(os.path.join(session_folder, sima_folder, 'signals_0.pkl'), 'rb') as temp:
        a = pickle.load(temp)
    numrois = len(a[sorted(a.keys())[-1]]['rois']) #Load the latest extraction
    im_shape = a[sorted(a.keys())[-1]]['rois'][0]['im_shape'][1:]
    #roi_polygons = [a[sorted(a.keys())[-1]]['rois'][roi_id]['polygons'][0][:,:-1] for roi_id in range(numrois)] # no z coordinate
    roi_polygons = [a[sorted(a.keys())[-1]]['rois'][roi_id]['polygons'][0] for roi_id in range(numrois)]  # with z coordinate
    
    return roi_polygons, im_shape

def calculate_roi_centroids(session_folder):
    roi_polygons, im_shape = load_rois_for_session(session_folder)
    roi_centroids = [Polygon(roi).centroid.coords[0] for roi in roi_polygons]
    return roi_centroids, im_shape, roi_polygons

def calculate_roi_masks(roi_polygons, im_size):
    masks = []
    if len(im_size) == 2:
        im_size = (1,) + im_size
    roi_polygons = _reformat_polygons(roi_polygons)
    for poly in roi_polygons:
        mask = np.zeros(im_size, dtype=bool)
        # assuming all points in the polygon share a z-coordinate
        z = int(np.array(poly.exterior.coords)[0][2])
        if z > im_size[0]:
            warn('Polygon with zero-coordinate {} '.format(z) +
                 'cropped using im_size = {}'.format(im_size))
            continue
        x_min, y_min, x_max, y_max = poly.bounds

        # Shift all points by 0.5 to move coordinates to corner of pixel
        shifted_poly = Polygon(np.array(poly.exterior.coords)[:, :2] - 0.5)

        points = [Point(x, y) for x, y in
                  product(np.arange(int(x_min), np.ceil(x_max)),
                          np.arange(int(y_min), np.ceil(y_max)))]
        points_in_poly = list(filter(shifted_poly.contains, points))
        for point in points_in_poly:
            xx, yy = point.xy
            x = int(xx[0])
            y = int(yy[0])
            if 0 <= y < im_size[1] and 0 <= x < im_size[2]:
                mask[z, y, x] = True
        masks.append(mask[0,:,:])
    return masks

def calculate_spatialweights_around_roi(indir, roi_masks, roi_centroids,
                                        neuropil_radius, min_neuropil_radius, h5file):
    # roi_centroids has order (x,y). The index for any roi_masks is in row, col shape or y,x shape.
    # So be careful to flip the order when you subtract from centroid
    numrois = len(roi_masks)
    allrois_mask = np.logical_not(np.sum(roi_masks, axis=0))
    (im_ysize, im_xsize) = allrois_mask.shape
    y_base = np.tile(np.array([range(1,im_ysize+1)]).transpose(), (1,im_xsize))
    x_base = np.tile(np.array(range(1,im_xsize+1)), (im_ysize,1))
    
    # Set weights for a minimum radius around all ROIs to zero as not the whole ROI is drawn
    deadzones_aroundrois = np.ones((im_ysize, im_xsize))
    for roi in range(numrois):
        x_diff = x_base-roi_centroids[roi][0]
        y_diff = y_base-roi_centroids[roi][1]
        dist_from_centroid = np.sqrt(x_diff**2 + y_diff**2)
        temp = np.ones((im_ysize, im_xsize))
        temp[dist_from_centroid<min_neuropil_radius] = 0
        deadzones_aroundrois *= temp
    
    allrois_mask *= deadzones_aroundrois.astype(bool)
      
    h5 = h5py.File(os.path.join(indir, '%s_spatialweights_%d_%d.h5'%(os.path.splitext(h5file)[0],
                                                                     min_neuropil_radius,
                                                                     neuropil_radius)),
                   'w', libver='latest')
    
    output_shape = (numrois, im_ysize, im_xsize)
    h5['/'].create_dataset(
        'spatialweights', output_shape, maxshape=output_shape,
        chunks=(1, output_shape[1], output_shape[2]))
    
    for roi in range(numrois):
        x_diff = x_base-roi_centroids[roi][0]
        y_diff = y_base-roi_centroids[roi][1]
        dist_from_centroid = np.sqrt(x_diff**2 + y_diff**2)
        spatialweights = np.exp(-(x_diff**2 + y_diff**2)/neuropil_radius**2)
        spatialweights *= im_ysize*im_xsize/np.sum(spatialweights)
        
        # Set weights for a minimum radius around the ROI to zero
        #spatialweights[dist_from_centroid<min_neuropil_radius] = 0
        # Set weights for pixels containing other ROIs to 0
        spatialweights *= allrois_mask
        """fig, ax = plt.subplots()
        ax.imshow(spatialweights, cmap='gray')
        raise Exception()"""
        h5['/spatialweights'][roi, :, :] = spatialweights

def calculate_neuropil_signals(h5filepath, neuropil_radius, min_neuropil_radius,
                               masked=False):
    
    savedir = os.path.dirname(h5filepath)
    h5file = h5py.File(h5filepath,'r') #Read-only 
    h5filename = os.path.basename(h5filepath)
    
    if os.path.splitext(h5filename)[0][-4:] == '_CH1':
        dual_channel = True
    else:
        dual_channel = False
        
    simadir = os.path.splitext(h5filename)[0]+'_mc.sima'
    
    correct_sima_paths(h5filepath, savedir, simadir, dual_channel, masked=masked)
    dataset = sima.ImagingDataset.load(os.path.join(savedir, simadir))
    sequence = dataset.sequences[0]
    frame_iter1 = iter(sequence)
    def fill_gaps(framenumber):  #adapted from SIMA source code  
        first_obs = next(frame_iter1)
        for frame in frame_iter1:
            for frame_chan, fobs_chan in zip(frame, first_obs):
                fobs_chan[np.isnan(fobs_chan)] = frame_chan[np.isnan(fobs_chan)]
            if all(np.all(np.isfinite(chan)) for chan in first_obs):
                break
        most_recent = [x * np.nan for x in first_obs]
        while True:
            frame = np.array(sequence[framenumber])[0,:,:,:,:]
            for fr_chan, mr_chan in zip(frame, most_recent):
                mr_chan[np.isfinite(fr_chan)] = fr_chan[np.isfinite(fr_chan)]
            temp=[np.nan_to_num(mr_ch) + np.isnan(mr_ch) * fo_ch
                for mr_ch, fo_ch in zip(most_recent, first_obs)]
            framenumber = yield np.array(temp)[0,:,:,0]


    fill_gapscaller = fill_gaps(0)
    fill_gapscaller.send(None)
    
    roi_centroids, im_shape, roi_polygons = calculate_roi_centroids(savedir)
    roi_masks = calculate_roi_masks(roi_polygons, im_shape)
        
    if not os.path.isfile(os.path.join(savedir, '%s_spatialweights_%d_%d.h5'%(os.path.splitext(h5filename)[0],
                                                                           min_neuropil_radius, neuropil_radius))):
        calculate_spatialweights_around_roi(savedir, roi_masks, roi_centroids,
                                            neuropil_radius, min_neuropil_radius, h5filename)
    h5weights = h5py.File(os.path.join(savedir, '%s_spatialweights_%d_%d.h5'%(os.path.splitext(h5filename)[0],
                                                                           min_neuropil_radius, neuropil_radius)), 'r')
    spatialweights = h5weights['/spatialweights']
    """for roi in range(1, spatialweights.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(spatialweights[roi,:,:], cmap='gray')
        ax.spy(roi_masks[roi], marker='.', markersize=2, color='r')
        ax.grid('off')
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        #fig.savefig(os.path.join(savedir, 'spatialweights_%d_%d.png'%(min_neuropil_radius, neuropil_radius)),
        #            format='png', dpi=300)
        raise Exception()"""
    
    numframes = h5file['/imaging'].shape[0]
    neuropil_signals = np.nan*np.ones((len(roi_masks), numframes))
    
    #pb = ProgressBar(numframes)
    start_time = time.time()
    for frame in range(numframes):
        #temp = np.array(dataset.sequences[0][frame])[:,0,:,:,0]
        temp = fill_gapscaller.send(frame)[None,:,:] #this will fill gaps in rows by interpolation
        neuropil_signals[:, frame] = np.einsum('ijk,ijk,ijk->i', spatialweights,
                                               temp, np.isfinite(temp))#/np.sum(spatialweights, axis=(1,2))
        # The einsum method above is way faster than multiplying array elements individually
        # The above RHS basically implements a nanmean and averages over x and y pixels
        
        #pb.animate(frame+1)
    neuropil_signals /= np.sum(spatialweights, axis=(1,2))[:,None]    
    print 'Took %.1f seconds to analyze %s\n' % (time.time()-start_time, savedir)
    np.save(os.path.join(savedir, '%s_neuropilsignals_%d_%d.npy'%(os.path.splitext(h5filename)[0],
                                                               min_neuropil_radius,
                                                               neuropil_radius)),
            neuropil_signals)
    
def calculate_neuropil_signals_for_session(indir, neuropil_radius=50,
                                           min_neuropil_radius=15, beta_neuropil=0.8,
                                           masked=False):
    # masked refers to whether any frames have been masked due to light artifacts
    print indir
    sys.stdout.flush()
    tempfiles = os.walk(indir).next()[2]

    npyfiles = [f for f in tempfiles if os.path.splitext(f)[1]=='.npy' and 'neuropil' not in f and 'temp' not in f]
    print npyfiles
    if len(npyfiles)>1:
        dendrite_or_soma = 'soma'
        try:
            npyfiles = [f for f in npyfiles if dendrite_or_soma in f]
        except:
            raise Exception('Too many .npy files found. Only keep the extracted signals file')
    npyfile = npyfiles[0]

    h5files = [f for f in tempfiles if os.path.splitext(f)[1]=='.h5' and 'spatialweights' not in f]
    if len(h5files) > 1:
        raise Exception('Too many .h5 files found. Only keep the data file for this session')
    h5file = h5files[0]

    if not os.path.isfile(os.path.join(indir, '%s_neuropilsignals_%d_%d.npy'%(os.path.splitext(h5file)[0],
                                                                              min_neuropil_radius,
                                                                              neuropil_radius))):
        calculate_neuropil_signals(os.path.join(indir, h5file), neuropil_radius,
                                   min_neuropil_radius, masked=masked)
    
    #raise Exception()
    
    h5filepath = os.path.join(indir, h5file)
    signals = np.squeeze(np.load(os.path.join(indir, npyfile)))
    savedir = os.path.dirname(h5filepath)
    h5file = h5py.File(h5filepath,'r') #Read-only 
    h5filename = os.path.basename(h5filepath)
    simadir = os.path.splitext(h5filename)[0]+'_mc.sima'
    dataset = sima.ImagingDataset.load(os.path.join(savedir, simadir))
    roi_centroids, im_shape, roi_polygons = calculate_roi_centroids(savedir)
    roi_masks = calculate_roi_masks(roi_polygons, im_shape)
    mean_roi_response = np.nansum(roi_masks*dataset.time_averages[:,:,:,0], axis=(1,2))/np.sum(roi_masks, axis=(1,2))
    #print mean_roi_response
        
    signals *= mean_roi_response[:,None]
    
    #raise Exception()
    
    
    
    neuropil_signals = np.squeeze(np.load(os.path.join(indir,
                                                       '%s_neuropilsignals_%d_%d.npy'%(os.path.splitext(h5filename)[0],
                                                                                       min_neuropil_radius,
                                                                                       neuropil_radius))))
    #print neuropil_signals
    #raise Exception()
    #neuropil_signals *= (512*494/np.sum(spatialweights, axis=(1,2)))[:,None]
    
    beta_rois, skewness_rois = calculate_neuropil_coefficients_for_session(indir, signals, neuropil_signals,
                                                                           neuropil_radius, min_neuropil_radius,
                                                                           beta_neuropil=beta_neuropil)
    
    save_neuropil_corrected_signals(indir, signals, neuropil_signals, beta_rois,
                                    neuropil_radius, min_neuropil_radius, h5filename)
        
def fit_regression(x, y):
    lm = sm.OLS(y, sm.add_constant(x)).fit()
    x_range = sm.add_constant(np.array([x.min(), x.max()]))
    x_range_pred = lm.predict(x_range)
    return lm.pvalues[1], lm.params[1], x_range[:,1], x_range_pred, lm.rsquared

def calculate_neuropil_coefficients_for_session(indir, signals, neuropil_signals,
                                                neuropil_radius, min_neuropil_radius, beta_neuropil=None):
    skewness_rois = np.nan*np.ones((signals.shape[0],2)) #before, after correction
    if beta_neuropil is None:
        beta_rois = np.nan*np.ones((signals.shape[0],))
        for roi in range(signals.shape[0]):        
            def f(beta):
                temp1 = signals[roi]-beta*neuropil_signals[roi]
                temp2 = neuropil_signals[roi]
                _,_,_,_,temp3 = fit_regression(temp1, temp2)
                return temp3
            #beta_rois[roi] = optimize.brent(f)
            beta_rois[roi] = optimize.minimize(f, [1], bounds=((0,None),)).x
            skewness_rois[roi,0] = stats.skew(signals[roi])
            temp1 = signals[roi]-beta_rois[roi]*neuropil_signals[roi]
            temp2 = neuropil_signals[roi]
            _,temp4,_,_,temp3 = fit_regression(temp1, temp2)
            skewness_rois[roi,1] = np.sqrt(temp3)* np.sign(temp4)
            """print beta_rois[roi]
            fig, ax = plt.subplots()
            ax.plot(signals[roi], 'b')
            ax.plot(neuropil_signals[roi], 'r')
            ax.plot(beta_rois[roi]*neuropil_signals[roi], 'g')
            raise Exception()"""
        #print beta_rois
        fig, axs = plt.subplots(1,2,figsize=(8,4))
        CDFplot(beta_rois, axs[0])
        CDFplot(skewness_rois[:,1], axs[1])
        #raise Exception()
        return beta_rois, skewness_rois
    else:
        skewness_rois[:,0] = stats.skew(signals, axis=1)
        skewness_rois[:,1] = stats.skew(signals-beta_neuropil*neuropil_signals, axis=1)
        """tempslope = np.nan*np.ones((signals.shape[0],))
        corrected_signals = signals-beta_neuropil*neuropil_signals
        for roi in range(signals.shape[0]):
            fig, ax = plt.subplots()
            ax.plot(signals[roi,:1000], 'b', alpha=0.5, label='uncorrected signal')
            ax.plot(corrected_signals[roi,:1000], 'b', label='corrected signal')
            ax.plot(neuropil_signals[roi,:1000], '--r', label='raw neuropil signal')
            ax.plot(beta_neuropil*neuropil_signals[roi,:1000], 'r', label='scaled neuropil signal') 
            ax.set_xlabel('Frame number (5Hz)')
            ax.set_ylabel('Signal amplitude')
            ax.legend(loc='upper right')
            fig.savefig(os.path.join(indir, 'neuropil', 'roi%d.png'%roi))
            fig.clf()"""
            #tempslope[roi] = sm.OLS(corrected_signals[roi,:], sm.add_constant(neuropil_signals[roi,:])).fit().params[1]
        #raise Exception()
        """fig, ax = plt.subplots(figsize=(4,4))
        CDFplot(tempslope, ax=ax)
        ax.set_xlabel('Correlation coefficient between\ncorrected signal and neuropil signal')
        ax.set_ylabel('Cumulative probability')
        ax.axvline(0,color='k',linestyle='--')
        fig.subplots_adjust(left=0.15)
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(top=0.98)
        fig.savefig(os.path.join(indir, 'correlationdistribution_%d_%d_beta_%.1f.png'%(min_neuropil_radius,
                                                                                      neuropil_radius,
                                                                                      beta_neuropil)),
                   format='png', dpi=300)"""
        #raise Exception()
        return beta_neuropil, skewness_rois

def save_neuropil_corrected_signals(indir, signals, neuropil_signals, beta_rois,
                                    neuropil_radius, min_neuropil_radius, h5file):
    if isinstance(beta_rois, np.ndarray):
        corrected_signals = signals-beta_rois[:,None]*neuropil_signals
        np.save(os.path.join(indir, '%s_neuropil_corrected_signals_%d_%d_betacalculated.npy'%(os.path.splitext(h5file)[0],
                                                                                           min_neuropil_radius,
                                                                                           neuropil_radius)),
                corrected_signals)
    else:
        corrected_signals = signals-beta_rois*neuropil_signals
        np.save(os.path.join(indir, '%s_neuropil_corrected_signals_%d_%d_beta_%.1f.npy'%(os.path.splitext(h5file)[0],
                                                                                         min_neuropil_radius,
                                                                                         neuropil_radius,
                                                                                         beta_rois)),
                corrected_signals)
    
def CDFplot(x, ax, color=None, label='', linetype='-'):
    x = np.squeeze(np.array(x))
    ix=np.argsort(x)
    ax.plot(x[ix], ECDF(x)(x)[ix], linetype, color=color, label=label)
    return ax