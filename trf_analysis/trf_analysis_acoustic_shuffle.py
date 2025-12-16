#!/usr/bin/env python


import mtrf
import pickle
import mne
import numpy as np
import scipy.stats as stats
import pandas as pd

import os
import multiprocessing
import sys 
import itertools


subnum=sys.argv[1]
subnum = int(subnum)
feat_shuffle=sys.argv[2]


from trf_functions import predict_time_constrained_trial
from trf_cv_functions import *
# +
subjects = ['sub05','sub06', 'sub07', 'sub08','sub09','sub10','sub11','sub12','sub13','sub14','sub15','sub16',
'sub18','sub19','sub20','sub21','sub22','sub23', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29', 'sub30', 
'sub31', 'sub32', 'sub33', 'sub34', 'sub35', 'sub36', 'sub37', 'sub38', 'sub39', 'sub40', 'sub41', 'sub42', 'sub43', 
'sub44', 'sub45', 'sub46', 'sub47', 'sub48', 'sub49', 'sub50', 'sub51', 'sub52', 'sub53', 'sub54', 'sub55', 'sub56', 
'sub57', 'sub58', 'sub59', 'sub60', 'sub61', 'sub62', 'sub63', 
'sub64', 'sub65', 'sub66', 'sub67', 'sub68', 'sub69', 'sub70', 'sub72']


sub = subjects[subnum]

from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse

feats_dict = {"ac_target": [1,2], "ac_dis":[0,3], "onsets_target":[4,6], "onsets_dis":[7,8], "acon_target":[1,2,4,6,9], "acon_dis":[0,3,7,8,10]}
feat_index = feats_dict[feat_shuffle]

featn = "ac"

seedlist = range(0,50)

globmean =[]

for ise, seed in enumerate(seedlist):
    random.seed(seed)
    num_it = seed


    ##Load features and eeg response
    path_in = "../trf_input/{s}/{f}_{s}_nb.pickle".format(s =sub, f = featn)
    with open(path_in, "rb") as input_file:
        feat_sub = pickle.load(input_file)


    path_in = "../trf_input/{s}/response_new_{s}_nb_within.pickle".format(s =sub)
    with open(path_in, "rb") as input_file:
        resp_sub = pickle.load(input_file)





    resp_concat = np.concatenate(resp_sub)

    # Calculate mean and standard deviation
    m_resp = np.mean(resp_concat, axis = 0)
    std_resp = np.std(resp_concat, axis = 0)

    resp_stan = []
    for i,ar in enumerate(resp_sub):
        
        s = (ar-m_resp)/std_resp
        resp_stan.append(np.array(s))


    ##Cut to target length
    for i in range(len(resp_stan)):
     


        targetlen = np.max(np.nonzero(feat_sub[i][:,1])[0])

        resp_stan[i] = resp_stan[i][64:targetlen,]
        feat_sub[i] = feat_sub[i][64:targetlen, ]


     


    trialnums = range(len(feat_sub))
    [trf_fits,pred, mod_weights, testtrials] = shuffle_data_ac(num_it, feat_sub, resp_stan, feat_index, sub, feat_shuffle, tr = True)


     
    #Summarize the fits
    trf_fits = trf_fits.groupby(['subject', 'trialnum', 'channel', 'feature_shuffle'], as_index = False).mean('fit')


    mean_predt = []
    ##Summarize the predictions for saving
    for tt in trialnums:
        indices = [i for i, x in enumerate(testtrials) if x == tt]

        pred_trial = [pred[i] for i in indices]

        mean_pred = np.mean(pred_trial, axis = 0)

        mean_predt.append(mean_pred)


    globmean.append(mean_predt)

    mod = "acoustic"
    newpath = "../trf_results/" + sub + "/" +"shuffle" + "/"

    if not os.path.exists(newpath):
        os.makedirs(newpath)


    name_fits = "fits_{m}_{s}_{i}_{f}_within_nb_stan_notr.csv".format(m = mod, s = sub, i = num_it, f = feat_shuffle)
    save_path = newpath + name_fits
    trf_fits.to_csv(save_path)


    trfs_name = "trfs_{m}_{s}_{i}_{f}_within_nb_stan.pickle".format(m = mod, s= sub, i = num_it, f = feat_shuffle)
    trfp = newpath + trfs_name
    with open(trfp, "wb") as output_file:
        pickle.dump(mod_weights, output_file)


n_iterations = len(globmean)
n_trials = len(globmean[0])

mean_trials = []

for trial_idx in range(n_trials):
    arrays = [globmean[it][trial_idx] for it in range(n_iterations)]
    tm = np.mean(arrays, axis = 0)
    mean_trials.append(tm)



resp_name = "prediction_{m}_{s}_{i}_{f}_within_nb_stan.pickle".format(m = mod, s=sub, i = num_it, f = feat_shuffle)
trfp = newpath + resp_name
with open(trfp, "wb") as output_file:
   pickle.dump(mean_trials, output_file)






