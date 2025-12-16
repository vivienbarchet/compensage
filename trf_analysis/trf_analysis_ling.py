#!/usr/bin/env python

import mtrf
import pickle
import mne
import numpy as np
import scipy.stats as stats
import pandas as pd
import random
import os
import multiprocessing
import sys 
import itertools



subnum = sys.argv[1]



from trf_functions import predict_time_constrained_trial

# +
subjects = ['sub05','sub06', 'sub07', 'sub08','sub09','sub10','sub11','sub12','sub13','sub14','sub15','sub16',
'sub18','sub19','sub20','sub21','sub22','sub23', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29', 'sub30', 
'sub31', 'sub32', 'sub33', 'sub34', 'sub35', 'sub36', 'sub37', 'sub38', 'sub39', 'sub40', 'sub41', 'sub42', 'sub43', 
'sub44', 'sub45', 'sub46', 'sub47', 'sub48', 'sub49', 'sub50', 'sub51', 'sub52', 'sub53', 'sub54', 'sub55', 'sub56', 
'sub57', 'sub58', 'sub59', 'sub60', 'sub61', 'sub62', 'sub63', 
'sub64', 'sub65', 'sub66', 'sub67', 'sub68', 'sub69', 'sub70', 'sub72']


from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse


featn = "ling"

feat_allsub = []
resp_allsub = []
subnum=int(subnum)
sub=subjects[subnum]

##Load features and eeg response
path_in = "../trf_input/{s}/{f}_{s}_nb_resid_within_withent_ws.pickle".format(s =sub, f = featn)
with open(path_in, "rb") as input_file:
    feat_sub = pickle.load(input_file)



path_in = "../trf_results/within/pred_resid/resp_resid_acoustic_8020_speaker_{s}.pickle".format(s =sub)
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
    lent = len(resp_stan[i])
    feat_sub[i] = feat_sub[i][64:lent+64,]





import copy

with open("../trf_input/channels.pickle", "rb") as input_file:
    channels = pickle.load(input_file)

tmin = -0.1
tmax = 0.8
print("TRF analysis starts now", flush=True)


mod_weights = []

pred_leftout = []
data_list = []
data_list_tr = []


nfolds = 50
testtrials = []
trialnums = np.array(range(len(feat_sub)))
trialnums_unshuffled = trialnums.copy()

#Make sure that each trial is sampled the same number of times 
for nb in range(5):
    random.seed(nb)
    random.shuffle(trialnums)

    splits = np.split(trialnums, 10)
    for i in range(len(splits)):
        t = (nb*10)+(i+1)
        testind = splits[i]   
        
        trainind = np.concatenate([splits[j] for j in range(len(splits)) if j != i]) 
    
    
        #random sampling 
        #90-10 CV 

        trainind = [i for i in trialnums if i not in testind]

        testtrials.extend(testind)
        rtrain = [resp_stan[i] for i in trainind]
        rtest = [resp_stan[i] for i in testind]
        
        strain = [feat_sub[i] for i in trainind]
        stest = [feat_sub[i] for i in testind]


    

        trf = TRF(metric=pearsonr)
        #TRain model
        reg = 1e-07
        trf = train_only_banded(trf, strain, rtrain, 128, tmin, tmax,reg)
        mod_weights.append(trf)
        #Test model
        res = predict_time_constrained_trial(trf,stest,rtest,average = False,TC = False)

        print("Fold " + str(t) + " done, test correlation = " + str(np.mean(res[1])), flush=True)

        r_nt = np.concatenate(res[1])
        
        trials_n = testind
        trials_n = np.repeat(trials_n, 61)

        chans = channels*len(stest)

        for fit_index,fit in enumerate(r_nt):

            data_list.append({
                'subject':sub,
                'feature':"all",
                'window':"none",
                'fit':fit, 
                'trialnum':trials_n[fit_index],
                'channel':chans[fit_index]
            })

        
        pred_leftout.extend(res[0])
        
        feats = ['ling_word_target', 'ling_word_dis', 'ling_phone_target', 'ling_phone_dis']
        indexfeats = {'ling_word_target':[0,2,8], 'ling_word_dis':[1,3,9], 'ling_phone_target':[4,6], 'ling_phone_dis':[5,7]}

        for feat in feats:


            indf = indexfeats[feat]
            res = predict_time_constrained_trial(trf,stest,rtest,average = False,TC = True,onsetVecDim = indf, onsetVecWinSt = 0, onsetVecWinEnd =800)

            r_nt = np.concatenate(res[2])
            
            trials_n = testind
            trials_n = np.repeat(trials_n, 61)

            chans = channels*len(stest)

            for fit_index,fit in enumerate(r_nt):

                data_list.append({
                    'subject':sub,
                    'feature':feat,
                    'window':"mid",
                    'fit':fit, 
                    'trialnum':trials_n[fit_index],
                    'channel':chans[fit_index]
                })


        ##Do time resolved encoding
        windowsize = 6
        stepsize = 3

        lags = np.arange(tmin, tmax, 1/128)

        nwind = int(len(lags)/stepsize)

        start = 0
        for wind in range(nwind):
            ##set trf to 0 outside of the window
            end = start + windowsize
            we = trf.weights.copy()
            we[:, :start, :] = 0
            we[:, end:, :] = 0  

            start = start + stepsize 

            trf_split = trf.copy()
            trf_split.weights = we 

            #Predict 
            for feat in feats:


                indf = indexfeats[feat]
                res = predict_time_constrained_trial(trf_split,stest,rtest,average = False,TC = False)

                r_nt = np.concatenate(res[1])

                trials_n = testind
                trials_n = np.repeat(trials_n, 61)

                chans = channels*len(stest)


                for fit_index,fit in enumerate(r_nt):

                    data_list_tr.append({
                        'subject':sub,
                        'feature':feat,
                        'window':wind,
                        'fit':fit, 
                        'trialnum':trials_n[fit_index],
                        'channel':chans[fit_index]
                    })





fits_sub_tr = pd.DataFrame(data_list_tr)
  
#Summarize the fits
fits_sub_tr = fits_sub_tr.groupby(['subject', 'trialnum', 'channel', 'window', 'feature'], as_index = False).mean('fit')


fits_sub = pd.DataFrame(data_list)
  
  
#Summarize the fits
fits_sub = fits_sub.groupby(['subject', 'trialnum', 'channel', 'feature', 'window'], as_index = False).mean('fit')



mean_predt = []
##Summarize the predictions for saving
for tt in trialnums_unshuffled:
    indices = [i for i, x in enumerate(testtrials) if x == tt]

    pred_trial = [pred_leftout[i] for i in indices]

    mean_pred = np.mean(pred_trial, axis = 0)

    mean_predt.append(mean_pred)




mod = "linguistic"
newpath = "../trf_results/" + sub + "/"

if not os.path.exists(newpath):
    os.makedirs(newpath)

name_fits = "fits_{m}_{s}_within_nb_stan.csv".format(m = mod, s = sub)
save_path = newpath + name_fits
fits_sub_tr.to_csv(save_path)


name_fits = "fits_{m}_{s}_within_nb_stan_notr.csv".format(m = mod, s = sub)
save_path = newpath + name_fits
fits_sub.to_csv(save_path)


trfs_name = "trfs_{m}_{s}_within_nb_stan.pickle".format(m = mod, s= sub)
trfp = newpath + trfs_name
with open(trfp, "wb") as output_file:
    pickle.dump(mod_weights, output_file)


resp_name = "prediction_{m}_{s}_within_nb_stan.pickle".format(m = mod, s=sub)
trfp = newpath + resp_name
with open(trfp, "wb") as output_file:
    pickle.dump(mean_predt, output_file)
