

import mtrf
import pickle
import mne
import numpy as np
import scipy.stats as stats
import pandas as pd
import random
import os
from trf_functions import predict_time_constrained_trial
import sys 


subnum = sys.argv[1]
subnum = int(subnum)

# +
subjects = ['sub05','sub06', 'sub07', 'sub08','sub09','sub10','sub11','sub12','sub13','sub14','sub15','sub16',
'sub18','sub19','sub20','sub21','sub22','sub23', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29', 'sub30', 
'sub31', 'sub32', 'sub33', 'sub34', 'sub35', 'sub36', 'sub37', 'sub38', 'sub39', 'sub40', 'sub41', 'sub42', 'sub43', 
'sub44', 'sub45', 'sub46', 'sub47', 'sub48', 'sub49', 'sub50', 'sub51', 'sub52', 'sub53', 'sub54', 'sub55', 'sub56', 
'sub57', 'sub58', 'sub59', 'sub60', 'sub61', 'sub62', 'sub63', 
'sub64', 'sub65', 'sub66', 'sub67', 'sub68', 'sub69', 'sub70', 'sub72']


from mtrf.model import TRF, load_sample_data
from mtrf.stats import pearsonr, neg_mse


with open("../trf_input/channels.pickle", "rb") as input_file:
    channels = pickle.load(input_file)


import itertools
import copy

featn = "ac"

tmin = -0.1
tmax = 0.8


sub = subjects[subnum]



mod_name = "acoustic"
newpath = "../trf_results/across/"



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


for i in range(len(resp_sub)):
    targetlen = np.max(np.nonzero(feat_sub[i][:,1])[0])

    resp_stan[i] = resp_stan[i][64:targetlen,]
    feat_sub[i] = feat_sub[i][64:targetlen, ]


rtrain = copy.deepcopy(resp_stan)
strain = copy.deepcopy(feat_sub)


print("Analysis starts with " + str(len(rtrain)), flush = True)
print("Analysis starts with " + str(len(strain)), flush = True)


trialnums = np.array(range(len(feat_sub)))
trialnums_unshuffled = trialnums.copy()

trf = TRF(metric=pearsonr)
pred_leftout =[]



splits = np.split(trialnums, 10)
for i in range(len(splits)):
    testind = splits[i]   
    
    trainind = np.concatenate([splits[j] for j in range(len(splits)) if j != i]) 


    #random sampling 
    #90-10 CV 
    
    trainind = [i for i in trialnums if i not in testind]

    rtrain = [resp_stan[i] for i in trainind]
    rtest = [resp_stan[i] for i in testind]
    
    strain = [feat_sub[i] for i in trainind]
    stest = [feat_sub[i] for i in testind]


    regularization = 0.046415888336127725

    reg = trf.train(strain, rtrain, 128, tmin, tmax, regularization)
    print(regularization)

    r_crossval_target = predict_time_constrained_trial(trf,stest,rtest,average=False)


    mod_weights = trf
    pred_leftout.extend(r_crossval_target[3])

mod_name = "acoustic"
newpath = "../trf_results/" + "within" + "/" +"pred_resid" + "/"

if not os.path.exists(newpath):
    os.makedirs(newpath)


trfs_name = "resp_resid_{m}_{s}_cv_nb.pickle".format(m = mod_name, s = sub)
trfp = newpath + trfs_name
with open(trfp, "wb") as output_file:
    pickle.dump(pred_leftout, output_file)
