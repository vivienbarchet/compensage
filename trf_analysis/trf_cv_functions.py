import numpy as np
import random
import itertools
import mtrf
from mtrf.model import TRF
from mtrf.stats import pearsonr
from trf_functions import predict_time_constrained_trial, train_diff,train_diff_noreg,train_only_banded,train_reg_notrain
import pickle 
import pandas as pd

def shuffle_data_ling(num_it, feat_sub, resp_stan, feat_index, sub, feat_shuffle, tr = False):
    print("Iteration " + str(num_it), flush = True)
    feat_shuff = []
    


    vals = []
    ars_shuff = []
    c=0
    for ar in feat_sub:
        
        ind = np.nonzero(ar[:, feat_index])
    
        vals.extend(list(ar[:,feat_index][ind]))
      
        c=c+1

    random.shuffle(vals)

    start = 0
    c=0
    for ar in feat_sub:
        
        ind = np.nonzero(ar[:, feat_index])
      
        numval = len(ind[0]) 
        num_ind = numval+start

      
        arf = ar[:,feat_index]
        arf[ind] = vals[start:num_ind]
        ar[:,feat_index] = arf
        start = start + numval
        c=c+1
    

        ars_shuff.append(ar)
    

    [trf_fits, pred_leftout, mod_weights, testtrials, fits_sub_tr] = cv_run_ling(ars_shuff,resp_stan, sub, "ling",feat_shuffle, tr = True)

    trf_fits['iteration'] = num_it
    trf_fits['feature_shuffle'] = feat_shuffle
    fits_sub_tr['iteration'] = num_it
    fits_sub_tr['feature_shuffle'] = feat_shuffle



    return trf_fits, pred_leftout, mod_weights, testtrials, fits_sub_tr


def shuffle_data_ac(num_it, feat_sub, resp_stan, feat_index, sub, feat_shuffle, tr = False):
    print("Iteration " + str(num_it), flush = True)
   
    feat_shuff = []
    
    
    
    subar = feat_sub
    sub_shuff = []
    ide_shuff = list(range(len(subar)))

    random.shuffle(ide_shuff)
    ide = list(range(len(subar)))
   
    ar_shuff = [subar[i][:,feat_index] for i in ide_shuff]
   
    for i,ar in enumerate(subar):
        len_orig = len(ar[:,0])
        len_shuff = len(ar_shuff[i])
      
        
        if len_orig > len_shuff:
            shuff = np.pad(ar_shuff[i], ((0,len_orig-len_shuff), (0,0)))
        elif len_shuff > len_orig:
            shuff = ar_shuff[i][:len_orig]
        else:
            shuff = ar_shuff[i]
        ar[:,feat_index] = shuff
        
    
    

        sub_shuff.append(ar)




    [trf_fits, pred_leftout, mod_weights, testtrials] = cv_run(sub_shuff,resp_stan, sub, "ac", feat_shuffle, tr = True)

    trf_fits['iteration'] = num_it
    trf_fits['feature_shuffle'] = feat_shuffle



    return trf_fits, pred_leftout, mod_weights, testtrials




def cv_run_ling(feat_sub, resp_stan, sub, mod, feat_shuffle, nfolds = 10, tr = False):

    mod_weights = []
    pred_leftout = []
    data_list = []
    data_list_tr = []


    with open("../trf_input/channels.pickle", "rb") as input_file:
        channels = pickle.load(input_file)



  


    tmin = -0.1
    tmax = 0.8

    nfolds = 50
    testtrials = []
    trialnums = np.array(range(len(feat_sub)))

    #Make sure that each trial is sampled the same number of times 
    for nb in range(5):
        random.seed(nb)
        random.shuffle(trialnums)

        splits = np.split(trialnums, 10)
        for i in range(len(splits)):
            t = (nb*10)+(i+1)
            testind = splits[i]   
            
            trainind = np.concatenate([splits[j] for j in range(len(splits)) if j != i]) 
        
            testtrials.extend(testind)

            rtrain = [resp_stan[i] for i in trainind]
            rtest = [resp_stan[i] for i in testind]
            
            strain = [feat_sub[i] for i in trainind]
            stest = [feat_sub[i] for i in testind]
        
            trf = TRF(metric=pearsonr)



 

            reg = 1e-07
            trf = trf.train(strain, rtrain, 128, tmin, tmax, reg)
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


            feat = feat_shuffle
            indexfeats = {'ling_word_target':[0,2,8], 'ling_word_dis':[1,3,9], 'ling_phone_target':[4,6], 'ling_phone_dis':[5,7]}


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

                feat = feat_shuffle
                indexfeats = {'ling_word_target':[0,2,8], 'ling_word_dis':[1,3,9], 'ling_phone_target':[4,6], 'ling_phone_dis':[5,7], "ac_target": [1,2], "ac_dis":[0,3], "onsets_target":[4,6], "onsets_dis":[7,8]}


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

    fits_sub = pd.DataFrame(data_list)
    fits_sub_tr = pd.DataFrame(data_list_tr)

    return fits_sub, pred_leftout, mod_weights, testtrials, fits_sub_tr





def cv_run(feat_sub, resp_stan, sub, mod, feat_shuffle, nfolds = 10, tr = False):

    mod_weights = []
    pred_leftout = []
    data_list = []


    with open("../trf_input/channels.pickle", "rb") as input_file:
        channels = pickle.load(input_file)



   


  


    tmin = -0.1
    tmax = 0.8

    nfolds = 50
    testtrials = []
    trialnums = np.array(range(len(feat_sub)))

    #Make sure that each trial is sampled the same number of times 
    for nb in range(5):
        random.seed(nb)
        random.shuffle(trialnums)

        splits = np.split(trialnums, 10)
        for i in range(len(splits)):
            t = (nb*10)+(i+1)
            testind = splits[i]   
            
            trainind = np.concatenate([splits[j] for j in range(len(splits)) if j != i]) 
        
            testtrials.extend(testind)

            rtrain = [resp_stan[i] for i in trainind]
            rtest = [resp_stan[i] for i in testind]
            
            strain = [feat_sub[i] for i in trainind]
            stest = [feat_sub[i] for i in testind]
        
            trf = TRF(metric=pearsonr)


            reg = 0.1

            trf = trf.train(strain, rtrain, 128, tmin, tmax, reg)
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


           

    fits_sub = pd.DataFrame(data_list)

    return fits_sub, pred_leftout, mod_weights, testtrials



