

import mtrf
import pickle
import mne
import numpy as np
import scipy.stats as stats
import pandas as pd
import numpy as np
import copy
import os
import multiprocessing
import sys 




subjects = ['sub05','sub06', 'sub07', 'sub08','sub09','sub10','sub11','sub12','sub13','sub14','sub15','sub16',
'sub18','sub19','sub20','sub21','sub22','sub23', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29', 'sub30', 
'sub31', 'sub32', 'sub33', 'sub34', 'sub35', 'sub36', 'sub37', 'sub38', 'sub39', 'sub40', 'sub41', 'sub42', 'sub43', 
'sub44', 'sub45', 'sub46', 'sub47', 'sub48', 'sub49', 'sub50', 'sub51', 'sub52', 'sub53', 'sub54', 'sub55', 'sub56', 
'sub57', 'sub58', 'sub59', 'sub60', 'sub61', 'sub62', 'sub63', 
'sub64', 'sub65', 'sub66', 'sub67', 'sub68', 'sub69', 'sub70', 'sub72']

import sys
sys.path.append('../')
from utils import *




for subject in subjects:
# +
     #Load the eeg data from mne
    eeg_path = "/data/pt_02917/eeg_mne/saved_{sub}_new-epo.fif".format(sub = subject)

    ep = mne.read_epochs(eeg_path)



    response = ep.get_data()
    subnum = int(subject[3:])
   

    env_path_target = "../features/trf_input/envelope_target.pickle"
    with open(env_path_target, "rb") as input_file:
        env_target = pickle.load(input_file)

    with open("../features/trf_input/envelope_dis.pickle", "rb") as input_file:
        env_dis = pickle.load(input_file)



    stimnum = "../features/trf_input/stimnum_target.pickle"

    with open(stimnum, "rb") as input_file:
        nums = pickle.load(input_file)

    with open("../features/trf_input/stimnum_env.pickle", "rb") as inp:
        num_env = pickle.load(inp)

    with open("../features/trf_input/f0_target.pickle", "rb") as input_file:
        f0_target = pickle.load(input_file)

    with open("../features/trf_input/f0_dis.pickle", "rb") as input_file:
        f0_dis = pickle.load(input_file)
    
   
    num_f0 = num_env
    
  
    
    with open("../features/trf_input/rate_dis.pickle", "rb") as input_file:
        rate_dis = pickle.load(input_file)

    with open("../features/trf_input/rate_target.pickle", "rb") as input_file:
        rate_target = pickle.load(input_file)

    ###Distractor


    with open("../features/trf_input/stimnum_dis.pickle", "rb") as input_file:
        stimnum_dis = pickle.load(input_file)

    # +
    ##Load the subject data (here for the stimulus order)
    subnum = subject[3:]
    subnum = int(subnum)

    if subnum == 69:
        subdat_path = "/data/pt_02917/data/sub69/main_exp/sub69_complete.csv"
    else:
        subdat_path = "/data/pt_02917/data/{sub}/main_exp/{sub}_mainexp.csv".format(sub = subject)
    subdat = pd.read_csv(subdat_path, sep = ";")

    #Remove baseline trials
    subdat = subdat[subdat['db_cond'] != 10]
    cases = subdat['target']
    num = cases.str[0:3]
    num = pd.to_numeric(num)

    cases_dis = subdat['distractor']
    num_dis = cases_dis.str[0:7]

    ###Get amplitude of the target 
    amp = subdat['srt_db']+subdat['db_cond']
    factor = pow(10,(amp/20))
    factor = list(factor)

    ###Remove baseline trials from responses and envelopes and onsets 

    num_nb = list(num.index)

    response= [response[i] for i in num_nb]
    
    print(num)
    print(num_env)
    env_target = feat_order(env_target, num, num_env)
    env_dis = feat_order(env_dis, num, num_env)
    rate_target = feat_order(rate_target, num, num_env)
    rate_dis = feat_order(rate_dis, num, num_env)


    ##Cut the eeg data into the stimulus length 
    lens = []
    for e in env_target:
        lens.append(len(e))

    resp_cut = []
    count = 0
    for arar in response:
        arar_cut = []
        countar = 0


        for ar in arar:
            if countar < 63 and countar != 35 and countar != 36:
                ar_cut = ar[:lens[count]]

                arar_cut.append(ar_cut)
            countar = countar + 1

        ar_t = np.transpose(np.array(arar_cut))
        resp_cut.append(ar_t)
    
            
        count = count + 1




    featstar =['surprisal', 'frequency','entropy','phone_surprisal', 'phone_entropy', 'word_onset', 'phone_onset']

    featdict ={}
    for f in featstar:
        with open("../features/trf_input/{f}_target_final_yo.pickle".format(f = f), "rb") as input_file:
            feat = pickle.load(input_file)
        
        print(len(feat))
        f_order = feat_order(feat, num, nums)
      
        f_pad = zero_pad(f_order, lens)
        
        featdict[f] = f_pad

    
    featstar =['surprisal', 'frequency','entropy','phone_surprisal', 'phone_entropy', 'word_onset', 'phone_onset']


    for f in featstar:
       
        with open("../features/trf_input/{f}_dis_final_yo.pickle".format(f = f), "rb") as input_file:
            feat = pickle.load(input_file)
        
        f_order = feat_order(feat, num_dis, stimnum_dis)
        f_pad = zero_pad(f_order, lens, stim = "distractor")
        fn = f + "_" + "dis"
        featdict[fn] = f_pad


    speakerfeats =['vertrauen_target', 'age_target','gebildet_target', 'dominant_target', 'attraktiv_target', 
    'gesund_target', 'professionell_target', 
    'vertrauen_dis', 'age_dis','gebildet_dis', 'dominant_dis', 'attraktiv_dis', 'gesund_dis', 'professionell_dis']

    for f in speakerfeats:
        with open("../features/trf_input/{f}.pickle".format(f = f), "rb") as input_file:
            feat = pickle.load(input_file)

        if "dis" in f:
            f_order = feat_order(feat, num_dis, stimnum_dis)
            f_pad = zero_pad(f_order, lens, stim = "distractor")
        else:
            f_order = feat_order(feat, num, nums)
            f_pad = zero_pad(f_order, lens)

        featdict[f] = f_pad


    f0_target = feat_order(f0_target, num, num_f0)
    f0_dis = feat_order(f0_dis, num, num_f0)



    # +
    ##Create the input arrays

    out_name = "response_new_{sub}_nb_within.pickle".format(sub = subject)
    trfp = "../features/trf_input/" + subject 

    if not os.path.exists(trfp):
        os.makedirs(trfp)

    path_out = trfp + "/" + out_name
    with open(path_out, "wb") as output_file:
        pickle.dump(resp_cut, output_file)




    trials = range(0,len(env_target))
    print(len(env_target))
    all_feat = []


    for t in trials:
        trialar = []

        trialar.append(featdict['surprisal'][t])
        trialar.append(featdict['surprisal_dis'][t])

        trialar.append(featdict['frequency'][t])
        trialar.append(featdict['frequency_dis'][t])

        trialar.append(featdict['phone_surprisal'][t])
        trialar.append(featdict['phone_surprisal_dis'][t])
        
        trialar.append(featdict['phone_entropy'][t])
        trialar.append(featdict['phone_entropy_dis'][t])

        trialar.append(featdict['entropy'][t])
        trialar.append(featdict['entropy_dis'][t])

        trialar = np.transpose(np.array(trialar))

        all_feat.append(trialar)
        
  
    ##And pickle 
    out_name = "ling_{sub}_nb_resid_within_withent_ws.pickle".format( sub = subject)
    trfp = "../features/trf_input/" + subject 

    if not os.path.exists(trfp):
        os.makedirs(trfp)

    path_out = "../features/trf_input/" + subject + "/" + out_name
    with open(path_out, "wb") as output_file:
        pickle.dump(all_feat, output_file)
    


    all_feat = []


    for t in trials:
        trialar = []
        ##Acoustic 
        
        #Distractor envelope
        trialar.append(env_dis[t].reshape(len(env_dis[t])))
    
        #Target envelope
        trialar.append(env_target[t].reshape(len(env_dis[t])))
        
        ##Acoustic onsets
        trialar.append(rate_target[t].reshape(len(env_dis[t])))
        
        #Target envelope
        trialar.append(rate_dis[t].reshape(len(env_dis[t])))
        
        #Word onset target (no first)      
        ind = np.nonzero(featdict['word_onset'][t])[0][0]

        nofirst = copy.deepcopy(featdict['word_onset'][t])
        nofirst[ind] = 0

        trialar.append(nofirst)

        
        ##Word onset target (only first)
        ind = np.nonzero(featdict['word_onset'][t])[0][1:]
    

        first_onset = copy.deepcopy(featdict['word_onset'][t])
        first_onset[ind] = 0

        trialar.append(first_onset)
        


        #Phone onset target (no first)
        ind = np.nonzero(featdict['phone_onset'][t])[0][0]

        nofirst = copy.deepcopy(featdict['phone_onset'][t])
        nofirst[ind] = 0

      

        trialar.append(nofirst)
        



        #Word onset dis 

        # print(onset_target_first_padded_all[t])
        first_onset = copy.deepcopy(featdict['word_onset_dis'][t])
        trialar.append(first_onset)


        first_onset = copy.deepcopy(featdict['phone_onset_dis'][t])

    
        trialar.append(first_onset)



        trialar.append(f0_target[t])
        trialar.append(f0_dis[t])


        trialar = np.transpose(np.array(trialar))

        all_feat.append(trialar)

 
    ##And pickle 
    out_name = "ac_{sub}_nb.pickle".format( sub = subject)
    trfp = "../features/trf_input/" + subject + "/"

    if not os.path.exists(trfp):
        os.makedirs(trfp)

    path_out = trfp + out_name
    with open(path_out, "wb") as output_file:
        pickle.dump(all_feat, output_file)
    


    control = []
    first_onset = []

    for t in trials:
        trialar = []
        ##Acoustic 
        #Distractor envelope
        trialar.append(env_dis[t].reshape(len(env_dis[t])))
    
        #Target envelope
        trialar.append(env_target[t].reshape(len(env_dis[t])))
        
        ##Acoustic onsets
        trialar.append(rate_target[t].reshape(len(env_dis[t])))
        
        #Target envelope
        trialar.append(rate_dis[t].reshape(len(env_dis[t])))
        
        #Word onset target (no first)      
        ind = np.nonzero(featdict['word_onset'][t])[0][0]

        nofirst = copy.deepcopy(featdict['word_onset'][t])
        nofirst[ind] = 0

        trialar.append(nofirst)

        
        ##Word onset target (only first)
        ind = np.nonzero(featdict['word_onset'][t])[0][1:]
    

        first_onset = copy.deepcopy(featdict['word_onset'][t])
        first_onset[ind] = 0

        trialar.append(first_onset)
        


        #Phone onset target (no first)
        ind = np.nonzero(featdict['phone_onset'][t])[0][0]

        nofirst = copy.deepcopy(featdict['phone_onset'][t])
        nofirst[ind] = 0



        trialar.append(nofirst)
        



        #Word onset dis 

        # print(onset_target_first_padded_all[t])
        first_onset = copy.deepcopy(featdict['word_onset_dis'][t])
        trialar.append(first_onset)


        first_onset = copy.deepcopy(featdict['phone_onset_dis'][t])

        trialar.append(first_onset)



        trialar.append(f0_target[t])
        trialar.append(f0_dis[t])

        
        for sf in speakerfeats:
            s = featdict[sf][t]

            ##Speaker features
            ind = np.nonzero(s)[0][0]
            first_onset = copy.deepcopy(s)
            first_onset[ind] = 0
            trialar.append(first_onset)


        
        trialar = np.transpose(np.array(trialar))

        control.append(trialar)

  
    out_name = "ac_control_{sub}_nb.pickle".format( sub = subject)
    trfp = "../features/trf_input/" + subject + "/"

    if not os.path.exists(trfp):
        os.makedirs(trfp)

    path_out = trfp + out_name
    with open(path_out, "wb") as output_file:
        pickle.dump(control, output_file)



