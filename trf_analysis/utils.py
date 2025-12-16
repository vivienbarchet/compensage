
import numpy as np

def feat_order(input_feat, num, nums):
    output_feat = []


    for n in num:
        ind = nums.index(n)
        s = input_feat[ind]
        output_feat.append(np.array(s))
    return output_feat


def zero_pad(input_feat, lens,stim = "target", pad = True):
    output_padded = []
    count = 0
    if stim == "target":
        beg_pad = 64
    else:
        beg_pad = 0
    for s in input_feat:
        if pad == True:
            env_len = lens[count]
           
            if env_len > len(s) or env_len == len(s):
                pad_r = env_len - (len(s)+beg_pad)
            
                surp_pad = np.pad(s, (beg_pad, pad_r), 'constant')
            else:
                surp_pad = s[0:env_len]
                print("Warning: Trial numbers might be incorrect.")
                print("Length envelope: " + str(env_len))
                print("Length feature: " + str(len(s)))
            surp_pad = np.nan_to_num(surp_pad)
        else:
            if stim == "target":
                env_len = lens[count]
                
                surp_pad = s[0:env_len]
                if len(surp_pad) < env_len:
                    surp_pad = np.pad(surp_pad, (0, env_len-len(surp_pad)), 'constant')
                surp_pad = np.nan_to_num(surp_pad)
            else:
                env_len = lens[count]
                
                surp_pad = s[64:env_len+64]
                if len(surp_pad) < env_len:
                    surp_pad = np.pad(surp_pad, (0, env_len-len(surp_pad)), 'constant')
                surp_pad = np.nan_to_num(surp_pad)
                
        output_padded.append(surp_pad)
        count = count +1
    return output_padded

def z_score(feat):
    feat_stan = []
    feat_concat = np.concatenate(feat)

    feat_mean = np.mean(feat_concat)
    feat_std = np.std(feat_concat)

    for f in feat:
        f_stan = (f-feat_mean)/feat_std
        feat_stan.append(f_stan)
    return feat_stan