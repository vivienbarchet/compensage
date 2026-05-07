
library(readr)
library(tidyverse)

dat<-read_csv("../behav_stories_eeg_c500_noresid_pred.csv")


dat <- dat %>%
  
  filter(SNR != 100 ) %>%
  filter(!(condi == "sin" & SNR == 0)) %>%
  
  group_by(subject) %>%
  dplyr::summarize(SU = mean(SU), 
                   SNR = mean(SNR), 
                   PTA_L = mean(PTA_L), 
                   PTA_R = mean(PTA_R), 
                   enc_unpred_early = mean(enc_unpred_early), 
                   enc_pred_early = mean(enc_pred_early), 
                   enc_unpred_late = mean(enc_unpred_late), 
                   enc_pred_late = mean(enc_pred_late), 
                   age = mean(age), 
                   LE1 = mean(LE1), 
                   LE2 = mean(LE2), 
                   RST = mean(RST), 
                   stroop = mean(stroop)) %>%
  ungroup() %>%
  
  rowwise() %>%
  mutate(LEm = mean(c(LE1, LE2))) %>%
  ungroup() %>%
  mutate(context = 500)




dat5<-read_csv("../behav_stories_eeg_c5_noresid_pred.csv")


dat5 <- dat5 %>%
  
  filter(SNR != 100 ) %>%
  filter(!(condi == "sin" & SNR == 0)) %>%
  
  group_by(subject) %>%
  dplyr::summarize(SU = mean(SU), 
                   SNR = mean(SNR), 
                   PTA_L = mean(PTA_L), 
                   PTA_R = mean(PTA_R), 
                   enc_unpred_early = mean(enc_unpred_early), 
                   enc_pred_early = mean(enc_pred_early), 
                   enc_unpred_late = mean(enc_unpred_late), 
                   enc_pred_late = mean(enc_pred_late), 
                   age = mean(age), 
                   LE1 = mean(LE1), 
                   LE2 = mean(LE2), 
                   RST = mean(RST), 
                   stroop = mean(stroop)) %>%
  ungroup() %>%
  rowwise() %>%
  mutate(LEm = mean(c(LE1, LE2))) %>%
  ungroup() %>%
  mutate(context = 5)




dat100<-read_csv("../behav_stories_eeg_c100_noresid_pred.csv")


dat100 <- dat100 %>%
  
  filter(SNR != 100 ) %>%
  filter(!(condi == "sin" & SNR == 0)) %>%
  
  group_by(subject) %>%
  dplyr::summarize(SU = mean(SU), 
                   SNR = mean(SNR), 
                   PTA_L = mean(PTA_L), 
                   PTA_R = mean(PTA_R), 
                   enc_unpred_early = mean(enc_unpred_early), 
                   enc_pred_early = mean(enc_pred_early), 
                   enc_unpred_late = mean(enc_unpred_late), 
                   enc_pred_late = mean(enc_pred_late), 
                   age = mean(age), 
                   LE1 = mean(LE1), 
                   LE2 = mean(LE2), 
                   RST = mean(RST), 
                   stroop = mean(stroop)) %>%
  ungroup() %>%
  rowwise() %>%
  mutate(LEm = mean(c(LE1, LE2))) %>%
  ungroup() %>%
  mutate(context = 100)

datc <- rbind(dat, dat5)

datc <- datc %>%
  mutate(
    age = scale(age), 
    enc_unpred_late = scale(enc_unpred_late), 
    enc_unpred_early = scale(enc_unpred_early),
    PTA_R = scale(PTA_R), 
    RST = scale(RST), 
    stroop = scale(stroop), 
    SNR = scale(SNR), 
    LE1 = scale(LE1), 
    LE2 = scale(LE2), 
    enc_diff_late = (enc_pred_late-enc_unpred_late)/(enc_pred_late + enc_unpred_late), 
    enc_diff_early = (enc_pred_early-enc_unpred_early)/(enc_pred_early + enc_unpred_early)) 


lmage <- lm(PTA_R ~ age , data = datc)
datc$ptaresid <- lmage$residuals

mod <- lmer(enc_unpred_late ~ age*context + RST + stroop + ptaresid + SNR + (1|subject), data = datc)
summary(mod)

plot_model(mod, type = "eff", terms = c( 'age', 'context'))
