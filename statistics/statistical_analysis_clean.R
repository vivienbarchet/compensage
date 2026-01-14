library(dplyr)
library(lme4)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(RColorBrewer)
library(data.table)
library(Hmisc)
library(DescTools)
library(tidyverse)
library(readxl)
library(Hmisc) 

setwd("/data/pt_03114/trf_analysis_yo/code_upload/statistics/")
#Load fits
fits_sub_ac <- fread("./fits_ac_within.csv")



fits_sub_ac <- fits_sub_ac %>%
  select(-V1) %>%
  mutate(correlation = FisherZ(correlation))%>%
  rename(feat = feature_shuffle)


fits_sub <- fread("./fits_ling_within.csv")

fits_sub <- fits_sub %>%
  select(-V1) %>%
  mutate(correlation = FisherZ(correlation)) %>%
  rename(feat = feature_shuffle)


fits_sub <- rbind(fits_sub_ac, fits_sub)


##Load TR fits 


fits_sub_tr <- fread("./tr_fits_ling.csv")

fits_sub_tr <- fits_sub_tr %>%
  select(-V1) %>%
  mutate(correlation = FisherZ(correlation)) %>%
  filter( feat == "ling_word_target") %>%
  mutate(feat = "ling_word_target_tr")


fits_sub <- rbind(fits_sub, fits_sub_tr)


fits_sub_wide <- fits_sub %>%
  pivot_wider(names_from = feat, id_cols = c(trialnum, subject), values_from = correlation)




#Load behavioral data 

data = read.csv("./behavdat.csv", sep = ";")

data <- data %>%
  mutate(subject = if_else(subject == "41", "sub41", subject))


###Create the trial numbers 
c <- data %>%
  filter(db_cond != 10) %>%
  group_by(subject,trialnum, target) %>%
  dplyr::summarize() %>%
  group_by(subject) %>%
  mutate(sent_id = 1:n()) %>%
  mutate(sent_id = sent_id-1)

data_merge <- data %>%
  merge(c, by = c('subject', 'trialnum', 'target'))


#Merge behavior and fits
fits_sub <- fits_sub_wide %>%
  rename(sent_id = trialnum) 

data_behav_eeg <- data_merge %>%
  rename(subject = subject) %>%
  merge(fits_sub, by = c('subject', 'sent_id'))





baseline <- data %>%
  filter(db_cond == 10)%>%
  mutate(acc = if_else(acc == "True", 1, 0)) %>%
  group_by(subject) %>%
  dplyr::summarize(meana = mean(acc))



data_model <- data_behav_eeg %>%
  rename(subject = subject) %>%
  filter(db_cond != 10) %>%
  mutate(acc = as.factor(acc)) %>%
  group_by(subject) %>%
  mutate(audibility = scale(audibility)) %>%
  ungroup() %>%
  #eliminate first word surprisal
  mutate(surprisal = if_else(word_order == 1, NaN, surprisal)) %>%
  mutate(surprisal = scale(surprisal), 
         entropy = scale(entropy)) %>% 
  ungroup() %>%
  group_by(subject, target) %>%
  mutate(sentence_len = n()) %>%
  ungroup() %>%
  mutate(frequency = scale(frequency)) 

mean(baseline$meana)

sd(data_model$srt_snr)
range(data_model$srt_snr)

data_model <- left_join(data_model, baseline, by = "subject")

data_model <- data_model %>%
  mutate(meana = scale(meana)) %>%
  mutate(
    onsets_target=scale(onsets_target),
    onsets_dis = scale(onsets_dis), 
    ac_target = scale(ac_target), 
    ac_dis = scale(ac_dis), 
    word_target = scale(ling_word_target), 
    phoneme_target= scale(ling_phone_target), 
    word_target_tr = scale(ling_word_target_tr)
  ) %>%
  ungroup() 


#Load demographics
age <- read_excel("./demographics/age_new.xlsx")
hearing <- read_csv("./demographics/hearing_thresholds_oneside.csv")
wl <- read.csv("./demographics/wordlist_meanacc.csv")

wl <- wl %>%
  rename(acc_wl = acc)




data_model <- left_join(data_model, age, by="subject")
data_model <- left_join(data_model, hearing, by="subject")
data_model <- left_join(data_model, wl, by= "subject")



data_model <- data_model %>%
  mutate(age = scale(Age)) %>%
  mutate(thresholdsc = scale(threshold)) 




data_sub <- data_model %>%
  group_by(subject, age,Age, threshold) %>%
  mutate(acc = if_else(acc == "True", 1, 0)) %>%
  dplyr::summarize(sub_acc = sum(acc)/n(), 
                   mean_acc_wl = mean(acc_wl),
                   sentence_len = mean(sentence_len),
                   meana = mean(meana),
                   srt_db = mean(srt_db),
                   surprisal = mean(surprisal, na.rm = T), 
                   frequency = mean(frequency), 
                   audibility = mean(audibility),
                   onsets_target=mean(onsets_target),
                   onsets_dis = mean(onsets_dis), 
                   ac_target = mean(ac_target), 
                   ac_dis = mean(ac_dis), 
                   word_target = mean(word_target), 
                   phoneme_target= mean(phoneme_target), 
                   word_target_tr = mean(word_target_tr)
  )%>%
  mutate(acc_diff = (sub_acc - mean_acc_wl)/(sub_acc+mean_acc_wl))

mean(data_sub$threshold)


corm <- data_sub %>%
  ungroup() %>%
  dplyr::select("age", 
                "threshold", "srt_db", "sub_acc", "meana")

coeffs <- rcorr(as.matrix(corm))[[1]]


pvalues <- rcorr(as.matrix(corm))[[3]]
pvalues

pvalues[is.na(pvalues)] <- 1
library(corrplot)
colnames(coeffs) <- c("Age",
                      "Pure Tone Average", "Speech Reception Threshold", "Mean Accuracy", 
                      "Control Accuracy")
rownames(coeffs) <-  c("Age", "Pure Tone Average", "Speech Reception Threshold", "Mean Accuracy", 
                       "Control Accuracy")

corrplot(coeffs, method = "color", type = "lower", addCoef.col = "black", tl.col = "black", diag = F)



rn <- quantile(data_sub$Age, probs = c(0.33, 0.66)) # quartile


data_sub <- data_sub %>%
  mutate(ageg = case_when(Age <= rn[1] ~ "young",
                          (Age > rn[1]) & (Age <= rn[2]) ~ 'middle',
                          Age > rn[2] ~ 'old'))


data_sub <- data_sub %>%
  ungroup() %>%
  mutate(
    onsets_target=scale(onsets_target),
    onsets_dis = scale(onsets_dis), 
    ac_target = scale(ac_target), 
    ac_dis = scale(ac_dis), 
    word_target = scale(word_target), 
    phoneme_target= scale(phoneme_target),
    word_target_tr = scale(word_target_tr))


#Residualize PTA
lmage <- lm(threshold ~ age, data = data_sub)
data_sub$ptaresid <- lmage$residuals




###Subject-level analysis ---- 
#Wordlist accuracy
t2 <- lm(acc_diff ~ age + sub_acc+ srt_db + meana + ptaresid, data =data_sub)
summary(t2)

feats = c('ac_target', 'ac_dis', 'onsets_target', 'onsets_dis', 'word_target', 'phoneme_target', 
          'word_target_tr')

for (f in feats){
  
  #Green 
  greens <- RColorBrewer::brewer.pal(4, "Greens")[4]
  #Purple
  purples <- RColorBrewer::brewer.pal(4, "Purples")[4]
  
  eq = as.formula(paste(f, " ~ Age + srt_db + meana+ptaresid+sub_acc"))
  
  t <- lm(eq , 
          data = data_sub)
  summary(t)
  print(summary(t))
  page = summary(t)$coef[2,4]
  
  }
  
  
  



###Compensation analysis


t2 <- lm(sub_acc~age*word_target*onsets_dis +age*word_target*ac_dis+meana+ptaresid+srt_db, 
         data = data_sub)

tab_model(t2, p.adjust ="fdr")


greens3 <- RColorBrewer::brewer.pal(4, "Greens")[2:4]
plot_model(t2, type = "eff",  terms = c( 'word_target', 'onsets_dis'), color = greens3, 
           line.size = 2) + 
  theme_classic() + 
  theme(axis.title = element_text(size = 30), 
        text = element_text(size = 25)) + 
  ggtitle("") + 
  ylab("Mean\nAccuracy") + 
  xlab("Linguistic\nWord Target") + 
  guides(fill=guide_legend(title="Onsets\nDistractor")) + 
  guides(color=guide_legend(title="Onsets\nDistractor"))

#ggsave("../figures/compfig.png", dpi=700)
sname = sprintf("../../../figures/compfig.png")
ggsave(sname, width = 5.5, height =4.85, unit = "in", dpi = 700)




###Trial level model ----

data_model <- data_model %>%
  mutate(wordlen = str_length(word))


eeg_mod <- glmer(acc ~ audibility*age + age*surprisal+
                    audibility*surprisal+
                    wordlen+
                    entropy*age+
                    db_cond+word_order+
                    +frequency +sentence_len  + meana + srt_db +threshold +
                    age*ac_target + age*ac_dis + 
                    age*onsets_target+ age*onsets_dis + 
                    age*word_target + 
                    age*phoneme_target+
                    (1|subject) + (1|target), 
                  data = data_model,
                  family = "binomial", control=glmerControl(optimizer="bobyqa",
                                                            optCtrl=list(maxfun=2e5)))
summary(eeg_mod) 

tab_model(eeg_mod, p.adjust = "fdr" )




library(DHARMa)
#Model diagnostics
simulationOutput <- simulateResiduals(fittedModel = eeg_mod, plot = T)

greens <- RColorBrewer::brewer.pal(4, "Greens")[2:5]

plot_model(eeg_mod_wd, type = "pred",  terms = c('audibility', 'surprisal'), 
         color = greens, 
         line.size = 2) + 
theme_classic() + 
theme(axis.title = element_text(size = 30), 
      text = element_text(size = 25)) + 
ggtitle("") + 
ylab("Predicted Accuracy") + 
xlab("Audibility") + 
guides(fill=guide_legend(title="Surprisal")) + 
guides(color=guide_legend(title="Surprisal"))


ggsave("../figures/surp_audind_yo.png", width = 4.5, height = 4.5, unit = "in", dpi = 700)


plot_model(eeg_mod, type = "pred",  terms = c('entropy', 'age[-1.09, -0.22, 1.24]'), 
         color = c("#98accf", "#5579aa", "#3f4756"), 
         line.size = 2) + 
theme_classic() + 
theme(axis.title = element_text(size = 30), 
      text = element_text(size = 25)) + 
ggtitle("") + 
ylab("Predicted Accuracy") + 
xlab("Entropy") + 
guides(fill=guide_legend(title="Age")) + 
guides(color=guide_legend(title="Age"))


ggsave("../figures/entropy_ageint.png", width = 4, height = 4.5, unit = "in", dpi = 700 )


plot_model(eeg_mod, type = "pred",  terms = c('onsets_dis', 'age[-1.09, -0.22, 1.24]'), color = c("#98accf", "#5579aa", "#3f4756"), 
         line.size = 2) + 
theme_classic() + 
theme(axis.title = element_text(size = 30), 
      text = element_text(size = 25)) + 
scale_y_continuous(breaks = seq(0.7, 1, 0.1), limits = c(0.7, 0.95)) + 
ggtitle("") + 
ylab("") + 
xlab("Encoding\nAccuracy") + 
guides(fill=guide_legend(title="Age")) + 
guides(color=guide_legend(title="Age"))


ggsave("../figures/onsets_dis_ageint.png", width = 4, height = 4, unit = "in" , dpi = 700)




library(interactions)
sim_slopes(eeg_mod, pred = onsets_dis, modx =age, johnson_neyman = TRUE, 
         modx.values = c(-1.09, -0.22, 1.24), jnplot = TRUE)





library(car)
vif(eeg_mod)
range(vif(eeg_mod))



#Save model outputs
summary(eeg_mod)$coefficients
cc<-confint(eeg_mod,parm="beta_",method="Wald")
ctab <- cbind(summary(eeg_mod)$coefficients,cc)

modest = data.frame(ctab)
modest = modest[,c(1,2,5,6,3,4)]
modest = round(modest, 2)

ord <- c("age", "audibility", "surprisal", "audibility:surprisal","frequency", "entropy","audibility:age",
       "age:surprisal",
       "age:entropy",
       "ac_target","ac_dis",  "onsets_target", 'onsets_dis',
       "age:ac_target", "age:ac_dis", "age:onsets_target", "age:onsets_dis",
       "word_target", "phoneme_target", 
       "age:word_target", "age:phoneme_target",
       'threshold',
       'db_cond','wordlen', 'word_order', 'sentence_len', 'meana', 'srt_db')

modest$feat = rownames(modest)
modest <- modest %>%
filter(feat != "(Intercept)")
modest = modest %>%
slice(match(ord, feat)) %>%
select(-feat) %>%
mutate(CI = paste0("[", X2.5.., ", ", X97.5.., "]")) %>%
select(-X2.5.., -X97.5..)

modest = modest[,c(1,2,5,3,4)]

write.csv(modest, "/data/pt_03114/figures/modeleffects_yo.csv")

