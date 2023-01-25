# comparison of different batch effect removal methods
# input: n*p (samples * features) matrix, batch
# output: harmonized features

# set work directory
setwd("G:/ADC_SCC/HarmOversamp/R/Harmonization/normFact/")

# add package
library(pracma)
library(R.matlab)
library(matrixStats)
library(BiocParallel)
#library(devtools)
#install_github("jfortin1/neuroCombatData")
#install_github("jfortin1/neuroCombat_Rpackage")
# load function
source("G:/ADC_SCC/HarmOversamp/R/Harmonization/neuroCombat_Rpackage/R/neuroCombat.R")
source("G:/ADC_SCC/HarmOversamp/R/Harmonization/neuroCombat_Rpackage/R/neuroCombatFromTraining.R")
source("G:/ADC_SCC/HarmOversamp/R/Harmonization/neuroCombat_Rpackage/R/neuroCombat_helpers.R")
source("G:/ADC_SCC/HarmOversamp/R/Harmonization/neuroCombat_Rpackage/R/viz.R")
source("normFact.R") # for ICA, SVD
source("scaleDStrain.R")
source("scaleDStest.R")

# load feature data and batch variate
Data<-readMat("G:/ADC_SCC/HarmOversamp/R/Harmonization/Data_example/data_internal.mat")  # modify
data<-Data$Data.internal
covar = Data$Label.internal

# test data
DataTe<-readMat("G:/ADC_SCC/HarmOversamp/R/Harmonization/Data_example/data_external.mat")  # modify
data_test<-DataTe$Data.external
covar_test = DataTe$Label.external

# batch
b = readMat("G:/ADC_SCC/HarmOversamp/R/Harmonization/Data_example/internal_batch_center.mat") # modify
b = b$interBatch

# test batch
bTe = readMat("G:/ADC_SCC/HarmOversamp/R/Harmonization/Data_example/external_batch_center.mat") # modify
b_test = bTe$exterBatch


# harmonization
normtypes = c('ICA 0.5', 'SVD', 'centering', 'combat', 'none')
K = 30 # number of components in stICA and SVD   
x = data
x_test = data_test

xnorm = list()
xnormTest = list()

for (i in 1:length(normtypes)){
  normtype = normtypes[i]
  cat(normtype)
  if ('combat' == normtype){
    # without considering covariate
    combat_v1 = neuroCombat(dat=t(x), batch=b,parametric=FALSE,ref.batch=1)
    xnorm[['combat_v1']] = t(combat_v1$dat.combat)
    estimate_v1 = combat_v1$estimates
    info_v1 = combat_v1$info
    
    # the trained estimates are applied to new data set
    TE_combat_v1 = neuroCombatFromTraining(dat=t(x_test), batch=b_test, estimates=estimate_v1)
    xnormTest[['combat_v1']] = t(TE_combat_v1$dat.combat)
    
  } 
  if ('ICA 0.5' == normtype) {
    obj = normFact('stICA',t(x),b,"categorical",k=K,alpha=0.5,ref2=covar,refType2="categorical")
    xnorm[[paste(normtype,'v2')]]= t(obj$Xn)
    effect = rowMeans2(obj$Xb)
    xnormTest[[paste(normtype,'v2')]] = x_test-effect
    
  }
  if ('centering' == normtype){
    centeringValues = scaleDStrain(b,x)
    xnorm[['centering']] = centeringValues$xc
    means = centeringValues$means
    stds = centeringValues$stds
    xnormTest[['centering']] = scaleDStest(b_test,x_test,means,stds)
  } 
  if ('none' == normtype){
    xnorm[['none']] =x
    xnormTest[['none']] = x_test
    
  } 
  
  if ('SVD' == normtype){
    obj = normFact('SVD',t(x),b,"categorical",k=K,ref2=covar,refType2="categorical")
    xnorm[['SVD_v2']]= t(obj$Xn)
    effect = rowMeans2(obj$Xb)
    xnormTest[['SVD_v2']] = x_test-effect
  } 
}


#---------------------------------------- save harmonized data
filename = "G:/ADC_SCC/HarmOversamp/Upload/HarmonizedData/DataDivision1/"  # modify
FeatureValue = xnorm[['combat_v1']]
writeMat(paste(filename, 'combat_v1.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)

FeatureValue = xnorm[['SVD_v2']]
writeMat(paste(filename, 'SVD_v2.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)

FeatureValue = xnorm[['centering']]
writeMat(paste(filename, 'centering.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)

FeatureValue = xnorm[['ICA 0.5 v2']]
writeMat(paste(filename, 'ICA 5_v2.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)


#---------test data
FeatureValue = xnormTest[['combat_v1']]
writeMat(paste(filename, 'TE_combat_v1.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)

FeatureValue = xnormTest[['SVD_v2']]
writeMat(paste(filename, 'TE_SVD_v2.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)

FeatureValue = xnormTest[['centering']]
writeMat(paste(filename, 'TE_centering.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)

FeatureValue = xnormTest[['ICA 0.5 v2']]
writeMat(paste(filename, 'TE_ICA 5_v2.mat',sep = "", collapse = ""), FeatureValue=FeatureValue)

