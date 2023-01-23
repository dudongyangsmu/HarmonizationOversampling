
# -*- coding: UTF-8 -*-
from __future__ import print_function
import logging
import os
import scipy.io as sio
import radiomics
from radiomics import featureextractor

patient_list = list()
name_len = list()
fea_len = list()
feaName = list()
feaValue = list()
dataPath = 'I:\\211_PT_CT\\NII_UPDATE\\PT'    

for i_patient in os.walk(dataPath):
    patient_list.append(i_patient)

for ii in range(len(patient_list)-1):  
    imageName = os.path.join(dataPath, patient_list[0][1][ii], 'image_crop.nii')  
    maskName = os.path.join(dataPath, patient_list[0][1][ii], 'mask_c.nii')   


    # Get the location of the example settings file
    paramsFile = os.path.abspath(os.path.join('PT_params.yaml'))

    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
        print('Error getting testcase!')
        exit()

    # Get the PyRadiomics logger (default log-level = INFO
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

    print("Calculating features")
    featureVector = extractor.execute(imageName, maskName)

    print(type(featureVector))

    featureName = list(featureVector.keys())  
    new_featureName = []
    for name in featureName:
      if 'diagnostics' not in name:
        new_featureName.append(name)

    featureValue = []
    for myKey in new_featureName:
      featureValue.append(featureVector[myKey])

    feaName.append(new_featureName)
    feaValue.append(featureValue)


savePath = 'I:\\HarmOversamp\\featureData\\pyradiomics\codes\\feaData_Test\\PT_b1_r2_features_c.mat'  
sio.savemat(savePath, {'FeatureValue': feaValue, 'patients': patient_list[0][1]})  
savePath2 = 'I:\\HarmOversamp\\featureData\\pyradiomics\\codes\\feaData_Test\\PT_b1_r2_featureName_c.mat'  
sio.savemat(savePath2, {'FeatureName': new_featureName})  




