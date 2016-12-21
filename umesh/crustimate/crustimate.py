# -*- coding: utf-8 -*-
"""
Classify a large patch into crust, non-crust, etc.
Analyzes every ply in the target folder

Created on Wed Nov 30 14:24:13 2016

@author: umesh
"""

#import svm
from time import time
import os
from multiprocessing import freeze_support

#%% Main function
if __name__ == '__main__':
    freeze_support()
    #%% Initialization
    progStartTime = time()

    #load ply
    pathPly = r'/media/umesh/DATADRIVE1/Processing_Softwares/Gitclones/Crustimate/testdata'
    pathOut = os.path.join(pathPly, 'out')
    
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    
    kernel_size = 0.05 #10cm wide square
    pointsThreshold = 50#100
    
    # Whether to regenearte the kernel files
    flagReGen = True
    
    # No. of parallel processes to run
    NUMBER_OF_PROCESSES = 6
    
    #%% SVM Classification information
    paramList = ['Avg_Luminosity','Std_Dev_Luminosity','Entropy']
    scaleFactorList = [255.0,100.0,5.0]
    offsetList = [0,0,0]

    # initialize SVM
    print '\nInitialize class'
    from SupportVectorMachines.svm_data_class import SvmVector
    testSet = SvmVector(paramList,scaleFactorList,offsetList) # use same parameters for training and testing

    # Load SVM classifier model
    from sklearn.externals import joblib
    fileSVMModel = r'/media/umesh/DATADRIVE1/3D_Coalition/Estimator_Models/7_linux/SVM_Estimator.pkl'
    svmModel = joblib.load(fileSVMModel)# Loading SVM model

    plyList = [x for x in os.listdir(pathPly) if '.ply' in x ]
    
    for fileName in plyList:
        onePlyFile = os.path.join(pathPly,fileName)
        print "Reading " + onePlyFile + "\n"
        pathOut2 = os.path.join(pathOut, fileName.split('.')[0])
        discPath = os.path.join(pathOut2, 'Discs_%s'%fileName)   
        pathCorner = os.path.join(pathOut2,'List_of_corners.csv')
        pathResults = os.path.join(pathOut2, 'Disk_Analysis_log.csv')
       
        if flagReGen == True:
            from Rasterize.functions.libPly import split_ply_square
            split_ply_square(onePlyFile, pathOut2, discPath, kernel_size, pointsThreshold, NUMBER_OF_PROCESSES)
            
            from Rasterize.functions.libPly import trim_ply_folder
            trim_ply_folder(discPath, pathOut2, pointsThreshold, NUMBER_OF_PROCESSES)
        
        from Rasterize.functions.libPlyAnalyse import analyse_ply_folder
        analyse_ply_folder(discPath, pathOut2, pathCorner, NUMBER_OF_PROCESSES)
        
#        file3 = r'/media/umesh/DATADRIVE1/Processing_Softwares/Gitclones/Crustimate/testdata/out/3D_0026000_to_0026700_/Disk_Analysis_log.csv'
        testSet.load_vectors(pathResults) # no need to load y values for testing data

        # Predict for unknown data
        pathSvmOut = os.path.join(pathOut2, 'SVM_out.csv')
        print "Predicting for the test data"
        testSet.predict_svm(svmModel,pathSvmOut)
    
        # Plot the classifier output into a 3D ply file
        from SupportVectorMachines.svm_library import make_ply_cubes
        pathCubes = os.path.join(pathOut2, 'SVM_Prediction.ply')
        make_ply_cubes(testSet,pathCubes,d = kernel_size)

    print("Code execution took %s seconds\n" % (time() - progStartTime))
