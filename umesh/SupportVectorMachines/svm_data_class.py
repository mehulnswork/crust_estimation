# -*- coding: utf-8 -*-
"""
SVM data class
Created on Wed Nov 25 12:14:33 2015

@author: Umesh Neettiyath

Contains a class for parsing and holding data needed  for SVM classification
v1.0 : Initial version. Based on 
        svm_trial2f.py
v1.1 : Loads x,y,z locations also into the class        
"""
import numpy as np
from os.path import exists
from sklearn import svm
from sklearn.externals import joblib

class SvmVector:
    'The m-dimensional feature vector of SVM training/testing data, and the one-dimensional prediction vector'
    __location_header = ['Avg_X','Avg_Y','Avg_Z']
    
    def __init__(self,paramList,scaleFactorList,offsetList):
        'Initiate the SVM vector. Does not read values yet.'
        if len(paramList) != len(scaleFactorList) or len(paramList) != len(offsetList):
            print 'Error in parameters. Check if all values are up to date.'
            raise RuntimeError('Error in parameters')
        self.features = list(paramList)
        self.scaleFactors = list(scaleFactorList)
        self.offsets = list(offsetList)
        self.dimension = len(paramList) # width of the feature vector, i.e. m
        self.length = 0 # number of data points, i.e. n
        self.vector = np.empty((0,self.dimension)) # instance vector of size m by n, i.e. X
        self.prediction = np.empty((0,),dtype=int) # vector containing prediction values, i.e. y
        self.location = np.empty((0,3)) # x,y,z locations of each of the points
        self.populated = 'False' # Flag indicating if list contains values
    
    def info(self):
        print "No. of features : %s\nNo. of points : %s"%(self.dimension,self.length)
        print "Features : " 
        print self.features
        print "List contains data : " + self.populated
        
    def load_vectors(self,xFileName,yFileorValue = [],delimiter=',',yHeader='Manual_Prediction'):
        '''Load test vectors and output vectors as applicable.
        Input a file of X values.y value can be a file or a value or empty.'''
        # Load vector containing values
        if not exists(xFileName):
            print 'Input file does not exist'
            raise RuntimeError('File containing values does not exist')
        nLines, indX = self.__load_vectors_x(xFileName,delimiter)
        self.__load_locations(xFileName,delimiter,nLines,indX)
        
        if yFileorValue == []:
            pass
        elif type(yFileorValue) in [int,float]:
            self.__load_vector_y_const(yFileorValue,nLines)
        elif type(yFileorValue) == str:
            if exists(yFileorValue):
                self.__load_vectors_y(yFileorValue,yHeader,nLines,indX)
            else:
                print 'File with y values not found'
                raise RuntimeWarning('File with y values not found. y values could not be loaded')
        else:
            print 'y values could not be loaded'
            raise RuntimeWarning('y values could not be loaded')
        self.populated = 'True'
    
    def __load_vectors_x(self,fileName,delimiter):
        'Load test vectors into a numpy array'
        _X = []
        indX = []
        with open(fileName) as f:
            header = f.readline()
            # Assume one line of header
            headerSplit = header.strip().split(delimiter)
            indexList = [None]*self.dimension
            for i in range(self.dimension):
                try:
                    indexList[i] = headerSplit.index(self.features[i])
                except ValueError:
                    print 'Error! Parameter' + self.features[i] + 'is not present in the file ' + fileName
                    raise RuntimeError('Error! Parameter' + self.features[i] + 'is not present in the file ' + fileName)
            nLines = 0
            j = 0
            for line in f.readlines():
                lineSplit = line.strip().split(delimiter)
                j += 1
                tempVector = []
                for i in range(self.dimension):
                    if not np.isnan(float(lineSplit[indexList[i]])):
                        tempVector.append((float(lineSplit[indexList[i]]) - self.offsets[i])/self.scaleFactors[i])
                    else:
                        break
                else:
                    _X.append(tempVector)
                    indX.append(j-1)
                    nLines += 1
        self.vector = np.vstack((self.vector,np.array(_X)))
        self.length += nLines
        return nLines, indX

    def __load_vector_y_const(self,value,nLines):
        _y = np.empty((nLines,))
        _y.fill(int(value))
        self.prediction = np.hstack((self.prediction,_y))
        
    def __load_vectors_y(self,fileName,header,nLines,indX):
        _y = np.empty((nLines,))
        with open(fileName) as f:
            if f.readline().strip() != header:
                print 'Header mismatch for prediction file. Verify!'
                raise RuntimeError('Header mismatch for prediction file. Verify!')
            lines = f.readlines()
            count = 0
            for index in indX:
                _y[count] = int(lines[index])
                count += 1
            else:
                if count != nLines:
                    print 'Mismatch in lengths of X and y files'
                    raise RuntimeError('Mismatch in lengths of X and y files')
        self.prediction = np.hstack((self.prediction,_y))
        
    def __load_locations(self,fileName,delimiter,nLines,indX):
        'Load xyz locations into a numpy array'
        _loc = np.empty((nLines,3))
        with open(fileName) as f:
            header = f.readline()
            # Assume one line of header
            headerSplit = header.strip().split(delimiter)
            indexList = [None]*3
            for i in range(3):
                try:
                    indexList[i] = headerSplit.index(SvmVector.__location_header[i])
                except ValueError:
                    print 'Error! Parameter' + SvmVector.__location_header[i] + 'is not present in the file ' + fileName
                    raise RuntimeError('Error! Parameter' + SvmVector.__location_header[i] + 'is not present in the file ' + fileName)
            lines = f.readlines()
            count = 0
            for index in indX:
                lineSplit = lines[index].strip().split(delimiter)
                for i in range(3):
                    _loc[count][i] = float(lineSplit[indexList[i]])
                count += 1
        self.location = np.vstack((self.location,_loc))
        
    def clear_data(self):
        'Clear any loaded data. Does not reset parameters'
        self.length = 0 # number of data points, i.e. n
        self.vector = np.empty((0,self.dimension)) # instance vector of size m by n, i.e. X
        self.location = np.empty((0,3)) # x,y,z locations of each of the points
        self.prediction = np.empty((0,)) # vector containing prediction values, i.e. y
        self.populated = 'False' # Flag indicating if list contains values
           
    def predict_svm(self,svmModel,fileOut=[],save = 'True'):
        'Predict for the given SVM model with the data present in the class'
        if not self.populated:
            print 'Test data not loaded into model'
            raise RuntimeError('Test data not loaded into model')
        self.prediction = svmModel.predict(self.vector)
        if save == 'True':
            if fileOut == []:
                print 'No file path provided for saving the data'
                raise RuntimeError('No file path provided for saving the data')
            np.savetxt(fileOut,self.prediction,fmt='%d',delimiter=',', newline='\n', header='SVM_Prediction',comments='')

if __name__ == '__main__':
    ''' Main function. Usage examples of SvmClass'''
    
    paramList = ['Avg_Luminosity','Std_Dev_Luminosity','Entropy']
    scaleFactorList = [255.0,100.0,5.0]
    offsetList = [0,0,0]

    # initialize
    print '\nInitialize class' 
    trainingSet = SvmVector(paramList,scaleFactorList,offsetList)
    trainingSet.info()
    
    # Load values
    print '\nLoad only x values' 
    file1 = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_1b\Run1c_swtesting\BrahmaLog_v5.0.csv'
    trainingSet.load_vectors(file1)
    trainingSet.info()

    print '\nClear and load x values with constant y value'
    trainingSet.clear_data()
    trainingSet.load_vectors(file1,1)
    trainingSet.info()

    print '\nClear and load x and y values from file. Useful for training SVM'
    file1out = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_1b\Run1c_swtesting\SVM_out.csv'
    trainingSet.clear_data()
    trainingSet.load_vectors(file1,file1out)
    trainingSet.info()

    print '\nLoad a second set and train classifier'
    file2 = r'E:\3D_Coalition\HPD1543_Morning_1_4\Patch_5c\Run1b_swtesting\BrahmaLog_v5.0.csv'
    file2out = r'E:\3D_Coalition\HPD1543_Morning_1_4\Patch_5c\Run1b_swtesting\SVM_out.csv'
    trainingSet.load_vectors(file2,file2out)
    trainingSet.info()
    
    # train SVM classifier - fit the model
    print "Fitting data"
    # The actual classification learning happens here. 
    # Kernel can be ploynomial 'poly', radial basis function 'rbf' or linear 'linear'
    svmModel = svm.SVC(kernel='poly')#,degree=3,coef0=0)
    svmModel.fit(trainingSet.vector,trainingSet.prediction)
    print "Saving the trained model to a file"
    fileModel = r'E:\3D_Coalition\Estimator_Models\7\SVM_Estimator.pkl'
    joblib.dump(svmModel,fileModel)
    
    # Create a test data
    print 'Create a test data'
    file3 = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_3b\Run1c_swtesting\BrahmaLog_v5.0.csv'
    testSet = SvmVector(paramList,scaleFactorList,offsetList) # use same parameters for training and testing
    testSet.load_vectors(file3) # no need to load y values for testing data
    
    # Predict for unknown data
    file3out = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_3b\Run1c_swtesting\SVM_Out.csv'
    print "Predicting for the test data"
    testSet.predict_svm(svmModel,file3out)
