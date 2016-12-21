# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:39:11 2015

@author: Umesh Neettiyath
A library containing functions related to SVM classification
Class svm_data_class is needed for this 

Ref : http://scikit-learn.org/stable/modules/svm.html

v1.0 : First version made from
        svm_trial2f.py
        accuracy_check.py
"""

import numpy as np
from svm_data_class import SvmVector
from sklearn.externals import joblib
#from sklearn import svm
#import sys
#import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Times New Roman')
#matplotlib.use('pgf')
#pgf_with_rc_fonts = {
#    "font.family": "serif",
#    "font.serif": [],                   # use latex default serif font
#    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
#}
#matplotlib.rcParams.update(pgf_with_rc_fonts)

#%% make ply file with coloured cubes showing SVM result
def make_ply_cubes(svmData,outPath):
    'make ply file with coloured cubes showing SVM result'
    depthoffset = -0.05
    d = 0.05
    height = d
    nPoints = svmData.length
        
    with open(outPath,'w') as f:
        f.seek(0)
        f.write('ply\nformat ascii 1.0\ncomment author: Umesh Neettiyath\n')
        f.write('element vertex %d\n'%(nPoints*8))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('element face %d\n'%(nPoints*6))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        
        for i in range(nPoints):
            x,y,z = svmData.location[i]
            z -= depthoffset
#            colorIndex = int(255*(thick[i]-tmin)/(tmax-tmin))
#            print colorIndex
#            r = int(cm.get_cmap(colormap)(colorIndex)[0]*255)
#            g = int(cm.get_cmap(colormap)(colorIndex)[1]*255)
#            b = int(cm.get_cmap(colormap)(colorIndex)[2]*255)
            if svmData.prediction[i] == 1:#1-crust
                r,g,b = 0,255,0
            elif svmData.prediction[i] == 0:#-1-sand
                r,g,b = 255,0,0
            else:
                r,g,b = 0,0,255
            
            f.write('%f %f %f %d %d %d\n'%(x-d/2, y-d/2, (z-height), r, g, b))
            f.write('%f %f %f %d %d %d\n'%(x-d/2, y-d/2, z, r, g, b))
            f.write('%f %f %f %d %d %d\n'%(x-d/2, y+d/2, z, r, g, b))
            f.write('%f %f %f %d %d %d\n'%(x-d/2, y+d/2, (z-height), r, g, b))
            f.write('%f %f %f %d %d %d\n'%(x+d/2, y-d/2, (z-height), r, g, b))
            f.write('%f %f %f %d %d %d\n'%(x+d/2, y-d/2, z, r, g, b))
            f.write('%f %f %f %d %d %d\n'%(x+d/2, y+d/2, z, r, g, b))
            f.write('%f %f %f %d %d %d\n'%(x+d/2, y+d/2, (z-height), r, g, b))

        for i in range(nPoints):
            j=i*8
            f.write('4 %d% d% d% d\n'%(j+0,j+1,j+2,j+3))
            f.write('4 %d% d% d% d\n'%(j+7,j+6,j+5,j+4))
            f.write('4 %d% d% d% d\n'%(j+0,j+4,j+5,j+1))
            f.write('4 %d% d% d% d\n'%(j+1,j+5,j+6,j+2))
            f.write('4 %d% d% d% d\n'%(j+2,j+6,j+7,j+3))
            f.write('4 %d% d% d% d\n'%(j+3,j+7,j+4,j+0))
            
#%% Accuracy check                 
def accuracy_check(fileManual,discLocationsFile,outPath):
    'Check the performance of classifier by comparing with manually classified data'
#    outFile = os.path.join(basePath, 'SVM_out.csv')
    pass
   
#%% PlotSVM 
def plot_2features(testSet,svmModel,outPath):
    plt.figure(1)
    plt.clf()
    plt.scatter(testSet.vector[:,0], testSet.vector[:, 1], c=testSet.prediction, zorder=10, cmap=plt.cm.Paired)
    
    # Circle out the test data
#        plt.scatter(X[:, 0]*testSet.scaleFactors[0], X[:, 1]*testSet.scaleFactors[1], s=80, facecolors='none', zorder=10)
    
    plt.xlabel(testSet.features[0])
    plt.ylabel(testSet.features[1])
    plt.axis('tight')
    x_min = testSet.vector[:, 0].min()
    x_max = testSet.vector[:, 0].max()
    y_min = testSet.vector[:, 1].min()
    y_max = testSet.vector[:, 1].max()
    
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = svmModel.decision_function(np.c_[XX.ravel(), YY.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX*testSet.scaleFactors[0], YY*testSet.scaleFactors[1], Z > 0, cmap=plt.cm.autumn,rasterized=True)
    plt.contour(XX*testSet.scaleFactors[0], YY*testSet.scaleFactors[1], Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])
    
#    plt.title('poly')
    plt.savefig(outPath, bbox_inches='tight', pad_inches=0.1)
    plt.show()
        
def plot_3features(testSet,svmModel,outPath):     
    
    x_min = testSet.vector[:, 0].min()
    x_max = testSet.vector[:, 0].max()
    y_min = testSet.vector[:, 1].min()
    y_max = testSet.vector[:, 1].max()
    z_min = testSet.vector[:, 2].min()
    z_max = testSet.vector[:, 2].max()
    rangeZ = [0,5,10,12,14,16,19]
    lenZ = len(rangeZ)
    divZ = 20
    lenZcx = complex(0,divZ)
    
    XX, YY, ZZ = np.mgrid[x_min:x_max:100j, y_min:y_max:100j, z_min:z_max:lenZcx]
#    AA = ZZ*0 + 56.0/90
    Z = svmModel.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(1, figsize=(6,8), facecolor='w', edgecolor='k')
    plt.clf()
    #        plt.rc('text', usetex=True)
    #        plt.rc('font', family='serif')
    #        plt.rc('font', serif='Times New Roman')

    for idx in range(lenZ-1):#enumerate(rangeZ):#range(4):
    #            if idx == lenZ-1:
    #                break
    #            plt.subplot(2,2,indZ+1)
        indZ = rangeZ[idx]
        print idx,indZ
        plt.subplot(3,2,idx+1)
        
        # select values in the range of indZ to indZ +1
        ind_1 = testSet.vector[:,2] <= z_max*(rangeZ[idx+1])/divZ
        ind_2 = testSet.vector[:,2] > z_max*indZ/divZ
        ind = np.logical_and(ind_1,ind_2)
        X_ = [val for i,val in zip(ind,testSet.vector) if i]
        Y_ = [val for i,val in zip(ind,testSet.prediction) if i]
        X_ = np.asarray(X_)
        Y_ = np.asarray(Y_)

        if sum(ind):
            plt.scatter(X_[:, 0]*testSet.scaleFactors[0], X_[:, 1]*testSet.scaleFactors[1], c=Y_, zorder=10, cmap=plt.cm.prism_r)
        else:
            plt.scatter(testSet.vector[0, 0], testSet.vector[0, 1], c=testSet.prediction[0], zorder=0, cmap=plt.cm.prism_r)
    
    # Circle out the test data
#            plt.scatter(X[:, 0]*testSet.scaleFactors[0], X[:, 1]*testSet.scaleFactors[1], s=80, facecolors='none', zorder=10)
    
        plt.xlabel(paramList[0])
        plt.ylabel(paramList[1])
        plt.axis('tight')
        plt.pcolormesh(XX[:,:,indZ]*testSet.scaleFactors[0], YY[:,:,indZ]*testSet.scaleFactors[1], Z[:,:,indZ] > 0, cmap=plt.cm.Pastel1_r,rasterized=True)#cm.cool
        plt.contour(XX[:,:,indZ]*testSet.scaleFactors[0], YY[:,:,indZ]*testSet.scaleFactors[1], Z[:,:,indZ], colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])
        
        plt.xlim(x_min*testSet.scaleFactors[0],x_max*testSet.scaleFactors[0])        
        plt.ylim(y_min*testSet.scaleFactors[1],y_max*testSet.scaleFactors[1])        
        plt.title('%4.2f $<$ %s $\leq$ %4.2f'%(ZZ[0,0,indZ]*testSet.scaleFactors[2],paramList[2],ZZ[0,0,rangeZ[idx+1]]*testSet.scaleFactors[2]))
    plt.tight_layout()
    #plt.Figure.set_size_inches(8.27, 11.69, forward=False)
    plt.savefig(outPath, bbox_inches='tight', pad_inches=0.1)
    plt.show() 

def plot_4features(testSet,svmModel,outPath):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_min = testSet.vector[:, 0].min()
    x_max = testSet.vector[:, 0].max()
    y_min = testSet.vector[:, 1].min()
    y_max = testSet.vector[:, 1].max()
    z_min = testSet.vector[:, 2].min()
    z_max = testSet.vector[:, 2].max()
    
    XX, YY, ZZ = np.mgrid[x_min:x_max:100j, y_min:y_max:100j, z_min:z_max:30j]
#    AA = ZZ*0 + 56.0/90
#    Z = svmModel.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel(), AA.ravel()])
    Z = svmModel.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    
    ax.scatter(testSet.vector[:, 0]*testSet.scaleFactors[0], testSet.vector[:, 1]*testSet.scaleFactors[1], testSet.vector[:, 2]*testSet.scaleFactors[2], c=testSet.prediction, zorder=10, cmap=plt.cm.Paired)
    Zheight = (np.sum(Z<0,axis=2))*testSet.scaleFactors[2]*(ZZ[0,0,1] - ZZ[0,0,0]) + ZZ[0,0,0]
    ax.plot_surface(XX[:,:,0]*testSet.scaleFactors[0],YY[:,:,0]*testSet.scaleFactors[1], Zheight, rstride=5, cstride=5, cmap=plt.cm.coolwarm)

#        for indZ in range(4):
#            ax.contour(XX[:,:,indZ]*testSet.scaleFactors[0], YY[:,:,indZ]*testSet.scaleFactors[1], Z[:,:,indZ], colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])
     
    plt.show()
#    ax.view_init(30, 120)
#        for angle in range(0, 360):
#            ax.view_init(30, angle)
#    plt.draw()
    plt.savefig(outPath, bbox_inches='tight', pad_inches=0.1)

def parse_manual_ply(fileManual,discLocationsFile,outPath,headertext = 'Manual_Prediction'):
    '''Compare prediction with manual classification. Find the accuracy of prediction'''

    from basic_utilities import read_ply_points
    nVertexOrig,pointsOrig,_temp,_temp = read_ply_points(discLocationsFile,outputList=['x', 'y', 'z'])
    nVertexMan,pointsMan,_temp,_temp = read_ply_points(fileManual,outputList=['x', 'y', 'z'])
    
    nMatch = 0
    output = np.zeros(nVertexOrig)
    lenB=nVertexMan
    for i in range(nVertexOrig):
        pointO = pointsOrig[i]
        pointO = [round(pointO[0],3),round(pointO[1],3),round(pointO[2],2)]
#        print pointO
        for j in range(lenB):
            if pointsMan[j] == pointO:
#                print j
                output[i] = 1
                nMatch += 1
                pointsMan.remove(pointO)
                lenB -= 1
                break

    np.savetxt(outPath,output,fmt='%d',delimiter=',', newline='\n', header=headertext,comments='')
    if nMatch == 0:
        raise RuntimeWarning("No Matches found. Please check the files.")
    if nMatch != nVertexMan:
        raise RuntimeWarning('Not all points could be matched')
    else:
        return nMatch
#    print nMatch,nVertexMan
#    print pointsMan
    
#%% Main Function
if __name__ == '__main__':
    
    # Parameters used for SVM. 
    # Can be of any length. Any column title in the input file can be used
    # parameter value used for classification = (parameter value - offset)/scaleFactor
#    paramList = ['Avg_Luminosity','Std_Dev_Luminosity','Entropy','Altitude_Orientation']
#    scaleFactorList = [255.0,100.0,5.0,90]
#    offsetList = [0,0,0,0]
    paramList = ['Avg_Luminosity','Std_Dev_Luminosity','Entropy']
    scaleFactorList = [255.0,100.0,5.0]
    offsetList = [0,0,0]

    # initialize
    print '\nInitialize class' 
    testSet = SvmVector(paramList,scaleFactorList,offsetList) # use same parameters for training and testing
    file3 = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_5\Run1\BrahmaLog_v5.0.csv'
    testSet.load_vectors(file3) # no need to load y values for testing data
    
    # Load SVM classifier model
    fileSVMModel = r'E:\3D_Coalition\Estimator_Models\7\SVM_Estimator.pkl'
    svmModel = joblib.load(fileSVMModel)# Loading SVM model

    # Predict for unknown data
    file3out = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_3b\Run1c_swtesting\SVM_Out.csv'
    print "Predicting for the test data"
    testSet.predict_svm(svmModel,file3out)

    # Compare prediction with manual classification. Find the accuracy of prediction
    fileManual = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_5\Run1\Disc_Locations_rounded_manual2.ply'
    discLocationsFile = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_5\Run1\Disc_Locations_rounded.ply'
    outPath = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_5\Run1\manual_classification.csv'
    parse_manual_ply(fileManual,discLocationsFile,outPath)

    # Plot the classifier output into a 3D ply file
    outPath = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_3b\Run1c_swtesting\SVM_Prediction.ply'
#    make_ply_cubes(testSet,outPath)

#    outPath = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_3b\Run1c_swtesting\plots\Classifier2.pdf'
#    plot_2features(testSet,svmModel,outPath)
#    plot_3features(testSet,svmModel,outPath)
    outPath = r'E:\3D_Coalition\HPD1543_Afternoon\Patch_3b\Run1c_swtesting\plots\Classifier3.pdf'
#    plot_4features(testSet,svmModel,outPath)    