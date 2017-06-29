# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:34:55 2017

@author: oplab
"""

def func(path_blockstats, num_clusters, path_namelist, path_resultlist):
    
    import numpy as np
    from sklearn.cluster import KMeans

    file_statsfile = open(path_blockstats,'r')
    line = file_statsfile.readline()

    r = []
    g = []
    b = []
    h = []
    s = []
    v = []
    mz = []
    vz = []


    for line in file_statsfile.readlines():

	d = line.split(',')
	r.append(int(d[0]))
	g.append(int(d[1]))
	b.append(int(d[2]))
	h.append(int(d[3]))
	s.append(int(d[4]))
	v.append(int(d[5]))
	mz.append(float(d[6]))
	vz.append(float(d[7]))

    file_statsfile.close()

    print('Number of lines read:' + str(len(r)))

    param_matrix = np.array([r,g,b,mz,vz])
    param_matrix = param_matrix.transpose()

    kmeans_res = KMeans(n_clusters = num_clusters, random_state=0).fit(param_matrix)    
    print kmeans_res.labels_

    file_namesfile = open(path_namelist,'r')
    line = file_namesfile.readline()
	
    file_resultfile = open(path_resultlist,'w')
    file_resultfile.write('image, class\n')

    count = 0
    for line in file_namesfile.readlines():

	d = line.split(',')
	res_list = kmeans_res.labels_
	file_resultfile.write('%s,%d\n' %(d[0], res_list[count]))
	count = count + 1

    file_namesfile.close()
    file_resultfile.close()

    print('Number of lines read:' + str(count))
    
    return None
    
#import crust_svm as csvm

# para_train_list, labels_train_list, para_predict_list, labels_verify_list = csvm.create_iris_dataset()
# svc = csvm.create_svm()
# print('Correct labels:' + str(labels_verify_list))
# csvm.train_crust_images(para_train_list, labels_train_list, svc)
# labels_predict_list = csvm.predict_crust_images(para_predict_list, svc)
# print('SCM labels:' + str(labels_predict_list))
