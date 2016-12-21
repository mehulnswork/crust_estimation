# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:34:49 2016
Predicting the split image properties for Cruse estimation 
@author: Sangekar Mehul

"""

 #%%



 #%%

def get_image_xyz_filenames(dir_splits, image_split_list, xyz_split_list, index):
    path_image = dir_splits + '/' + image_split_list[index]
    search_str =  image_split_list[index].split('.')[0]
    print search_str
    d = [ s for s in xyz_split_list if search_str in s]  
    path_xyz = dir_splits + '/' + d[0]
    return(path_image, path_xyz)

def get_split_imagelist(dir_splits):
    import os
    d = os.listdir(dir_splits)
    split_list = [ s for s in d if '.jpeg' in s]
    print('Number of JPEG files found:' + str(len(split_list)))
    return split_list    

def get_split_xyzlist(dir_splits):
    import os
    d = os.listdir(dir_splits)
    split_list = [ s for s in d if '.txt' in s]
    print('Number of txt files found:' + str(len(split_list)))
    return split_list    

 #%%
def get_block_stats(dir_splits, image_split_list, xyz_split_list, path_blockstats):

    file_blockstats = open(path_blockstats,'w')
    
    for index in range(10):
        path_image, path_xyz = get_image_xyz_filenames(dir_splits, image_split_list, xyz_split_list, index)

        
        cv2_img, img_height, img_width =  load_opencv_image(path_image)
        avg_r, avg_g, avg_b = get_avg_rgb(cv2_img)
        avg_h, avg_s, avg_v = get_avg_hsv(cv2_img)
        points_x, points_y, points_z   =  load_xyz(path_xyz)   
        mean_z, std_z = get_xyz_moments(points_x, points_y, points_z)

        file_blockstats.write(path_image + ',')
        file_blockstats.write(str(avg_r) + ',')
        file_blockstats.write(str(avg_g) + ',')
        file_blockstats.write(str(avg_b))
        file_blockstats.write('\n')    

    file_blockstats.close()    
    return None

 #%%

def load_xyz(path_xyz):
    import numpy as np
    points_x = np.array([])
    points_y = np.array([])
    points_z = np.array([])
    
    file_xyz = open(path_xyz,'r')    
    
    for line in file_xyz.readlines():
        d = line.split()
        if(float(d[2]) > 0):
            points_x = np.append(points_x,float(d[0]))
            points_y = np.append(points_y,float(d[1]))
            points_z = np.append(points_z,float(d[2]))
    
    file_xyz.close()        
    return(points_x, points_y, points_z)

def get_xyz_moments(points_x, points_y, points_z):    
    import numpy as np
    mean_z = np.mean(points_z)
    std_z  = np.std(points_z)
    return(mean_z, std_z)

 #%%

def load_opencv_image(path_image):
    import cv2
    cv2_image  = cv2.imread(path_image)
    img_width, img_height, channels = cv2_image.shape
    return(cv2_image, img_height, img_width)
    
def get_avg_rgb(image):
    import numpy as np
    avg_row   = np.average(image, axis=0)    
    avg_color = np.average(avg_row, axis=0)
    avg_r = avg_color[0]
    avg_g = avg_color[1]
    avg_b = avg_color[2]    
    return(avg_r, avg_g, avg_b)    

def get_avg_hsv(image):
    import numpy as np
    import cv2
    hsv       = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    avg_row   = np.average(hsv, axis=0)    
    avg_hsv   = np.average(avg_row, axis=0)
    avg_h = avg_hsv[0]
    avg_s = avg_hsv[1]
    avg_v = avg_hsv[2]    
    return(avg_h, avg_s, avg_v)       
   

#%% Reading blocks from folder

if __name__ == '__main__':
    try:        
        dir_splits = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/Splits'
        path_blockstats = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/block_stats.txt'
        
        image_split_list = get_split_imagelist(dir_splits)
        xyz_split_list   = get_split_xyzlist(dir_splits)
        get_block_stats(dir_splits, image_split_list, xyz_split_list, path_blockstats)
        
        import crust_svm as csvm
        
        para_train_list, labels_train_list, para_predict_list, labels_verify_list = csvm.create_iris_dataset()
        svc = csvm.create_svm()
        print('Correct labels:' + str(labels_verify_list))
        
        csvm.train_crust_images(para_train_list, labels_train_list, svc)
        labels_predict_list = csvm.predict_crust_images(para_predict_list, svc)
        
        print('SCM labels:' + str(labels_predict_list))

    except BaseException as e:
        print(e)
        raise
    finally:
        print('Execution Completed')

#%% Extracting parameters from each block for analysis



#%% Unsupervised clustering of blocks into groups








