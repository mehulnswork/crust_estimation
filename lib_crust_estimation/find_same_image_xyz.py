# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:13:44 2017

@author: oplab
"""

def func(dir_splits, path_namelist):
    
    import os
    d = os.listdir(dir_splits)
    image_list = [ s for s in d if '.jpeg' in s]
    print('Number of JPEG files found:' + str(len(image_list)))

    xyz_split_list = [ s for s in d if '.txt' in s]
    print('Number of TXT files found:' + str(len(xyz_split_list)))
    
    file_namelist = open(path_namelist,'w')
    file_namelist.write('image_path, xyz_path\n')

    for index in range(len(image_list)):
        path_image = dir_splits + '/' + image_list[index]        
        make_str   =  image_list[index].split('.')[0]
        search_str = make_str.split('_')[0] + '_' + make_str.split('_')[1]

        d = [ s for s in xyz_split_list if search_str in s]
        path_xyz = dir_splits + '/' + d[0]    
        
        file_namelist.write(str(path_image) + ',' + str(path_xyz) + '\n')
        
    file_namelist.close()
    
    return None