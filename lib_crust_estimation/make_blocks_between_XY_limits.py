# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:06:10 2017

@author: oplab
"""

def func(dir_splits_full, dir_splits_select, startXbin,endXbin,startYbin,endYbin):

    import glob
    import shutil
    import os

    files = glob.glob(dir_splits_select + '/' + '*')
    
    for f in files:
        os.remove(f)

    print('Deleting files from > ' + str(dir_splits_select))
    
    dir_list = os.listdir(dir_splits_full)

    total_files = 0

    print('Copying files from > ' + str(dir_splits_full))    

    for i in range(startXbin,endXbin + 1):
        x_search_string = 'X' + str(i) + '-'
    
        dir_search_res_x = [s for s in dir_list if x_search_string in s]
        
        for j in range(startYbin,endYbin + 1):
            y_search_string = 'Y' + str(j) + '-'
            dir_search_res_y = [s for s in dir_search_res_x if y_search_string in s]

            total_files = total_files + len(dir_search_res_y)

            for k in range(len(dir_search_res_y)):
                path_source = dir_splits_full   + '/' + dir_search_res_y[k]
                path_dest   = dir_splits_select + '/' + dir_search_res_y[k]

                shutil.copy(path_source, path_dest)

    print('Total number of files copied >' + str(total_files))        
    
    return total_files