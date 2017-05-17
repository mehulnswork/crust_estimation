# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:58:33 2017

@author: oplab
"""

def dir_make_if_none_list(name_list_of_dirs):
    
    import os
    
    for i in range(len(name_list_of_dirs)):
        print('Creating ' + name_list_of_dirs[i])
        dir_name = name_list_of_dirs[i]
        if os.path.isdir(dir_name) is False:
            os.makedirs(dir_name)
            
    return None
    
def dir_delete_files(name_list_of_dirs):

    import glob
    import os

    for i in range(len(name_list_of_dirs)):
        
        print('Removing ' + name_list_of_dirs[i])
        dir_name = name_list_of_dirs[i]        

        if os.path.isdir(dir_name):
            files = glob.glob(dir_name + '/' + '*')    
            for f in files:
                os.remove(f)

    return None    
    
def join_str_for_filename(name_entries):
    
    path_filename = '_'.join(name_entries)    
    return path_filename