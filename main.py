# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:09:02 2017

@author: oplab
"""


def callCommandLine(command):
    import os
    print 'Executing str:' + command
    os.system(command)    
    return None

import sys    
sys.path.append('/home/oplab/sources/ply_to_geotiff')
from   multiprocessing import Pool
import lib_m as m
import ply_to_geotiff as p2g  

if __name__ == '__main__':
    try:

        full_startXbin = 0
        full_endXbin   = 400 #1000
        full_startYbin = 0
        full_endYbin   = 400
        
        block_Xsize    = 200
        block_Ysize    = 200
        
        start_Xbins, end_Xbins, start_Ybins, end_Ybins, num_blocks = m.blocks.get_split_locations(full_startXbin, full_endXbin, full_startYbin, full_endYbin, block_Xsize, block_Ysize)  
         
        classify_str = []
            
        dir_splits_full       = '/media/oplab/CRCBOSSA/cruiseData/NT15-03/Publish/splits_all'
        dir_processing        = '/home/oplab/cruiseData/NT15-03/HPD1780/Analysis/classify'
        dir_training          = '/home/oplab/cruiseData/NT15-03/HPD1780/Analysis/training'
        path_init             = '/home/oplab/cruiseData/NT15-03/HPD1780/Analysis/classification.init'

        m.dir.dir_make_if_none_list([dir_processing])            

        for SX,EX,SY,EY in zip(start_Xbins, end_Xbins, start_Ybins, end_Ybins):
            
            str_filename = m.dir.join_str_for_filename([str(SX), str(EX), str(SY), str(EY)])
        
            cmd_str_exe = 'python crust_analysis.py'
            cmd_str_limits   = ' --limits '       + str(SX) + '-'  + str(EX) + '-' + str(SY) + '-'  + str(EY)    
            cmd_dir_splits   = ' --dir_splits '   + dir_splits_full    
            cmd_dir_process  = ' --dir_output '   + dir_processing
            cmd_dir_train    = ' --dir_training ' + dir_training
            cmd_str          = cmd_str_exe + cmd_str_limits + cmd_dir_splits + cmd_dir_process + cmd_dir_train
        
            classify_str.append(cmd_str)
        
        bathy_pool = Pool(processes = 3)
        bathy_pool.map(callCommandLine,classify_str, chunksize = 1)

        dir_restiffs_blocks   = dir_processing + '/block_tiffs'
        path_vrt              = dir_processing + '/full_map.vrt'
        path_output_tiff      = dir_processing + '/full_map.tif'
        path_qmlstyle         = dir_processing + '/full_map.qml'

        p2g.join_anytiffs_vrt(dir_restiffs_blocks, path_vrt, path_output_tiff)
        
        m.make_ggis_qmlstylefile.func(path_qmlstyle, path_init)

    except BaseException as e:
        print(e)
        raise
    finally:
        
        print('Execution Completed')
