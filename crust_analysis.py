# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:34:49 2016
Predicting the split image properties for Cruse estimation
@author: Sangekar Mehul

New library

"""

#%%


import sys
import argparse
sys.path.append('/lib_crust_estimation')
sys.path.append('/lib_m')


if __name__ == '__main__':
    try:

        sys.path.append('/home/oplab/sources/ply_to_geotiff')
        
        import lib_m as m
        import lib_crust_estimation as f
        import ply_to_geotiff as p2g  
        import sys
        
        ap = argparse.ArgumentParser()
        ap.add_argument("-l", "--limits",     required=True, help="sX-eX-sY-eY")
        ap.add_argument("-i", "--dir_splits", required=True, help="path to the tiff block")
        ap.add_argument("-o", "--dir_output", required=True, help="path to the output processing folder")
        ap.add_argument("-t", "--dir_training", required=True, help="path to training data set")
        
        args = vars(ap.parse_args())
        
        block_limits    =  args["limits"]
        dir_splits_full = args["dir_splits"]
        dir_processing  = args["dir_output"]
        dir_training    = args["dir_training"]
        
        print block_limits
        print dir_splits_full
        print dir_processing
        
        d = block_limits.split('-')
        
        startXbin = int(d[0])
        endXbin   = int(d[1])
        startYbin = int(d[2])
        endYbin   = int(d[3])                

        str_filename = m.dir.join_str_for_filename([str(startXbin), str(endXbin), str(startYbin), str(endYbin)])
        
        dir_processing_block  = dir_processing + '/' + str_filename
        dir_splits_block      = dir_processing_block +  '/splits_selected'        
        dir_splits_info       = dir_processing_block +  '/splits_info'

        path_namelist         = dir_splits_info + '/split_matched_image_xyz_names.csv'
        path_blockstats       = dir_splits_info + '/split_statistics.csv'
        path_resultlist       = dir_splits_info + '/splits_classfied_kmeans.csv'
        path_tiff_splits_info = dir_splits_info + '/full_geotiff_splits_position.csv'

        path_depth_tiff       = '/home/oplab/cruiseData/NT15-03/HPD1780/Analysis/geotiff/depth.tiff'

        dir_restiffs_block    = dir_processing_block +  '/splits_tiffs'
        dir_output_tiff       = dir_processing +  '/block_tiffs'
        path_init             = '/home/oplab/cruiseData/NT15-03/HPD1780/Analysis/classification.init'

        num_clusters = 5
        split_size   = 1
        
        
        path_vrt              = dir_output_tiff + '/blocks_' + str_filename + '.vrt'
        path_output_tiff      = dir_output_tiff + '/blocks_' + str_filename + '.tif'

        m.dir.dir_make_if_none_list([dir_processing_block, dir_splits_block, dir_splits_info, dir_restiffs_block, dir_output_tiff])
        
        total_files = f.make_blocks_between_XY_limits.func(dir_splits_full, dir_splits_block, startXbin, endXbin, startYbin, endYbin)        

        if total_files > 0:
            
            f.get_same_image_xyz_filenames.func(dir_splits_block, path_namelist)        
            
            #f.get_blocks_stats.func(dir_splits_block, path_blockstats, path_namelist)
    
            #f.classify_blocks_stats_kmeans.func(path_blockstats, num_clusters, path_namelist, path_resultlist)
            
            f.classify_blocks_texture_linearsvm.func(dir_training, dir_splits_block, path_resultlist, path_init)
        
            f.get_blocks_XY_from_geotiff.func(path_depth_tiff,split_size,path_tiff_splits_info)
    
            f.make_blocks_classified_tiffs.func(path_resultlist, dir_restiffs_block, path_tiff_splits_info, path_depth_tiff)
    
            p2g.join_tiffs_vrt(dir_restiffs_block, path_vrt, path_output_tiff)
                

    except BaseException as e:
        print(e)
        raise
    finally:
        
        print('Execution Completed')

#%% Unsupervised clustering of blocks into groups
