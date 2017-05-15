# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:34:49 2016
Predicting the split image properties for Cruse estimation
@author: Sangekar Mehul

New library

"""

#%%


import sys
sys.path.append('/lib_crust_estimation')
sys.path.append('/lib_m')


if __name__ == '__main__':
    try:

        sys.path.append('/home/oplab/sources/ply_to_geotiff')

        import lib_m as m
        import lib_crust_estimation as f
        import ply_to_geotiff as p2g  
        import sys
        
        startXbin    = 400
        endXbin      = 500
        startYbin    = 300
        endYbin      = 400
        num_clusters = 5
        split_size   = 1
                
        
        dir_splits_full       = '/media/oplab/CRC BOSS-A DATA/cruiseData/NT15-03/Publish/splits_all'
        dir_splits_select     = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/splits_selected'
        dir_splits_info       = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/splits_info'
        path_namelist         = dir_splits_info + '/split_matched_image_xyz_names.csv'
        path_blockstats       = dir_splits_info + '/split_statistics.csv'
        path_resultlist       = dir_splits_info + '/splits_classfied_kmeans.csv'
        path_tiff_splits_info = dir_splits_info + '/full_geotiff_splits_position.csv'
        path_depth_tiff       = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/geotiff/depth.tiff'

        dir_result_tiffs      = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/splits_tiffs'
        dir_output_tiff       = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/full_classified_tiff'
        path_vrt              = dir_output_tiff + '/all_blocks.vrt'
        path_output_tiff      = dir_output_tiff + '/all_blocks.tif'
        dir_training          = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/LBP/training'

        #m.dir.dir_make_if_none_list([dir_splits_select, dir_splits_info, dir_result_tiffs, dir_output_tiff])
        
        #f.select_split_files_from_limits.func(dir_splits_full, dir_splits_select, startXbin,endXbin,startYbin,endYbin)        
        #f.find_same_image_xyz.func(dir_splits_select, path_namelist)        
        
        
        #f.get_block_stats.func(dir_splits_select, path_blockstats, path_namelist)
        #f.classify_blocks.func(path_blockstats, num_clusters, path_namelist, path_resultlist)
        
        #f.classify_blocks_from_texture.func(dir_training, dir_splits_select, path_resultlist)
    
        f.geotiff_split_info.func(path_depth_tiff,split_size,path_tiff_splits_info)
        f.make_tiff_blocks_for_classes.func(path_resultlist, dir_result_tiffs, path_tiff_splits_info, path_depth_tiff)


        p2g.join_tiffs_vrt(dir_result_tiffs, path_vrt, path_output_tiff)

    except BaseException as e:
        print(e)
        raise
    finally:
        
        print('Execution Completed')

#%% Unsupervised clustering of blocks into groups
