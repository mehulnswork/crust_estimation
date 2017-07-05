# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:54:46 2017

@author: oplab
"""

def get_split_locations(full_startXbin, full_endXbin, full_startYbin, full_endYbin, block_Xsize, block_Ysize):
    
    start_Xbins = []
    end_Xbins   = []
    start_Ybins = []
    end_Ybins   = []
    
    for i in range(full_startXbin ,full_endXbin ,block_Xsize):
        for  j in range(full_startYbin,full_endYbin,block_Ysize):
            start_Xbins.append(i)
            end_Xbins.append(i + block_Xsize)               
            start_Ybins.append(j)
            end_Ybins.append(j + block_Ysize)       
            
    num_blocks =  len(start_Xbins)
    
    print('Splitting the map into ' + str(num_blocks) + ' blocks\n')
    return(start_Xbins, end_Xbins, start_Ybins, end_Ybins, num_blocks)
