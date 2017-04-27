# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:09:20 2017

@author: oplab
"""
def gdalinfo_latlon(input_file):
    
    import subprocess
    import re
    
    str_gdal_info = subprocess.check_output('gdalinfo {}'.format(input_file), shell=True)
    str_gdal_info = str_gdal_info.split('/n') # Creates a line by line list.
    for line in str_gdal_info:
         d = line.split('\n')
                
    for line in d:
        if 'Upper Left' in line:
            k = line.split('(') 
            kk = k[2].split(')')
            e = re.findall(r'\d+', kk[0])
            tiff_lat = float(e[0]) + float(e[1])/60  + float(e[2])/3600 + float(e[3])/360000
            tiff_lon = float(e[4]) + float(e[5])/60  + float(e[6])/3600 + float(e[7])/360000
    return(tiff_lat, tiff_lon)
    
def func(path_resultlist, dir_result_tiffs, path_tiff_splits_info, input_depth):
    
    import re
    import numpy as np
    from   osgeo import osr,gdal
    import glob
    import os

    src_ds = gdal.Open(input_depth)
    #RASTER_NO = 1   
        
    #srcband      = src_ds.GetRasterBand(RASTER_NO)
    #width        = src_ds.RasterXSize
    #height       = src_ds.RasterYSize
    #bands        = src_ds.RasterCount
    driver       = src_ds.GetDriver().LongName
    geotransform = src_ds.GetGeoTransform()
    #top_leftY    = geotransform[0]
    #top_leftX    = geotransform[3]
    wepixelres   = geotransform[1]
    nspixelres   = geotransform[5]
    #bandtype     = gdal.GetDataTypeName(srcband.DataType)
    ew_origin_deg,ns_origin_deg      = gdalinfo_latlon(input_depth) 

    files = glob.glob(dir_result_tiffs + '/' + '*')
    
    for f in files:
        os.remove(f)
    
    file_tiff_splits_info = open(path_tiff_splits_info,'r')    
    tiff_line = file_tiff_splits_info.readlines()        
    file_tiff_splits_info.close()   
    
    
    file_resultlist = open(path_resultlist,'r')    
    line = file_resultlist.readline()
    
    img_name = []
    img_type = []
    
    for line in file_resultlist.readlines():
        d = line.split(',')    
        img_name.append(d[0])    
        img_type.append(int(d[1]))
    
    file_resultlist.close()

    img_name_info = []
    xpos = []
    ypos = []
    
    for i in range(len(img_type)):
        ds = img_name[i].split('/')
        d = re.findall(r'(\d+)',ds[-1])
        fname = ds[-1]
        #startX = int(d[4])
        #startY = int(d[6])
        sizeX = int(d[5])
        sizeY = int(d[7])     
    
        path_output_tiff = dir_result_tiffs  + '/part' + str(i) + '.tif'
    
        
        d  = [s for s in tiff_line if fname in s]
        
        d = d[0].split(',')
        img_name_info.append(d[0])
        xpos = float(d[5])
        ypos = float(d[6])
        
        if img_type[i] == 0:
            points = np.full((sizeY,sizeX), 1.0, dtype=np.float32)
        if img_type[i] == 1:
            points = np.full((sizeY,sizeX), 2.0, dtype=np.float32)
        if img_type[i] == 2:
            points = np.full((sizeY,sizeX), 3.0, dtype=np.float32)
        if img_type[i] == 3:
            points = np.full((sizeY,sizeX), 4.0, dtype=np.float32)
        if img_type[i] == 4:
            points = np.full((sizeY,sizeX), 5.0, dtype=np.float32)
    
        ew_upper_left = ypos
        ns_upper_left = xpos   # x_min & y_max are like the "top left" corner.
        
        driver = gdal.GetDriverByName('GTiff')
        gdal_mosaictiff = driver.Create(path_output_tiff, sizeY, sizeX, 1, gdal.GDT_Byte, options = ['COMPRESS=DEFLATE'])
        gdal_mosaictiff.SetGeoTransform([ew_upper_left, wepixelres, 0, ns_upper_left, 0, nspixelres])  
    
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('WGS84')
        srs.SetTM(ns_origin_deg,ew_origin_deg , 1, 0, 0)
        gdal_mosaictiff.SetProjection(srs.ExportToWkt())
    
        intenbandr = gdal_mosaictiff.GetRasterBand(1)
        intenbandr.WriteArray(np.transpose(points))
        intenbandr.SetNoDataValue(0)
    
        gdal_mosaictiff.FlushCache()  # Write to disk.        