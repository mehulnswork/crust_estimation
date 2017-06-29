# -*- coding: utf-8 -*-
def func(input_geotiff,split_size,path_tiff_splits_info):

    from osgeo import gdal
    gdal.UseExceptions()
    import time
    import sys    
    
    TIME_START = time.time()    
    src_ds = gdal.Open(input_geotiff)
    RASTER_NO = 1   
        
    srcband      = src_ds.GetRasterBand(RASTER_NO)
    width        = src_ds.RasterXSize
    height       = src_ds.RasterYSize
    bands        = src_ds.RasterCount
    driver       = src_ds.GetDriver().LongName
    geotransform = src_ds.GetGeoTransform()
    top_leftY    = geotransform[0]
    top_leftX    = geotransform[3]
    wepixelres   = geotransform[1]
    nspixelres   = geotransform[5]
    bandtype     = gdal.GetDataTypeName(srcband.DataType)
   
    # Coordinate system used for data processing
    # X - Northwards, Y - Eastwards, Z - Downwards
    # X - Rows of dataset, Y - Columns of dataset
    # Xres - N-S resolution, Yres - E-W resolution    
   
    print('-------------- GeoTiff files -----------------')
    
    print 'RASTOR NO         >> ' +  str(RASTER_NO)    
    print 'NO DATA           >> ' +  str(srcband.GetNoDataValue())
    print 'MINIMUM VALUE     >> ' +  str(srcband.GetMinimum()) + ' Maximum value >> ' +  str(srcband.GetMaximum())
    print 'SCALE             >> ' +  str(srcband.GetScale()) +  ' Unit type >> ' +  str(srcband.GetUnitType())
    print 'WIDTH(Y)          >> ' +  str(width) + ' HEIGHT(X) >> ' + str(height) + ' ALL BANDS >> ' + str(bands) 
    print 'FILE FORMAT       >> ' +  driver     
    print 'TOP LEFT(X,Y)     >> ' +  '(' + str(top_leftX) + ','+ str(top_leftY) + ')'    
    print 'W-E(Y) RESOLUTION >> ' +  str(wepixelres)
    print 'N-S(X) RESOLUTION >> ' +  str(nspixelres)    
    print 'DATA TYPE         >> ' +  bandtype   

    #%% Calculating Xstart Xend XSize and YSize for data extraction
    
    XRES     = float(nspixelres)
    YRES     = float(wepixelres)
    XRANGE = height
    YRANGE = width
    
    #finding bins from 0 to start so that they can be removed from 0 to end

    TIME_GEOTIFF = time.time()
    print 'Time required >> ' + str(TIME_GEOTIFF - TIME_START) + ' seconds'

    XMAPBINS  = range(0,XRANGE) 
    YMAPBINS  = range(0,YRANGE) 

    XBINS_POS = []
    YBINS_POS = []
    
    for i in range(0,XRANGE):
       XBINS_POS.append((float(XMAPBINS[i]) * XRES) + float(top_leftX))

    for i in range(0,YRANGE):
       YBINS_POS.append((float(YMAPBINS[i]) * YRES) + float(top_leftY)) 
       
    XMAPSTART = 0
    XMAPEND   = int(abs(XRES) * XRANGE)
    YMAPSTART = 0
    YMAPEND   = int(abs(YRES) * YRANGE)
    
    TOTAL_SPLITS = len(range(XMAPSTART,XMAPEND,split_size)) * len(range(YMAPSTART,YMAPEND,split_size))

    count_split = 0
    check_perc  = 0
    check_perc_minor  = 0
    
    
    file_tiff_info = open(path_tiff_splits_info,'w')
    
    print('Getting Geotiff splits info\n')
    
    for xl in range(XMAPSTART,XMAPEND,split_size):
        for ym in range(YMAPSTART,YMAPEND,split_size): 

            count_split += 1            
            curr_perc = float(count_split)/float(TOTAL_SPLITS) * 100.0
            
            if curr_perc >= check_perc:
                sys.stdout.write(str(int(check_perc)))
                check_perc = check_perc + 10.0
                sys.stdout.flush()

            if curr_perc >= check_perc_minor:
                sys.stdout.write('.')
                check_perc_minor = check_perc_minor + 2.0
                sys.stdout.flush()
            
                
            X1_START = 0
            X1_END   = xl      
            X1BINEND = int(( X1_END - X1_START)/abs(XRES))   
         
            X2_START = 0
            X2_END   = xl + split_size
            X2BINEND = int(( X2_END - X2_START)/abs(XRES))
        
            START_X = X1BINEND + 1
            SIZE_X  = X2BINEND - X1BINEND
                    
            Y1_START = 0
            Y1_END   = ym   
            Y1BINEND = int(( Y1_END - Y1_START)/abs(YRES))
           
         
            Y2_START = 0
            Y2_END   = ym + split_size
            Y2BINEND = int(( Y2_END - Y2_START)/abs(YRES))
        
            START_Y = Y1BINEND + 1
            SIZE_Y  = Y2BINEND - Y1BINEND
            
            START_X_M = START_X * XRES + (top_leftX)
            START_Y_M = START_Y * YRES + (top_leftY)

            file_tiff_info.write('X' + str(xl) + '-' + str(xl + split_size) + '_Y' + str(ym) + '-' + str(ym + split_size) + '_' + str(xl) + '_' + str(SIZE_Y) + '_' + str(ym) + '_' + str(SIZE_Y) + '.jpeg')
            file_tiff_info.write(',' + str(START_X) + ',' + str(SIZE_X) + ',' + str(START_Y) + ',' + str(SIZE_Y))
            file_tiff_info.write(',' + str(START_X_M) + ',' + str(START_Y_M) + '\n')
                
    file_tiff_info.close()
    TIME_JPEG = time.time()
    print '>> Time required is ' + str(TIME_JPEG - TIME_GEOTIFF) + ' seconds'
   
   
    return None  