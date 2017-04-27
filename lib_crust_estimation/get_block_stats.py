# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:17:23 2017

@author: oplab
"""

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
    
    mean_z = np.mean(points_z)
    std_z  = np.std(points_z)
    return(points_x, points_y, points_z, mean_z, std_z)


def load_python_image(path_image):
    from scipy import misc
    import numpy as np
    import matplotlib
    
    img = misc.imread(path_image) # 640x480x3 array    
    mat_image = np.array(img)
    mat_r = mat_image[:,:,0]
    mat_g = mat_image[:,:,1]
    mat_b = mat_image[:,:,2]
    
    avg_r = np.average(mat_r)
    avg_g = np.average(mat_g)
    avg_b = np.average(mat_b)

    mat_image_unity = mat_image/255.0    
    hsv_image = matplotlib.colors.rgb_to_hsv(mat_image_unity)
      
    mat_h = hsv_image[:,:,0]
    mat_s = hsv_image[:,:,1]
    mat_v = hsv_image[:,:,2]

    avg_h = np.average(mat_h)
    avg_s = np.average(mat_s)
    avg_v = np.average(mat_v)
    
    avg_h = avg_h * 360.0
    avg_s = avg_s * 100.0
    avg_v = avg_v * 100.0    
    
    return(avg_r, avg_g, avg_b, avg_h, avg_s, avg_v)    


def func(dir_splits,path_blockstats, path_namelist):

    file_namelist = open(path_namelist,'r')
    line = file_namelist.readline()
    
    path_image_list = []
    path_xyz_list   = []

    for line in file_namelist.readlines():
        d = line.split(',')
        path_image_list.append(d[0])
        path_xyz_list.append(d[1])        

    file_namelist.close()    


    file_blockstats = open(path_blockstats,'w')
    file_blockstats.write('red, green, blue, hue, satur, inten, mean z, std z\n')

    num_files = len(path_image_list)
    
    print('Numnber of files found ' + str(num_files) + '\n')


    for index in range(num_files):

        path_image = path_image_list[index].strip()
        path_xyz   = path_xyz_list[index].strip()
    
        avg_r, avg_g, avg_b, avg_h, avg_s, avg_v     = load_python_image(path_image)        
        points_x, points_y, points_z, mean_z, std_z  = load_xyz(path_xyz)

        file_blockstats.write('%d,%d,%d,' %(int(avg_r),int(avg_g),int(avg_b)))
        file_blockstats.write('%d,%d,%d,' %(int(avg_h),int(avg_s),int(avg_v)))
        file_blockstats.write('%f,%f' %(float(mean_z),float(std_z)))
        file_blockstats.write('\n')

    file_blockstats.close()
    return None