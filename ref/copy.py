# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:40:38 2017

@author: oplab
"""

import sys
sys.path.append('/home/oplab/sources/crust_estimation')

import lib_crust_estimation as f

dir_splits_full       = '/media/oplab/CRCBOSSA/cruiseData/NT15-03/Publish/splits_all'
dir_splits_block      = '/home/oplab/Downloads'

f.select_split_files_from_limits.func(dir_splits_full, dir_splits_block, 0, 200, 400, 500)        



