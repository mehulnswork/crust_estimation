# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:28:07 2017

@author: oplab
"""

def func(path_qmlstyle, path_init):
    
    
    array_colors = []
    array_values = []
    array_labels = []

       
    file_init = open(path_init,'r')
    for line in file_init.readlines():
        d = line.split(',')
        array_labels.append(d[0])
        array_values.append(int(d[1]))        
        array_colors.append(d[2].strip())
    
    num_values = len(array_values)
    
    file_init.close()
        
    file_qmlstyle = open(path_qmlstyle,'w')
    file_qmlstyle.write('<!DOCTYPE qgis PUBLIC "'"http://mrcc.com/qgis.dtd"'" "'"SYSTEM"'">\n')
    file_qmlstyle.write('<qgis version="">\n')
    file_qmlstyle.write('  <pipe>\n')
    
    file_qmlstyle.write('    <rasterrenderer opacity="' + str(1) + '"') 
    file_qmlstyle.write(' alphaBand="' + str(0) + '"')  
    file_qmlstyle.write(' classificationMax="' + str(num_values) + '"') 
    file_qmlstyle.write(' classificationMinMaxOrigin="CumulativeCutFullExtentEstimated"') 
    file_qmlstyle.write(' band="' + str(1) + '"') 
    file_qmlstyle.write(' classificationMin="' + str(1) + '"') 
    file_qmlstyle.write(' type="singlebandpseudocolor">\n')

    file_qmlstyle.write('      <rasterTransparency/>\n')
    file_qmlstyle.write('      <rastershader>\n')

    file_qmlstyle.write('        <colorrampshader colorRampType="INTERPOLATED" clip="0">\n')
    
    for i in range(num_values):
        file_qmlstyle.write('          <item alpha="255" value="' + str(array_values[i]) + '" label="' + str(array_labels[i]) + '" color="' + str(array_colors[i]) + '"/>\n')

    file_qmlstyle.write('        </colorrampshader>\n')
    
    
    file_qmlstyle.write('      </rastershader>\n')
    file_qmlstyle.write('    </rasterrenderer>\n')
    file_qmlstyle.write('    <brightnesscontrast />\n')
    file_qmlstyle.write('    <huesaturation/>\n')
    file_qmlstyle.write('    <rasterresampler/>\n')
    file_qmlstyle.write('  </pipe>\n')
    file_qmlstyle.write('  <blendMode>0</blendMode>\n')
    file_qmlstyle.write('</qgis>')

    file_qmlstyle.close()
    
    return None
    
    
    



  
  
