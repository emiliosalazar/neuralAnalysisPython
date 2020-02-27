"""
Spyder Editor

This is a temporary script file.
"""
import sys
import numpy
import mysql.connector

def GenerateDataPaths(animal, task):
    
    cnx = mysql.connector.connect(user='rs350044', password='$milelab44',
                                  host='skullcap.sni.pitt.edu',
                                  database='smiledb')
    
    cursor = cnx.cursor()
    
    animal = 'Earl'
    
    query = ("select ExperimentDate, Task from monkeyLog where Subject = '{}'".format(animal))
    cursor.execute(query)
    
    pathSet = []
    expDateSet = []
    
    for ExperimentDate, Task in cursor:
        if Task.find(task) != -1:
            yr = numpy.floor(ExperimentDate/10000)
            mn = '{:02.0f}'.format(numpy.floor(ExperimentDate/100) - yr*100)
            #print('yr: {}, mn: {}, dy: {}'.format(yr,mn,dy))
            pathSet.append('Animals/{}/{:.0f}/{}/{:.0f}'.format(animal, yr, mn, ExperimentDate))
            expDateSet.append('{:.0f}'.format(ExperimentDate))
    
    cursor.close()
    cnx.close()
    
    return pathSet, expDateSet
