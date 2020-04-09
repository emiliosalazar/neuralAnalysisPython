#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:50:31 2020

@author: emilio
"""


#%% GPFA Simulations
gpfaSim = GPFAGenerator(numGPs = 5, endT = 1000)
outNeurTraj = gpfaSim.runSimulation(numSimulations=100)

outNeurTraj.gpfa(eng, "GPFA Sim", Path("/Users/emilio/Documents/BatistaLabData/gpfaTest"),
                 xDimTest = [2,5,8], timeBeforeAndAfterStart=None, timeBeforeAndAfterEnd=(-250,0))