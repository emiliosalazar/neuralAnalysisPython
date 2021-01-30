#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:23 2020

@author: emilio
"""

from methods.GeneralMethods import loadDefaultParams
from methods.GeneralMethods import saveFiguresToPdf
from matplotlib import pyplot as plt

from classes.Dataset import Dataset
from setup.DataJointSetup import DatasetInfo, DatasetGeneralLoadParams
import datajoint as dj


from pathlib import Path

import numpy as np
import re
import dill as pickle # because this is what people seem to do?
import hashlib
import json

defaultParams = loadDefaultParams(defParamBase = ".")
dataPath = defaultParams['dataPath']

data = []
# M1
data.append({'description': 'Earl 2019-03-18 M1 thresh - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/18/threshCrossings'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'threshold'});
data.append({'description': 'Earl 2019-03-18 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/18/nasFromLincoln'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Earl 2019-03-20 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/20'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Earl 2019-03-21 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/21'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Earl 2019-03-22 M1 thresh - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/22/threshCrossings'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'threshold'});
data.append({'description': 'Earl 2019-03-22 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/22/nasFromLincoln'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Earl 2019-03-23 M1 thresh - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/23/threshCrossings'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'threshold'});
data.append({'description': 'Earl 2019-03-23 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Earl/2019/03/23/nasFromLincoln'),
              'keyStates' : {
                  'delay': 'Delay Period',
                  'stimulus': 'Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'ReachTargetAppear',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Quincy 2020-10-20 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Quincy/2020/10/20/Array1_M1'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.2});
data.append({'description': 'Quincy 2020-10-20 M1 nas g0.5 - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Quincy/2020/10/20/Array1_g05_M1'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'Target Reach'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Quincy 2020-10-21 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Quincy/2020/10/21/Array1_M1'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'HC T1'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.2});
data.append({'description': 'Quincy 2020-10-22 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Quincy/2020/10/22/Array1_M1'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'HC T1'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.2});
data.append({'description': 'Quincy 2020-11-07 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Quincy/2020/11/07/Array1_M1'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'HC T1'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Quincy 2020-11-24 M1 nas - MGR',
              'area' : 'M1',
              'path': Path('memoryGuidedReach/Quincy/2020/11/24/Array1_M1'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'HC T1'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.2});


# PMd
#data.append({'description': 'Quincy 2020-10-20 PMd nas - MGR',
#              'area' : 'PMd',
#              'path': Path('memoryGuidedReach/Quincy/2020/10/20/Array2_PMd'),
#              'keyStates' : {
#                  'delay': 'HC Delay Period 1',
#                  'stimulus': 'HC Target Flash',
#                  'action' : 'Target Reach'
#              },
#              'stateWithAngleName' : 'HC_T1',
#              'alignmentStates': [],
#              'processor': 'Erinn',
#              'spikeIdMethod' : 'nas',
#              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
#              'nasGamma' : 0.2});
#data.append({'description': 'Quincy 2020-10-21 PMd nas - MGR',
#              'area' : 'PMd',
#              'path': Path('memoryGuidedReach/Quincy/2020/10/21/Array2_PMd'),
#              'keyStates' : {
#                  'delay': 'HC Delay Period 1',
#                  'stimulus': 'HC Target Flash',
#                  'action' : 'HC T1'
#              },
#              'stateWithAngleName' : 'HC_T1',
#              'alignmentStates': [],
#              'processor': 'Erinn',
#              'spikeIdMethod' : 'nas',
#              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
#              'nasGamma' : 0.2});
#data.append({'description': 'Quincy 2020-10-22 PMd nas - MGR',
#              'area' : 'PMd',
#              'path': Path('memoryGuidedReach/Quincy/2020/10/22/Array2_PMd'),
#              'keyStates' : {
#                  'delay': 'HC Delay Period 1',
#                  'stimulus': 'HC Target Flash',
#                  'action' : 'HC T1'
#              },
#              'stateWithAngleName' : 'HC_T1',
#              'alignmentStates': [],
#              'processor': 'Erinn',
#              'spikeIdMethod' : 'nas',
#              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
#              'nasGamma' : 0.2});
data.append({'description': 'Quincy 2020-11-07 PMd nas - MGR',
              'area' : 'PMd',
              'path': Path('memoryGuidedReach/Quincy/2020/11/07/Array2_PMd'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'HC T1'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.5});
data.append({'description': 'Quincy 2020-11-24 PMd nas - MGR',
              'area' : 'PMd',
              'path': Path('memoryGuidedReach/Quincy/2020/11/24/Array2_PMd'),
              'keyStates' : {
                  'delay': 'HC Delay Period 1',
                  'stimulus': 'HC Target Flash',
                  'action' : 'HC T1'
              },
              'stateWithAngleName' : 'HC_T1',
              'alignmentStates': [],
              'processor': 'Erinn',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'nas_PmdLincoln_NickSortsProbs',
              'nasGamma' : 0.2});


# PFC
data.append({'description': 'Pepe A1 2018-07-14 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/Array1_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe A2 2018-07-14 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe A1 2018-07-14 PFC thresh - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/ArrayNoSort1_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'threshold'});
data.append({'description': 'Pepe A2 2018-07-14 PFC thresh - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Pepe/2018/07/14/ArrayNoSort2_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'threshold'});
data.append({'description': 'Wakko A1 2018-02-11 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Wakko/2018/02/11/Array1_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Wakko A2 2018-02-11 PFC - MGS',
              'area' : 'PFC',
              'path': Path('memoryGuidedSaccade/Wakko/2018/02/11/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'SACCADE'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe 2016-02-02 PFC - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/02/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'handSort'});
data.append({'description': 'Pepe 2016-02-02 PFC NAS - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/02/ArrayNasSort2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe 2016-02-03 PFC - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/03/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'handSort'});
data.append({'description': 'Pepe 2016-02-03 PFC NAS - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/03/ArrayNasSort2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe 2016-02-04 PFC NAS - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/04/ArrayNasSort2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe 2016-02-05 PFC NAS - cuedAttn',
              'area' : 'PFC',
              'path': Path('cuedAttention/Pepe/2016/02/05/ArrayNasSort2_PFC/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Satchel A1 2020-12-10 PFC NAS - cursorMove',
              'area' : 'PFC',
              'path': Path('cursorMove/Satchel/2020/12/10/Array1_PFC/'),
              'keyStates' : {
                  'delay': 'FIX_MOVE',
                  'stimulus': 'STIM_ON',
                  'action' : 'CURSOR_POS'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'EmilioJoystick',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Satchel A2 2020-12-10 PFC NAS - cursorMove',
              'area' : 'PFC',
              'path': Path('cursorMove/Satchel/2020/12/10/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'FIX_MOVE',
                  'stimulus': 'STIM_ON',
                  'action' : 'CURSOR_POS'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'EmilioJoystick',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Satchel A1 2020-12-22 PFC NAS - cursorMove',
              'area' : 'PFC',
              'path': Path('cursorMove/Satchel/2020/12/22/Array1_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'CURSOR_ON'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'EmilioJoystick',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Satchel A2 2020-12-22 PFC NAS - cursorMove',
              'area' : 'PFC',
              'path': Path('cursorMove/Satchel/2020/12/22/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'CURSOR_ON'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'EmilioJoystick',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Satchel A1 2021-01-15 PFC NAS - cursorMove no help',
              'area' : 'PFC',
              'path': Path('cursorMove/Satchel/2021/01/15/Array1_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'CURSOR_ON'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN','CURSOR_POS'],
              'processor': 'EmilioJoystick',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Satchel A2 2021-01-15 PFC NAS - cursorMove no help',
              'area' : 'PFC',
              'path': Path('cursorMove/Satchel/2021/01/15/Array2_PFC/'),
              'keyStates' : {
                  'delay': 'TARG_OFF',
                  'stimulus': 'TARG_ON',
                  'action' : 'CURSOR_ON'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN','CURSOR_POS'],
              'processor': 'EmilioJoystick',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});






# V4
data.append({'description': 'Pepe 2016-02-02 V4 - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/02/Array1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'handSort'});
data.append({'description': 'Pepe 2016-02-02 V4 NAS - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/02/ArrayNasSort1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe 2016-02-03 V4 - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/03/Array1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'handSort'});
data.append({'description': 'Pepe 2016-02-03 V4 NAS - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/03/ArrayNasSort1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe 2016-02-04 V4 NAS - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/04/ArrayNasSort1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});
data.append({'description': 'Pepe 2016-02-05 V4 NAS - cuedAttn',
              'area' : 'V4',
              'path': Path('cuedAttention/Pepe/2016/02/05/ArrayNasSort1_V4/'),
              'keyStates' : {
                  'delay': 'Blank Before',
                  'stimulus': 'Target',
                  'action' : 'Success'
              },
              'alignmentStates': ['SOUND_CHANGE','ALIGN'],
              'processor': 'Emilio',
              'spikeIdMethod' : 'nas',
              'nasUsed' : 'UberNet_N50_L1',
              'nasGamma' : 0.2});


data = np.asarray(data) # to allow fancy indexing
#%% process data
processedDataMat = 'processedData.mat'

# dataset = [ [] for _ in range(len(data))]
dataUseLogical = np.zeros(len(data), dtype='bool')
dataIndsProcess = np.arange(len(data))#np.array([1,6])#np.array([1,4,6])
dataUseLogical[dataIndsProcess] = True

removeCoincidentChans = True
coincidenceTime = 1 #ms
coincidenceThresh = 0.2 # 20% of spikes
#checkNumTrls = 0.1 # use 10% of trials
checkNumTrls = 1 # use 100% of trials
datasetGeneralLoadParams = {
    'remove_coincident_chans' : removeCoincidentChans,
    'coincidence_time' : coincidenceTime, 
    'coincidence_thresh' : coincidenceThresh, 
    'coincidence_fraction_trial_test' : checkNumTrls
}
dsgl = DatasetGeneralLoadParams()
if len(dsgl & datasetGeneralLoadParams)>1:
    raise Exception('multiple copies of the same parameters have been saved... I thought I avoided that')
elif len(dsgl & datasetGeneralLoadParams)>0:
    genParamId = (dsgl & datasetGeneralLoadParams).fetch1('ds_gen_params_id')
else:
    # tacky, but it doesn't return the new id, syoo...
    currIds = dsgl.fetch('ds_gen_params_id')
    DatasetGeneralLoadParams().insert1(datasetGeneralLoadParams)
    newIds = dsgl.fetch('ds_gen_params_id')
    genParamId = list(set(newIds) - set(currIds))[0]

datasetDill = 'dataset.dill'
#dsiLoadParamsJson = json.dumps(datasetGeneralLoadParams, sort_keys=True) # needed for consistency as dicts aren't ordered
#dsiLoadParamsHash = hashlib.md5(dsiLoadParamsJson.encode('ascii')).hexdigest()

dsiLoadParamsHash = dsgl['ds_gen_params_id={}'.format(genParamId)].shorthash()

for dataUse in data[dataUseLogical]:
# for ind, dataUse in enumerate(data):
#    dataUse = data[1]
    # dataUse = data[ind]
    dataMatPath = dataPath / dataUse['path'] / processedDataMat

    dataDillPath = dataPath / dataUse['path'] / ('dataset_%s' % dsiLoadParamsHash) / datasetDill
    
    if dataUse['processor'] == 'Erinn':
        removeSort = np.array([31, 0])
    elif dataUse['processor']  == 'EmilioJoystick':
        removeSort = np.array([255])
    else:
        removeSort = np.array([])

    if dataDillPath.exists():
        continue
#        print('loading dataset ' + dataUse['description'])
#        with dataDillPath.open(mode='rb') as datasetDillFh:
#            datasetHere = pickle.load(datasetDillFh)
##        if 'stateWithAngleName' in dataUse:
##            setattr(datasetHere, 'stateWithAngleName', dataUse['stateWithAngleName'])
##        else:
##            setattr(datasetHere, 'stateWithAngleName', None)
#        with dataDillPath.open(mode='wb') as datasetDillFh:
#            pickle.dump(datasetHere, datasetDillFh)
#        continue 
    else:
        print('processing dataset ' + dataUse['description'])
        datasetHere = Dataset(dataMatPath, dataUse['processor'], metastates = dataUse['alignmentStates'], keyStates = dataUse['keyStates'], stateWithAngleName = dataUse['stateWithAngleName'] if 'stateWithAngleName' in dataUse else None)

        
        # first, removed explicitly ignored channels
        if removeSort is not None:
            datasetHere.removeChannelsWithSort(removeSort)

        # now, initialize a logical with all the channels/sorts (same thing when each channel is considered a neuron)
        chansKeepLog = datasetHere.maskAgainstSort([])

        # remove non-coincident spikes
        if removeCoincidentChans:
            chansKeepLog &= datasetHere.findCoincidentSpikeChannels(coincidenceTime=coincidenceTime, coincidenceThresh=coincidenceThresh, checkNumTrls=checkNumTrls)[0]

        chansKeepNums = np.where(chansKeepLog)[0]
        datasetHere.keepChannels(chansKeepNums)

        if dataUse['description'].find('Quincy') >= 0:
            trialsRemoveLog = datasetHere.findInitialHighSpikeCountTrials()
            breakpoint()
            datasetHere, trialsKeepLog = datasetHere.filterTrials(~trialsRemoveLog)
            trialsKeepNums = np.where(trialsKeepLog)[0]
        else:
            trialsKeepNums = np.arange(len(datasetHere.spikeDatTimestamps))

        dataDillPath.parent.mkdir(parents=True, exist_ok = True)
        with dataDillPath.open(mode='wb') as datasetDillFh:
            pickle.dump(datasetHere, datasetDillFh)
        breakpoint()

    datasetHash = datasetHere.hash().hexdigest()
    # do some database insertions here
    datasetHereInfo = {
        'dataset_relative_path' : str(dataDillPath.relative_to(dataPath)),
        'dataset_hash' : datasetHash,
        'dataset_name' : dataUse['description'],
        'ds_gen_params_id' : genParamId,
        'processor_name' : dataUse['processor'],
        'brain_area' : dataUse['area'],
        'task' : Path(dataUse['path']).parts[0],
        'date_acquired' : re.search('.*?(\d+-\d+-\d+).*', dataUse['description']).group(1),
        'spike_identification_method' : dataUse['spikeIdMethod'],
    }
    if 'nasUsed' in dataUse:
        datasetHereInfo.update({
            'nas_used' : dataUse['nasUsed'], 
            'nas_gamma' : dataUse['nasGamma']
        }) 

    dsi = DatasetInfo()
    if len(dsi[datasetHereInfo]) > 1:
        raise Exception('multiple copies of same dataset in the table...')
    elif len(dsi & datasetHereInfo) > 0:
        dsId = (dsi & datasetHereInfo).fetch1('dataset_id')
    else:
        dsId = len(dsi) # 0 indexed
        datasetHereInfo.update({
            'dataset_id' : dsId,
            'explicit_ignore_sorts' : removeSort,
            'channels_keep' : chansKeepNums,
            'trials_keep' : trialsKeepNums
        })
        dsi.insert1(datasetHereInfo)


    datasetHere.id = dsId
    dataUse['dataset'] = datasetHere
