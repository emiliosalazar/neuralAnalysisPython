#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:46:04 2020

@author: emilio
"""
from methods.MatFileMethods import LoadDataset
import numpy as np
from classes.BinnedSpikeSet import BinnedSpikeSet
from copy import copy
import hashlib
from matplotlib import pyplot as plt
#from mayavi import mlab

class Dataset():
    
    colorset = np.array([[127,201,127],[190,174,212],[253,192,134],[200,200,153],
                         [56,108,176],[240,2,127],[191,91,23],[102,102,102]])/255
    colorsetMayavi = [tuple(col) for col in colorset]
    
    def __init__(self, dataMatPath, preprocessor, metastates = [], keyStates = [], keyStateEnds = [], stateWithAngleName = None):
        print("loading data")
        annots = LoadDataset(dataMatPath)
        print("data loaded")
        self.cosTuningCurveParams = {'thBestPerChan': np.empty(0), 'modPerChan': np.empty(0), 'bslnPerChan': np.empty(0), 'tuningCurves': np.empty(0)}
        self.trialLabels = {}
        if preprocessor == 'Erinn':
        
            # This is a boolean indicating trials where spikes exist... we're getting rid of all other trials...
            spksExist = np.stack([(len(trlDat['spikes'][0][0]) and ('timestamps' in trlDat['spikes'][0][0].dtype.names) and len(trlDat['spikes'][0][0]['timestamps'][0]))>0 for trlDat in annots['Data']['TrialData'][0]])

            annots['Data'] = annots['Data'][:,spksExist]

            self.trialStatuses = np.stack([trlParam['trialStatus'][0][0] for trlParam in annots['Data']['Overview'][0]]).squeeze()
            
            self.spikeDatTimestamps = np.stack([trlDat['spikes'][0][0]['timestamps'][0] for trlDat in annots['Data']['TrialData'][0]])
            self.isi = np.empty(self.spikeDatTimestamps.shape, dtype=object) # initialize to right size
            with np.nditer([self.isi, self.spikeDatTimestamps], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                for valOut, valIn in iterRef:
                    if valIn[()].size != 0:
                        valOut[()] = 1/np.diff(valIn[()])
            
            self.spikeDatChannels = np.stack([trlDat['spikes'][0][0]['channel'][0] for trlDat in annots['Data']['TrialData'][0]])
            
            self.spikeDatSort = np.stack([trlDat['spikes'][0][0]['sort'][0] for trlDat in annots['Data']['TrialData'][0]])
            
            markerTargWindows = [trlParam['MarkerTargets'][0][0]['window'][0] for trlParam in annots['Data']['Parameters'][0]]
            
            markerTargWindowNames = [trlParam['MarkerTargets'][0][0]['targetName'][0] for trlParam in annots['Data']['Parameters'][0]]
            
            _, self.markerTargAngles = self.computeTargetCoordsAndAngle(markerTargWindows, markerTargWindowNames, stateWithAngleName = stateWithAngleName)
            
            statesPres = [trlDat['stateTransitions'][0][0] for trlDat in annots['Data']['TrialData'][0]]
            
            # here, if there are multiple sets of state names, we assign
            # stateNames to an object array containing one set per trial;
            # otherwise, we assign stateNames to the one unique set
            stateNamesList = [trlDat['stateNames'][0][0] for trlDat in annots['Data']['Parameters'][0]]
#            self.stateNames = annots['Data']['Parameters'][0][0]['stateNames'][0][0]
#            if not np.all([np.all(stNms == stateNamesList[0]) for stNms in stateNamesList]):
            stateNamesObjArr = np.ndarray((len(stateNamesList),), dtype='object')
            stateNamesObjArr[:] = [stNms.squeeze() for stNms in stateNamesList]
#            else:
#                self.stateNames = stateNamesList[0]

            stateHandLoc = annots['Data']['Parameters'][0][0]['StateTable'][0,0][0,0]['Hand'][0,0]['window'][0,0:3]
            
            kinematicsTemp = [trlDat['HandKinematics'][0][0]['cursorKinematics'][0][0] for trlDat in annots['Data']['TrialData'][0]]
            moveStart = [trlDat['timeMoveOnset'][0][0][0,0] if 'timeMoveOnset' in trlDat.dtype.names else np.nan for trlDat in annots['Data']['TrialData'][0]]
            for idx, (kinStruc, stNames, stPresInfo, mvStTm) in enumerate(zip(kinematicsTemp, stateNamesObjArr, statesPres, moveStart)):
                emptyObjArr = np.empty((0,0), dtype='uint8')
                # fa = np.array([[(emptyObjArr)]], dtype=[('hah', 'object')])
                #np.ndarray((0,0), dtype='int8')# np.array(([[np.ndarray((0,0), dtype='int8')]]), dtype=[('0', 'object')])
                # np.array()
                if not kinStruc.size:
                    kinematicsTemp[idx] = np.array([[(emptyObjArr,emptyObjArr,emptyObjArr,emptyObjArr)]], dtype=[('time', 'O'), ('position', 'O'), ('velocity', 'O'), ('acceleration', 'O')])
                else:
                    if not np.isnan(mvStTm): # only if there was a valid reach do we do this...
                        reachState = np.array([keyStates['action']]) # weird construction but trying to keep it in line with how it's loaded up from Matlab
                        stNames = np.append(stNames, reachState)
                        # this re-setting is to have it wrapped in an array like the rest of the states...
                        stNames[-1] = reachState
                        stateNamesObjArr[idx] = stNames

                        stateActionInd = stNames.shape[0]

                        stActionInfo = np.array([stateActionInd, mvStTm], dtype='int16')

                        stateTimes = stPresInfo[1, :]
                        indAfterInsert = (stateTimes>mvStTm).nonzero()[0][0]
                        newStPresInfo = np.insert(stPresInfo, indAfterInsert, stActionInfo, axis=1)
                        statesPres[idx] = newStPresInfo

                    
            self.statesPresented = statesPres
            self.stateNames = stateNamesObjArr 
            self.kinematics = np.stack(kinematicsTemp)
            self.kinematicCenter = stateHandLoc
        elif preprocessor == 'Yuyan':
            # if 'angle' in annots['S'].dtype.names:
            spikesArrayTrlChanTm = annots['S']['spikes'][0]
            # else:
            #     spikesArrayTrlChanTm = [spks for spks in annots['S']['spikes'][0]]
                # np.stack([])
                # np.stack([trlDat['spikes'][0][0]['sort'][0] for trlDat in annots['Data']['TrialData'][0]])
            
            # note that the [:,None].T part is to make the shape the same as Erinn's preprocessing above...
            weirdIntermediate = np.squeeze(np.stack([[np.where(chan)[0][:, None] for chan in trl] for trl in spikesArrayTrlChanTm]))
            with np.nditer(weirdIntermediate, ['refs_ok'], ['readwrite']) as iterRef:
                for valIn in iterRef:
                    valIn[()] = valIn[()].T
                    
            self.spikeDatTimestamps = weirdIntermediate
            self.isi = np.empty(self.spikeDatTimestamps.shape, dtype=object) # initialize to right size
            with np.nditer([self.isi, self.spikeDatTimestamps], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                for valOut, valIn in iterRef:
                    if valIn[()].size != 0:
                        valOut[()] = 1/np.diff(valIn[()])
            
            
            self.spikeDatSort = np.ones(self.spikeDatTimestamps.shape[0:2])
            
            # if 'angle' in annots['S'].dtype.names:
            self.markerTargAngles = np.expand_dims(np.squeeze(annots['S']['angle']),axis=1)/180*np.pi #gotta convert these angles to radians to match above...
            self.trialStatuses = np.ones((self.spikeDatTimestamps.shape[0]))
            # spikeDatChannels = np.stack(range(0, self.spikeDatTimestamps.shape[1]))
            # self.spikeDatChannels = np.repeat(spikeDatChannels[None, :], len(self.trialStatuses), axis=0)
            # elif 'cue' in annots['S'].dtype.names:
            #     self.markerTargAngles = np.expand_dims(np.squeeze(annots['S']['cue']),axis=1)
            #     self.trialStatuses = np.squeeze(annots['S']['status'])
            #     # spikeDatChannels = np.stack(range(0, spikesArrayTrlChanTm[0].shape[0]))
            #     # self.spikeDatChannels = np.repeat(spikeDatChannels[None, :], len(self.trialStatuses), axis=0)
                
            spikeDatChannels = np.stack(range(0, self.spikeDatTimestamps.shape[1]))
            self.spikeDatChannels = np.repeat(spikeDatChannels[None, :], len(self.trialStatuses), axis=0)
            
            # this is a (hopefully) temporary hard coding to reflect how Yuyan
            # preprocessed MGS data to include only the delay period 
            # Note that it's one-indexed to reflect Matlab's 1-indexing from Erinn's data...
            # Also note that we're adding a separate 'Target Flash' state, as in MGR
            # data the 'Delay Period' indicates the time when the monkey would have been
            # consciously aware of the flash, so ~100ms later. Here, we're just assuming
            # they happen at the same time
            # if 'angle' in annots['S'].dtype.names:
            self.statesPresented = [np.stack([[1, 2, -1], [0, 0, sATCT.shape[1]]]) for sATCT in spikesArrayTrlChanTm]
            # else:
            #     # note that this is based on V4 cued data having 300ms of blank present before target presentation...
            #     self.statesPresented = [np.stack([[1,2,-1], [0,300,sATCT.shape[1]]]) for sATCT in spikesArrayTrlChanTm]
            
            stateNames = np.array(['Target Flash', 'Delay Period'])
            self.stateNames = stateNames[None, :]
            
            self.kinematics = None
        elif preprocessor == 'Emilio':
            spikesArrayTrlChanTm = [spks for spks in annots['S']['spikes'][0]]
            
            # note that the [:,None].T part is to make the shape the same as Erinn's preprocessing above...
            weirdIntermediate = np.squeeze(np.stack([[np.where(chan)[0][:, None] for chan in trl] for trl in spikesArrayTrlChanTm]))
            with np.nditer(weirdIntermediate, ['refs_ok'], ['readwrite']) as iterRef:
                for valIn in iterRef:
                    valIn[()] = valIn[()].T
                    
            self.spikeDatTimestamps = weirdIntermediate
            self.isi = np.empty(self.spikeDatTimestamps.shape, dtype=object) # initialize to right size
            
            with np.nditer([self.isi, self.spikeDatTimestamps], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                for valOut, valIn in iterRef:
                    if valIn[()].size != 0:
                        valOut[()] = 1/np.diff(valIn[()])
            
            
            self.spikeDatSort = np.ones(self.spikeDatTimestamps.shape[0:2])
            
            if 'cue' in annots['S']:
                self.markerTargAngles = np.vstack([np.array([cue[0], rfOri[0]]).T for cue, rfOri in zip(annots['S']['cue'][0], annots['S']['rfOri'][0])])*np.pi # /*180 is to convert to dtype=float64 for matching to others
                self.trialLabels['cueLocation'] = np.vstack([np.array([loc[0]]).T for loc in annots['S']['cue'][0]]) # sequence length
                self.trialLabels['rfOrientation'] = np.vstack([np.array([loc[0]]).T for loc in annots['S']['rfOri'][0]]) # sequence length
                self.trialLabels['sequenceLength'] = np.vstack([np.array([loc[0]]).T for loc in annots['S']['sequenceLength'][0]]) # sequence length
                self.trialLabels['sequencePosition'] = np.vstack([np.array([pos[0]]).T for pos in annots['S']['sequencePosition'][0]]) # sequence position

            elif 'angle' in annots['S']:
                self.markerTargAngles = np.vstack([np.array([cue[0], distance[0]]).T for cue, distance in zip(annots['S']['angle'][0],annots['S']['distance'][0])])/180*np.pi 
            self.trialStatuses = np.stack([stat[0,0] for stat in annots['S']['status'][0]])
            
            spikeDatChannels = np.stack(range(0, self.spikeDatTimestamps.shape[1]))
            self.spikeDatChannels = np.repeat(spikeDatChannels[None, :], len(self.trialStatuses), axis=0)
            
            self.statesPresented = [stPres for stPres in annots['S']['statesPresented'][0]]
            
            self.stateNames = annots['S']['stateNames'][0,0]
            self.kinematics = None

        elif preprocessor == "EmilioJoystick":
            from methods.GeneralMethods import loadDefaultParams
            from methods.MatFileMethods import LoadMatFile
            params = loadDefaultParams()
            trlCodes = LoadMatFile(params['smithLabTrialCodes'])

            # here we check if I accidentally subtracted 10000 from the
            # position values instead subtracting the positions from 10000 in
            # my code... thus sending a negative value... to uint16... which
            # then is represented by adding UINT16_MAX+1 (or potentially +2?)
            if np.any([np.any(evt[evt[:,1]>50000,1:]) for evt in annots['dat']['event'][0]]):
                cursorPosXandY = [evt[evt[:,1]>50000,1:] for evt in annots['dat']['event'][0]]
            else:
                # well let's hope nothing happens within 9000-11000 besides
                # these... (also that there's never a >1000 cursor movement...
                # but... I don't think there can be...?
                cursorPosXandY = [evt[(evt[:,1]>9000) & (evt[:,1]<11000),1:] for evt in annots['dat']['event'][0]]
            # for datasets that have the CURSOR_ON code, we don't have to do
            # anything else to encode the trial has started
            sToMsConv = 1000;
            recHz = 30000 # recording frequency in Matt's rig
            cursorOnCode = trlCodes['codesStruct']['CURSOR_ON']
            if np.any([np.any(evt[:,1]==cursorOnCode) for evt in annots['dat']['event'][0]]):
                statesPresentedOrig = [cds[:,1:].T for cds in annots['dat']['trialcodes'][0]]
                statesPresentedWithMs = [cds*np.array([1,sToMsConv])[:,None] for cds in statesPresentedOrig]
                trlStMs = [cds[1,0] for cds in statesPresentedWithMs]
                statesPresentedWithMsTmInTrial = [(cds - np.array([0,cds[1,0]])[:,None]) for cds in statesPresentedWithMs]
            else:
                stOfMove = [crsPos[0,1]/recHz if crsPos.size>0 else None for crsPos in cursorPosXandY]
                statesPresentedNoMove = [cds[:,1:].T for cds in annots['dat']['trialcodes'][0]]
                statesPresentedWithMove = [np.insert(cds, (cds[1,:]>tm).nonzero()[0][0], [cursorOnCode,tm], 1) if tm is not None else cds for cds, tm in zip(statesPresentedNoMove, stOfMove)]
                statesPresentedWithMoveMs = [cds*np.array([1,sToMsConv])[:,None] for cds in statesPresentedNoMove]
                trlStMs = [cds[1,0] for cds in statesPresentedWithMoveMs]
                statesPresentedWithMsTmInTrial = [(cds - np.array([0,cds[1,0]])[:,None]) for cds in statesPresentedWithMoveMs]


            self.statesPresented = statesPresentedWithMsTmInTrial

            numChans = annots['dat']['channels'][0,0].shape[0]
            weirdIntermediate = np.stack([[ (np.hstack((trl['firstspike'][0], trl['spiketimesdiff'][:,0])).cumsum()[np.all(trl['spikeinfo'] == chan, axis=1)][:, None]/recHz*sToMsConv - trlSt).astype('uint32') for chan in trl['channels']] for trl, trlSt in zip(annots['dat'][0], trlStMs)])

            with np.nditer(weirdIntermediate, ['refs_ok'], ['readwrite']) as iterRef:
                for valIn in iterRef:
                    valIn[()] = valIn[()].T
                    
            self.spikeDatTimestamps = weirdIntermediate
            self.isi = np.empty(self.spikeDatTimestamps.shape, dtype=object) # initialize to right size
            
            with np.nditer([self.isi, self.spikeDatTimestamps], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                for valOut, valIn in iterRef:
                    if valIn[()].size != 0:
                        valOut[()] = 1/np.diff(valIn[()])
            
            
            self.spikeDatSort = np.stack([ch[:,1] for ch in annots['dat']['channels'][0]])
            
            # # NOTE the targetAgle misspelling is unfortunately necessary
            # # because the code for 'n' kept not passing through to the nev for
            # # some reason...
            # trlDat = [trl['trial'][0,0] for trl in annots['dat']['params'][0]]
            # mrkAngle = np.vstack([trl['targetAngle'][0,0] if 'targetAngle' in trl.dtype.names else trl['targetAgle'][0,0] for trl in trlDat])
            # for ind, ang in enumerate(mrkAngle):
            #     try:
            #         newAngInt = int(ang)
            #         mrkAngle[ind] = newAngInt
            #     except ValueError:
            #         for indAngStr in range(1,len(ang[0])):
            #             try:
            #                 int(ang[0][:indAngStr])
            #                 newAngInt = int(ang[0][:indAngStr])
            #             except ValueError:
            #                 break
            #         mrkAngle[ind] = newAngInt

            # self.markerTargAngles = mrkAngle

            self.trialStatuses = np.stack([np.any(res==150) for res in annots['dat']['result'][0]]).astype('int8')
            
            self.spikeDatChannels = np.stack(annots['dat']['channels'].squeeze())[:,:,0]
            
            
            self.stateNames = trlCodes['codesArray']

            kinematicsTemp = [np.vstack((cPXY[0:-1:2, 0], cPXY[1::2, 0], cPXY[0:-1:2, 1]/recHz*sToMsConv-trlSt)).T for cPXY, trlSt in zip(cursorPosXandY, trlStMs)]
            kinematicsCont = np.empty((len(kinematicsTemp),), dtype='object')
            for ind,kin in enumerate(kinematicsTemp):
                kinematicsCont[ind] = kin

            # NOTE the following is meant to correct codes or add missing ones
            behMatPath = dataMatPath.parent.parent / 'allCodes.mat'
            try:
                breakpoint()
                behDatForKin = LoadDataset(behMatPath)['allCodes'].squeeze()

                mrkAng = []
                for cds in behDatForKin:
                    cdsTrl = cds['codes'][0,0][:, 0]
                    asciiCodeTrlDat = (cdsTrl[(cdsTrl>=256) & (cdsTrl < 512)] - 256).astype('int')
                    strCharArray = np.array([chr(x) for x in range(256)])[asciiCodeTrlDat]
                    strChar = "".join(strCharArray)
                    targAngSt = strChar.find('targetAngle=') + len('targetAngle=')
                    lenOfAng = strChar[targAngSt:].find(';')
                    targAng = int(strChar[targAngSt:targAngSt+lenOfAng])
                    mrkAng.append(targAng)

                self.markerTargAngles = np.array(mrkAng)[:,None]
                # since these aren't uint32, the equivalent of checking for >50000
                # is actually checking for the negative value...
                self.kinematicsCenter = [0, 0]
                if np.any([np.any(cds['codes'][0,0][:,0]<-9000) for cds in behDatForKin]):
                    cursorPosXandYFromBeh = [cds['codes'][0,0][cds['codes'][0,0][:,0]<-9000,:] for cds in behDatForKin]
                else:
                    # well let's hope nothing happens within 9000-11000 besides
                    # these... (also that there's never a >1000 cursor movement...
                    # but... I don't think there can be...?
                    #
                    # Also note the inversion of the coordinates, which will
                    # allow us to just subtract the offshift later to get the
                    # right coordinates
                    cursorPosXandYFromBeh = [(cds['codes'][0,0][(cds['codes'][0,0][:,0]>9000) & (cds['codes'][0,0][:,0]<11000),:])*np.array([-1, 1]) for cds in behDatForKin]

                cursPosXandYNevTimesInMs = [cPXY[:, 1]/recHz*sToMsConv - trlSt for cPXY, trlSt in zip(cursorPosXandY, trlStMs)]
                cursPosXandYBehTimesInMs = [cPXY[:, 1]*sToMsConv for cPXY in cursorPosXandYFromBeh]

                # NOTE aaaactually... I think I determined that the timings here are
                # actually pretty nicely aligned... syooo... I'm just going to
                # replace the NEV cursor movements with the ones in the
                # behavioral file. Kthx hopefully nothing breaks.
                #
                # BUT! Sometimes I'll record some trials before/after I was
                # neural recording I guess, so I need to use the timings to
                # check which ones to keep...
                cursPosXandYUse = []
                indStBeh = 0
                for indNev, nevTrl in enumerate(cursPosXandYNevTimesInMs):
                    # if indNev==1046:
                    #     breakpoint()
                    if nevTrl.size == 0:
                        # I can see a corner case bug here where there was only
                        # a brief cursor motion, which the NEV missed, but the
                        # behavior didn't--here I'm just gonna throw that away.
                        # But with a case like that I think I wouldn't care
                        # much for the trial anyway, syo...
                        cursPosXandYUse.append(nevTrl)
                    else:
                        trialFound = False
                        for indBeh in list(range(indStBeh, len(cursPosXandYBehTimesInMs))) + list(range(indStBeh)):
                            behTrl = cursPosXandYBehTimesInMs[indBeh]
                            if behTrl.size == 0:
                                continue
                            # all times must be within 3ms... which given the 100Hz frame rate
                            # hopefully means we won't accidentally pick some overlapping
                            # frames or stuff...
                            if np.all([np.abs(nvTm-behTrl).min() < 3 for nvTm in nevTrl]):
                                cursPosXandYUse.append(cursorPosXandYFromBeh[indBeh])
                                trialFound = True
                                break
                                
                        if not trialFound:
                            # probably an indication that the 3ms similarity
                            # is too conservative...
                            breakpoint()
                        else:
                            # when behavior and NEV are in order, this speeds
                            # up finding the correct trial by assuming that a
                            # correct trial in the behavior data for an NEV
                            # data will not occur before a previous correct
                            # trial. But at the same time if something weird
                            # happens it also loops back around from the start
                            # as well... but since that's weird I'm breakpointing
                            if indBeh < indStBeh:
                                breakpoint()
                            indStBeh = indBeh+1

                reqShift = np.array([-10000, -10000])
                kinematicsTemp = [np.vstack((cPXY[0:-1:2, 0]-reqShift[0], cPXY[1::2, 0]-reqShift[1], cPXY[0:-1:2, 1]*sToMsConv)).T if cPXY.size>0 else np.array([]) for cPXY in cursPosXandYUse]
                kinematicsCont = np.empty((len(kinematicsTemp),), dtype='object')
                for ind,kin in enumerate(kinematicsTemp):
                    kinematicsCont[ind] = kin
            except OSError:
                raise Exception('Remember to upload the relevate allCodes behavior file!')

            breakpoint()
            self.kinematics = kinematicsCont
        # NOTE HERE WE ARE
        elif preprocessor == "EmilioJoystickHE" or preprocessor == "EmilioKalmanBci":
            from methods.GeneralMethods import loadDefaultParams
            from methods.MatFileMethods import LoadMatFile
            params = loadDefaultParams()
            trlCodes = LoadMatFile(params['smithLabTrialCodes'])
            bciCursCode = trlCodes['codesStruct']['BCI_CURSOR_POS'][0,0].squeeze()
            cursCode = trlCodes['codesStruct']['CURSOR_POS'][0,0].squeeze()

            # cursor movements will be between 9000 and 11000
            datName = list(annots.keys())[0]
            # cursorPosXandY = [evt[(evt[:,1]>9000) & (evt[:,1]<11000),1:].astype(float)-[10000, 0] for evt in annots[datName]['event'][0]]
            # cursorPosXandY = [evt[(evt[:,1]>9000) & (evt[:,1]<11000),1:].astype(float)-[10000, 0] for evt in annots[datName]['event'][0]]

            # set the states presented
            sToMsConv = 1000;
            recHz = 30000 # recording frequency in Matt's rig

            # here we compute the time when the reaction happens and then mark it as a state that can be processed
            if preprocessor == "EmilioJoystickHE":
                cursorPosXandY = np.array([cds[np.where(cds[:,1]==cursCode)[0][:,None]+np.array([1,2]),1:].astype(int)-np.vstack([[10000,0],[10000,0]]) for cds in annots[datName]['event'][0]])
                cursorRtCode = trlCodes['codesArray'].squeeze().shape[0]+1
                cursorRtState = np.array(['CURSOR_REACT'])
                trlCodes['codesArray'] = np.append(trlCodes['codesArray'], cursorRtState[:,None], axis=1)
                trlCodes['codesArray'][0,-1] = cursorRtState
                posAndTime = [np.vstack((cPXY[:, 0, 0], cPXY[:, 1, 0], cPXY[:, 0, 1]/recHz)).T for cPXY in cursorPosXandY]
                distFromCentAndTotMoveAndTime = [np.vstack([np.sqrt(np.sum(pt[:, :2]**2, axis=1)),np.sqrt(np.sum((pt[:, :2] - pt[0,:2])**2, axis=1)), pt[:, -1]] ).T if pt.size else np.array([]) for pt in posAndTime]
                timeAndPosDiff = [np.hstack([np.diff(pt,axis=0), pt[:-1, [-1]]] ) for pt in posAndTime]
                # speed below is in pixels/ms
                speedAndTimeInMv = [np.vstack([np.sqrt(np.sum(pD[:,:2]**2, axis=1))/pD[:, -2], pD[:,-1]-pD[0,-1] if pD.size else np.array([])]).T for pD in timeAndPosDiff]
                joystickHoldRtWindow = [cds['block']['joystickHoldTolerance'][0,0] for cds in  annots[datName]['params'].squeeze()]
                speedThresh = .5 # pixels/ms--process of staring at data
                binsWithMovement = [((sAT[:,0]>speedThresh) | (np.sqrt(np.sum(crsPosAndT[:-1, :2]**2, axis=1)) > jHT)) if sAT.size>0 else None for crsPosAndT, sAT, jHT in zip(posAndTime, speedAndTimeInMv, joystickHoldRtWindow)]
                stOfMove = [crsPosAndT[(binsWithMvmt).nonzero()[0][0], 2] if np.any(binsWithMvmt) else None for crsPosAndT, binsWithMvmt in zip(posAndTime, binsWithMovement)]
                statesPresentedNoMoveStart = [cds[:,1:].T for cds in annots[datName]['trialcodes'][0]]
                statesPresentedWithMove = [np.insert(cds, (cds[1,:]>tm).nonzero()[0][0], [cursorRtCode,tm], 1) if tm is not None else cds for cds, tm in zip(statesPresentedNoMoveStart, stOfMove)]
                statesPresentedWithMoveMs = [cds*np.array([1,sToMsConv])[:,None] for cds in statesPresentedWithMove]
                trlStMs = [cds[1,0] for cds in statesPresentedWithMoveMs]
                statesPresentedWithMsTmInTrial = [(cds - np.array([0,cds[1,0]])[:,None]) for cds in statesPresentedWithMoveMs]

            else:
                cursorPosXandY = np.array([cds[np.where(cds[:,1]==bciCursCode)[0][:,None]+np.array([1,2]),1:].astype(int)-np.vstack([[10000,0],[10000,0]]) for cds in annots[datName]['event'][0]])
                statesPresentedNoMoveStart = [cds[:,1:].T for cds in annots[datName]['trialcodes'][0]]
                statesPresentedNoMoveStartMs = [cds*np.array([1,sToMsConv])[:,None] for cds in statesPresentedNoMoveStart]
                trlStMs = [cds[1,0] for cds in statesPresentedNoMoveStartMs]
                statesPresentedWithMsTmInTrial = [(cds - np.array([0,cds[1,0]])[:,None]) for cds in statesPresentedNoMoveStartMs]

            self.statesPresented = statesPresentedWithMsTmInTrial

            # set spike timestamps
            numChans = annots[datName]['channels'][0,0].shape[0]
            weirdIntermediate = np.stack([[ (np.hstack((firstSpk.squeeze(), spkTimeDiffs[:,0])).cumsum()[np.all(spikeInfos == chan, axis=1)][:, None]/recHz*sToMsConv - trlSt).astype('uint32') for chan in allChans] for allChans, firstSpk, spkTimeDiffs, spikeInfos, trlSt in zip(annots[datName]['channels'][0], annots[datName]['firstspike'][0], annots[datName]['spiketimesdiff'][0], annots[datName]['spikeinfo'][0], trlStMs)])

            with np.nditer(weirdIntermediate, ['refs_ok'], ['readwrite']) as iterRef:
                for valIn in iterRef:
                    valIn[()] = valIn[()].T
                    
            self.spikeDatTimestamps = weirdIntermediate
            self.isi = np.empty(self.spikeDatTimestamps.shape, dtype=object) # initialize to right size
            
            with np.nditer([self.isi, self.spikeDatTimestamps], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                for valOut, valIn in iterRef:
                    if valIn[()].size != 0:
                        valOut[()] = 1/np.diff(valIn[()])
            
            
            self.spikeDatSort = np.stack([ch[:,1] for ch in annots[datName]['channels'][0]])
            
            self.trialStatuses = np.stack([np.any(res==150) for res in annots[datName]['result'][0]]).astype('int8')
            
            self.spikeDatChannels = np.stack(annots[datName]['channels'].squeeze())[:,:,0]
            
            self.stateNames = trlCodes['codesArray']

            kinematicsTemp = [np.vstack((cPXY[:, 0, 0], cPXY[:, 1, 0], cPXY[:, 0, 1]/recHz*sToMsConv-trlSt)).T for cPXY, trlSt in zip(cursorPosXandY, trlStMs)]
            kinematicsCont = np.empty((len(kinematicsTemp),), dtype='object')
            for ind,kin in enumerate(kinematicsTemp):
                kinematicsCont[ind] = kin

            mrkAng = []
            for cds in annots[datName]['event'].squeeze():
                cdsTrl = cds[:, 1]
                asciiCodeTrlDat = (cdsTrl[(cdsTrl>=256) & (cdsTrl < 512)] - 256).astype('int')
                strCharArray = np.array([chr(x) for x in range(256)])[asciiCodeTrlDat]
                strChar = "".join(strCharArray)
                targAngSt = strChar.find('targetAngle=') + len('targetAngle=')
                lenOfAng = strChar[targAngSt:].find(';')
                targAng = int(strChar[targAngSt:targAngSt+lenOfAng])
                mrkAng.append(targAng)

            self.markerTargAngles = np.array(mrkAng)[:,None]
            self.kinematics = kinematicsCont

            
        allComboTimestamps = np.concatenate(self.spikeDatTimestamps.flatten(), axis=1)
        if np.any(allComboTimestamps < 0):
            print('things might break if there are negative timestamps right from the offset...')
            print('or they might not... for now, just know that the minimum timestamp is {}'.format(allComboTimestamps.min()))
        
        self.minTimestamp = np.min(allComboTimestamps)
        self.maxTimestamp = np.stack([np.max(np.concatenate(spDT.flatten(), axis=1)) for spDT in self.spikeDatTimestamps])
        

        self.id = None
        self.stateWithAngleName = stateWithAngleName # this is the state for some preprocessing that will let us extract the angle name
        self.metastates = metastates # these are states that don't refer to what the monkey is seeing, but rather some state of the session itself (for alignment purposes, say)
        self.keyStates = keyStates # I'm trying to allow some semblance of consistency for states of 'interest' among different stimuli -- i.e. 'delay period', 'stim period', etc.
        self.keyStateEnds = keyStateEnds # I'm trying to allow some semblance of consistency for states of 'interest' among different stimuli -- i.e. 'delay period', 'stim period', etc.
        

    def hash(self):
        return hashlib.md5(str(self.spikeDatTimestamps).encode('ascii')) # this is both unique, fast, and good info
            
    def filterTrials(self, mask):
        filteredTrials = copy(self)
        filteredTrials.trialStatuses = filteredTrials.trialStatuses[mask]
        filteredTrials.spikeDatTimestamps = filteredTrials.spikeDatTimestamps[mask, :]
        filteredTrials.isi = filteredTrials.isi[mask, :]
        filteredTrials.spikeDatChannels = filteredTrials.spikeDatChannels[mask, :]
        filteredTrials.spikeDatSort = filteredTrials.spikeDatSort[mask, :]
        filteredTrials.markerTargAngles = filteredTrials.markerTargAngles[mask, :]
        if not hasattr(filteredTrials, 'trialLabels'):
            filteredTrials.trialLabels = {}
        if len(filteredTrials.trialLabels) > 0:
            filtTrialLabels = copy(filteredTrials.trialLabels)
            for label, values in filtTrialLabels.items():
                filtTrialLabels[label] = values[mask,:]

            filteredTrials.trialLabels = filtTrialLabels
        if len(filteredTrials.stateNames)>1:
            filteredTrials.stateNames = filteredTrials.stateNames[mask]
        else:
            filteredTrials.stateNames = filteredTrials.stateNames

        filteredTrials.statesPresented = [filteredTrials.statesPresented[hr] for hr in range(0, len(filteredTrials.statesPresented)) if mask[hr]]
        filteredTrials.cosTuningCurveParams = {'thBestPerChan': np.empty(0), 'modPerChan': np.empty(0), 'bslnPerChan': np.empty(0), 'tuningCurves': np.empty(0)}
        filteredTrials.minTimestamp = filteredTrials.minTimestamp
        filteredTrials.maxTimestamp = filteredTrials.maxTimestamp[mask]
        if filteredTrials.kinematics is not None:
            filteredTrials.kinematics = filteredTrials.kinematics[mask]
        return filteredTrials, mask
        
    def successfulTrials(self):
        successfulTrialsLog = self.trialStatuses==1
        successfulTrials, maskFilt = self.filterTrials(successfulTrialsLog)
        return successfulTrials, maskFilt
        
    def failTrials(self):
        failTrialsLog = self.trialStatuses == 0
        failTrials, maskFilt = self.filterTrials(failTrialsLog)
        return failTrials, maskFilt
        
    def filterOutState(self, stateName):
        targetWithState = self.trialsWithState(stateName)
        trialsWithoutStateLog = np.logical_not(targetWithState)
        datasetWithoutState, maskFilt = self.filterTrials(trialsWithoutStateLog)
        return datasetWithoutState, maskFilt
    
#    def filterOutCatch(self):
#        return self.filterOutState('Catch')
#        stNm = self.stateNames
#        #stNm = stNm[None, :]
#        stNm = np.repeat(stNm, len(self.statesPresented), axis=0)
#        stPres = self.statesPresented
#        stPres = [stPr[0, stPr[0]!=-1] for stPr in stPres]
#        stPresTrl = zip(stPres, stNm, range(0, len(self.statesPresented)))
#        targetWithCatch = np.stack([np.any(np.core.defchararray.find(np.stack(stNm[np.int32(stPres-1)]),'Catch')!=-1) for stPres, stNm, trialNum in stPresTrl])
#        trialsWithoutCatchLog = np.logical_not(targetWithCatch)
#        datasetWithoutCatch = self.filterTrials(trialsWithoutCatchLog)
#        return datasetWithoutCatch

    def trialsWithState(self, stateName, asLogical=True):
        stNm = self.stateNames
        if len(stNm)==1: # otherwise there's already a unique set of state names per trial
            stNm = np.repeat(stNm, len(self.statesPresented), axis=0)
        stPres = self.statesPresented
        stPres = [stPr[0, stPr[0]!=-1] for stPr in stPres]
        stPresTrl = zip(stPres, stNm, range(0, len(self.statesPresented)))
        # NOTE the -1 in stPres-1 serves to conver *MATLAB* 1-indexing to *PYTHON* 0-indexing
        trialsWithState = np.stack([np.any(np.core.defchararray.find(np.stack(stNames[np.int32(stPres-1)]),stateName)!=-1) for stPres, stNames, trialNum in stPresTrl])

        if not asLogical:
            trialsWithState = np.where(targetWithState)[0]
        
        return trialsWithState
    
    def binSpikeData(self, startMs=0, endMs=3600, binSizeMs=50, alignmentPoints=None):
        
        smallestSpiketime = 0 # this should always be the case...
        if type(endMs) is list:
            assert len(endMs) == self.trialStatuses.shape[0], "different number of dataset trials and ms locations"
            # Adding binSizeMs to the end ensures that that last spike gets counted...
            binsUse = [np.arange(sMs, eMs, binSizeMs)[np.logical_and(smallestSpiketime<=np.arange(sMs, eMs, binSizeMs), np.arange(sMs, eMs, binSizeMs)<=lrgSpTm+binSizeMs)] for sMs, eMs, lrgSpTm in zip(startMs, endMs, self.maxTimestamp)]
            startMs = np.stack([bU[0] for bU in binsUse])
            endMs = np.stack([bU[-1] for bU in binsUse])
            alignmentBins = np.stack([tuple((aP-bnsUse[0])/binSizeMs for aP in alPo) for alPo, bnsUse in zip(alignmentPoints, binsUse)]) if alignmentPoints is not None else None
            spikeDatBinnedList = [[[[counts for counts in np.histogram(chanSpks, bins=bnUse)][0] for chanSpks in trlChans][0] for trlChans in trl] for trl, bnUse in zip(self.spikeDatTimestamps, binsUse)]
        else:
            binsUse = np.arange(startMs, endMs, binSizeMs)[np.logical_and(smallestSpiketime<=np.arange(startMs, endMs, binSizeMs), np.arange(startMs, endMs, binSizeMs)<=np.max(self.maxTimestamp))]
            startMs = np.stack([binsUse[0]])
            endMs = np.stack([binsUse[-1]])
            alignmentBins = np.stack(tuple((alignmentPoints-binsUse[0])/binSizeMs for aP in alignmentPoints)) if alignmentPoints is not None else None
            spikeDatBinnedList = [[[[counts for counts in np.histogram(chanSpks, bins=binsUse)][0] for chanSpks in trlChans][0] for trlChans in trl] for trl in self.spikeDatTimestamps]
            

        spikeDatBinnedArr = [np.stack(trl) for trl in spikeDatBinnedList]
        
            
        try:
            spikeDatBinned = np.stack(spikeDatBinnedArr)
        except ValueError:
            # spikeDatBinnedByChans = np.empty(spikeDatBinnedArr[0].shape[0], dtype='object')
            # spikeDatBinned = [sp.view(BinnedSpikeSet) for sp in spikeDatBinnedArr]
            # the None here indicates that there is only one trial for each bin... ensures that the
            # dimensions appropriately reflect what they should
            # spikeDatBinned = [sp[None,chansOfInt, :] for sp in spikeDatBinned]
            
            
            # for idx, sp in enumerate(spikeDatBinned):
            #     sp = sp/(binSizeMs/1000)
            #     sp.binSize = binSizeMs
            #     sp.start = np.stack([startMs[idx]])
            #     sp.end = np.stack([endMs[idx]])
            #     sp.alignmentBins = np.stack([alignmentBins[idx]]) if alignmentBins is not None else None
            #     spikeDatBinned[idx] = sp
            
            spikeDatBinnedBreakout = [[sp for sp in spBArr] for spBArr in spikeDatBinnedArr]
            spikeDatBinned = BinnedSpikeSet(np.asarray(spikeDatBinnedBreakout)/(binSizeMs/1000),
                                            binSize = binSizeMs,
                                            start = startMs,
                                            end = endMs,
                                            alignmentBins = alignmentBins,
                                            units = 'Hz',
                                            labels = self.trialLabels)
        else:
            spikeDatBinned = spikeDatBinned/(binSizeMs/1000)
            spikeDatBinned = spikeDatBinned.view(BinnedSpikeSet)
            spikeDatBinned.binSize = binSizeMs
            spikeDatBinned.start = startMs
            spikeDatBinned.end = endMs
            spikeDatBinned.alignmentBins = alignmentBins
            spikeDatBinned.units = 'Hz'
            spikeDatBinned.labels = self.trialLabels
        
        
            
        return spikeDatBinned
    
    # coincidenceTime is how close spikes need to be, in ms, to be considered coincident, 
    # coincidenceThresh in percent/100 (i.e. 20% = 0.2)
    # checkNumTrls is the percentage of trials to go through
    def findCoincidentSpikeChannels(self, coincidenceTime=1, coincidenceThresh=0.2, checkNumTrls=0.1, plotResults = False):
        
        
        # spikeCountOverallPerChan = [len(trlStmps)]
        # wanna keep random state the same every time...
        initSt = np.random.get_state()
        np.random.seed(seed=0)
        
        numTrls = len(self.spikeDatTimestamps)
        trlIdxes = np.arange(0, numTrls)
        randIdxes = np.random.permutation(trlIdxes)
        idxesUse = randIdxes[:round(numTrls*checkNumTrls)]

        # get the overall spike count
        spikeCountOverallPerChan = np.stack(np.stack([np.concatenate(trlSpks[:], axis=1).shape[1] for trlSpks in self.spikeDatTimestamps[idxesUse,:].T]))
        coincCnt = np.zeros((spikeCountOverallPerChan.shape[0],  spikeCountOverallPerChan.shape[0]))
        
        for numRound, trlIdx in enumerate(idxesUse):
            if not numRound % 10 and numRound != len(idxesUse)-1:
                print(str(100*numRound/(len(idxesUse)-1)) + "% done")
            if numRound == len(idxesUse)-1:
                print("100% done")

            # this gets you the spike times for each channel in a trial
            trlChTmstmp = self.spikeDatTimestamps[trlIdx]

            # then we go through each channel ch1, and look at its coincidence
            # with another channel, ch2
            for idx1, ch1 in enumerate(trlChTmstmp):
                for idx2, ch2 in enumerate(trlChTmstmp):
                    # we're only computing one half of the matrix since its
                    # symmetric
                    if idx2>idx1:
                        # find the spike time difference and only add the spikes
                        # that are closer together than the coincidenceTime
                        ch1ch2TDiff = ch1 - ch2.T
                        coincCnt[idx1, idx2] = coincCnt[idx1, idx2]+np.where(abs(ch1ch2TDiff)<coincidenceTime)[0].shape[0]
                        coincCnt[idx2, idx1] = coincCnt[idx1, idx2]
                    
        # each column of coincProp is the proportion of *that column* channel's
        # spikes that are coincident with a given *row* channel's spikes (this
        # is because broadcasting works on trailing dimensions *first*--so the
        # columns here)
        coincProp = coincCnt/spikeCountOverallPerChan
        
        # here we're obviously checking that the proportion of coincident
        # spikes is below the threshold (I use 0.2, or 20% coincidence), but
        # we're also checking that both channels actually *have* spikes.
        # (Algorithmically, that gets rid of NaNs caused by a divide by zero
        # above, which gives you an NaN. But also it makes sense to remove
        # channels that don't fire).
        chanPairsGoodLogical = (coincProp<coincidenceThresh) & (spikeCountOverallPerChan != 0)[:,None] & (spikeCountOverallPerChan != 0)[:,None].T # every column is the division with that channel's spike count
        chanPairsBadLogicalOrig = (coincProp>=coincidenceThresh) | (spikeCountOverallPerChan == 0)[:,None] | (spikeCountOverallPerChan == 0)[:,None].T# every column is the division with channel's spike count

        # NOTE: current heuristic for bad channels is as follows: we want no
        # channels that are coincident with other channels; however, if we
        # remove one channel, we're by definition removing another channel's
        # coincidence as well... so we're going to remove most
        # coincidences--either too many channels coincident with the current
        # channel, or the current channel coincident with too many channels--to
        # least coincidences until there are no coincidences left above the
        # threshold
        chanPairsBadLogical = chanPairsBadLogicalOrig.copy()
        badChans = []
        timesOtherChannelsCoincidentToThisChannelOrig = chanPairsBadLogical.sum(axis=1)
        timesThisChannelCoincidentToOtherChannelsOrig = chanPairsBadLogical.sum(axis=0)

        # doing it this way so that at the breakpoint below you can get an easy
        # overview of what things were like to start with
        timesOtherChannelsCoincidentToThisChannel = timesOtherChannelsCoincidentToThisChannelOrig
        timesThisChannelCoincidentToOtherChannels = timesThisChannelCoincidentToOtherChannelsOrig 

        # the while loop just keeps going until there are no coincident channels
        while timesOtherChannelsCoincidentToThisChannel.sum() + timesThisChannelCoincidentToOtherChannels.sum() > 0:
            mxOthCh = timesOtherChannelsCoincidentToThisChannel.max()
            mxThsCh = timesThisChannelCoincidentToOtherChannels.max()
            
            # as stated above, we remove channels one by one, and always going
            # for the channel that has the highest coincidence with other
            # channels
            if mxThsCh >= mxOthCh:
                chRem = timesThisChannelCoincidentToOtherChannels.argmax()
            else:
                chRem = timesOtherChannelsCoincidentToThisChannel.argmax()

            # keep track of channels to remove
            badChans.append(chRem)
            # set them as not coincident to any channel, and no channel
            # coincident to them, because we know we'll remove them
            chanPairsBadLogical[:,chRem] = False
            chanPairsBadLogical[chRem,:] = False
            
    
            # recompute the coincidence numbers so we can find the new channel
            # that has most coincidence
            timesOtherChannelsCoincidentToThisChannel = chanPairsBadLogical.sum(axis=1)
            timesThisChannelCoincidentToOtherChannels = chanPairsBadLogical.sum(axis=0)

        chansKeepLog = np.full(trlChTmstmp.shape, True)
        chansKeepLog[badChans] = False

        badChansLog = ~chansKeepLog

        if not plotResults:
            # normal case; plotting for now is only being done once we've done this round
            if badChansLog.nonzero()[0].size > 0:
                print("Removed channels " + str(list(badChansLog.nonzero()[0])))
            else:
                print("No coincident channels to remove")
            print('Worth a manual review here, methinks')
            print('Perhaps look at variables:')
            print('    timesOtherChannelsCoincidentToThisChannelOrig')
            print('    timesThisChannelCoincidentToOtherChannelsOrig')
            print('    coincProp')

            import plotly as ply
            import plotly.express as px 
            px.imshow(coincProp).show()

            breakpoint()

        
        # reset random state
        np.random.set_state(initSt)

        if plotResults:
            import matplotlib.gridspec as gridspec

            fig = plt.figure(constrained_layout=True)

            axSpec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

            ax = []
            ax.append(fig.add_subplot(axSpec[0,0]))
            ax.append(fig.add_subplot(axSpec[0,1]))
            ax.append(fig.add_subplot(axSpec[1,:]))

            im = ax[0].imshow(coincProp)
            im.set_clim(0, 1)
            ax[0].set_title('coinc between channels')
            ax[0].set_xlabel('channel coincident to')
            ax[0].set_ylabel('channel checked')

            im = ax[1].imshow(coincProp[chansKeepLog][:, chansKeepLog])
            im.set_clim(0, 1)
            fig.colorbar(im, ax=ax[1])

            ax[1].set_title('remove high coincidence channels')
            ax[1].set_xlabel('channel coincident to')

            ax[2].hist(coincProp.flatten(), bins=np.arange(0, 1, 0.01))
            ax[2].set_title('coinc. % across channels')
            ax[2].set_xlabel('coincidence %')
            ax[2].set_ylabel('count')
            
        
        return chansKeepLog, badChansLog

    def findInitialHighSpikeCountTrials(self):
        # NOTE this is a heuristic function, meant for certain datasets where
        # high firing indicates some sort of artifact (typically, movement)
        # that occurs at the beginning of a session
        breakpoint()

        # first find the number of spikes per channel in each trial
        totalTrialChanSpikeCounts = np.stack([np.array([trlChan.shape[1] for trlChan in trlChanSpks]) for trlChanSpks in self.spikeDatTimestamps])
        # for each channel, find which trials were at least one standard deviation above average spike count
        highSpkCountTrialPerChan = [fr > fr.mean() + fr.std() for fr in totalTrialChanSpikeCounts.T]
        # for each trial, find how many channels had a high spike count rate
        numHighSpkCountChansInTrial = np.sum(np.stack(highSpkCountTrialPerChan), axis=0)
        # make a list of '1' and '0' for trials that had (or didn't) high spike counts on at least one channel
        listOneZeroStrTrialHasHighSpikeCountChans = (numHighSpkCountChansInTrial>0).astype(int).astype('str')
        # convert the list to a full-fledge string
        strHighSpikeCountChan = ''.join(listOneZeroStrTrialHasHighSpikeCountChans)
        # use the .find() method of strings to basically convolve a '00' and find the first time a pair of trials has no high spike count... we'll take that to be the end of the movement artifact
        firstPairCalmTrials = strHighSpikeCountChan.find('00')

        trialMaskHighSpikeCount = np.full(numHighSpkCountChansInTrial.shape[0], False)

        trialMaskHighSpikeCount[:firstPairCalmTrials] = True

        return trialMaskHighSpikeCount

        
    def filterChannels(self, chanMask):
        self.spikeDatTimestamps = np.stack([trlTmstmp[chanMask] for trlTmstmp in self.spikeDatTimestamps])
        self.spikeDatChannels = self.spikeDatChannels[:, chanMask]
        self.spikeDatSort = self.spikeDatSort[:, chanMask]

    def removeChannels(self, chan):
        chanMask = np.full(self.spikeDatTimestamps[0].shape, True)
        chanMask[chan] = False
        self.filterChannels(chanMask)

    def keepChannels(self, chan):
        chanMask = np.full(self.spikeDatTimestamps[0].shape, False)
        chanMask[chan] = True
        self.filterChannels(chanMask)
    
    def removeChannelsWithSort(self, sort):
        keepChanMask = self.maskAgainstSort(sort)
        self.spikeDatTimestamps = np.stack([trlTmstmp[keepChanMask] for trlTmstmp in self.spikeDatTimestamps])
        self.spikeDatChannels = self.spikeDatChannels[:, keepChanMask]
        self.spikeDatSort = self.spikeDatSort[:, keepChanMask]
        
    def keepChannelsWithSort(self, sort):
        keepChanMask = self.maskForSort(sort)
        self.spikeDatTimestamps = np.stack([trlTmstmp[keepChanMask] for trlTmstmp in self.spikeDatTimestamps])
        self.spikeDatChannels = self.spikeDatChannels[:, keepChanMask]
        self.spikeDatSort = self.spikeDatSort[:, keepChanMask]
    
    def maskForSort(self, sortList):
        breakpoint()
        sortMask = np.full(self.spikeDatSort[0].shape, False)
        for sort in sortList:
            sortMask = np.logical_or(sortMask, self.spikeDatSort[0] == sort)
            
        return sortMask
            
    def maskAgainstSort(self, sortList):
        sortMask = np.full(self.spikeDatSort[0].shape, True)
        for sort in sortList:
            sortMask = np.logical_and(sortMask, self.spikeDatSort[0] != sort)
            
        return sortMask
    
    def computeTargetCoordsAndAngle(self, markTargWin, markTargWinNm, stateWithAngleName = 'ReachTargetAppear'):
        targetInTrialsWithTrial = np.stack([np.append(target[0][0:3], trialNum) for targetsTrial, targetNamesTrial, trialNum in zip(markTargWin, markTargWinNm, range(0, len(markTargWin))) for target, name in zip(targetsTrial, targetNamesTrial) if name[0][0][0].find(stateWithAngleName)!=-1])
        _, idx = np.unique(targetInTrialsWithTrial, axis=0, return_index=True)
        #idx = list(np.sort(idx)) # because targetInTrialsWithTrial is still a list, not an nparray
        targetsCoords = targetInTrialsWithTrial[np.sort(idx), 0:3]
        
        # this is making the assumption that the target coordinates are
        # balanced... should probably update this to take into account some
        # way of actually loading the known center...
        targCenter = np.median(np.unique(targetsCoords, axis=0), axis=0)
        targetsCoords = targetsCoords - targCenter
        targAngle = np.arctan2(targetsCoords[:, 1], targetsCoords[:, 0])
        targAngle = np.expand_dims(targAngle, axis=1)
        
        return targetsCoords, targAngle
    
    
    #%% methods to do stuff on data
    def computeCosTuningCurves(self, spikeDatBinnedArr = None, notChan=[31, 0]):
        uniqueTargAngle, trialsPresented = np.unique(self.markerTargAngles, axis=0, return_inverse=True)
        
        if spikeDatBinnedArr is None:
            spikeDatBinnedArr = self.binSpikeData()

#        spikeDatBinnedArrImpTmAvg = spikeDatBinnedArr.timeAverage()

        groupedSpikes = self.groupSpikes(trialsPresented, uniqueTargAngle, spikeDatBinnedArr)

        targAvgList = [groupedSpikes[targ].trialAverage() for targ in range(0, len(groupedSpikes))]
        targTmTrcAvgArr = np.stack(targAvgList).view(BinnedSpikeSet)
        
        self.cosTuningCurveParams['targAvgNumSpks'] = targTmTrcAvgArr.timeAverage()
        targAvgNumSpks = self.cosTuningCurveParams['targAvgNumSpks']

        predictors = np.concatenate((np.ones_like(uniqueTargAngle), np.sin(uniqueTargAngle), np.cos(uniqueTargAngle)), axis=1)
        out = [np.linalg.lstsq(predictors, np.expand_dims(targAvgNumSpks[:, chan], axis=1), rcond=None)[0]for chan in range(targAvgNumSpks.shape[1])]
        
        paramsPerChan = np.stack(out).squeeze()
        self.cosTuningCurveParams['thBestPerChan'] = np.arctan2(paramsPerChan[:, 1], paramsPerChan[:, 2])
        self.cosTuningCurveParams['modPerChan'] = np.sqrt(pow(paramsPerChan[:,2], 2) + pow(paramsPerChan[:,1], 2))
        self.cosTuningCurveParams['bslnPerChan'] = paramsPerChan[:, 0]
        
        thBestPerChan = self.cosTuningCurveParams['thBestPerChan']
        modPerChan = self.cosTuningCurveParams['modPerChan']
        bslnPerChan = self.cosTuningCurveParams['bslnPerChan']
        
        angs = np.linspace(np.min(uniqueTargAngle), np.min(uniqueTargAngle)+2*np.pi)
        cosTuningCurves = np.stack([bslnPerChan[chan] + modPerChan[chan]*np.cos(thBestPerChan[chan]-angs) for chan in range(0, len(modPerChan))])
        
        self.cosTuningCurveParams['tuningCurves'] = cosTuningCurves
        
        return cosTuningCurves
    
    # note that they get grouped in the order of the given labels
    def groupSpikes(self, groupAssignment, groupLabels, binnedSpikes = None, notChan = None):
        if binnedSpikes is None:
            binnedSpikes = self.binSpikeData()
            
        groupedSpikes = [binnedSpikes[groupAssignment==i] for i in range(0, len(groupLabels))]
        
        return groupedSpikes
    
    # NOTE: this is a rough metric of trial start time that assumes no
    # intertrial interval... could definitely be improved, and might actually
    # require resaving all these Dataset instances with this as a property so it
    # can be correctly grabbed from raw data
    def trialStartTimeInSession(self, units='ms'):
        # this will output in ms because the maxTimestamps are saved in ms
        startTimeOfTrialsMs = np.cumsum(self.maxTimestamp)

        if units == 's':
            startTimeOfTrials = startTimeOfTrialsMs/1000
        else:
            startTimeOfTrials = startTimeOfTrialsMs

        return startTimeOfTrials
    
    def timeOfState(self, state, forRepeats = 'grabLongest', ignoreStates = []):
        stNm = self.stateNames
        if len(stNm)==1: # otherwise there's already a unique name per trial
            stNm = np.repeat(stNm, len(self.statesPresented), axis=0)
        
        stateTmPres = []
        if state == 'Start of trial':
            stateTmPres = [np.array([0])]*len(self.statesPresented)
        elif state == 'End of trial':
            for statesPres in self.statesPresented:
                stateTmPres.append(np.array([statesPres[1,-1]]))
        else:
            for stNames, statesPres in zip(stNm, self.statesPresented):
                statePresNum = np.nonzero(stNames == state)[0]
                if statePresNum.size>0:
                    statePresNum = statePresNum[0]
                    locStateLog = statesPres[0,:]==(statePresNum+1) # remember Matlab is 1-indexed
                else:
                    locStateLog = []

                if np.any(locStateLog):
                    # initialize a 0 length duration, so that we can start
                    # comparing and pick the *longest* state presentation
                    # (right?)
                    durationOfStatePres = -1 
                    stTimeUse = []
                    for locStateStrtSt in locStateLog.nonzero()[0]:
                        # locStateStrtSt = np.where(locStateLog)[0][0]
                        # delayStartSt = locStateStrtSt
                        locStateEndSt = locStateStrtSt + 1

                        stNumPres = int(statesPres[0, locStateEndSt] - 1) # go back to Python 0-indexing
                        # go to next state until we find one not to ignore
                        while stNames[stNumPres] in ignoreStates:
                            locStateEndSt = locStateEndSt + 1 
                            stNumPres = int(statesPres[0, locStateEndSt] - 1) # go back to Python 0-indexing

                        durationOfThisStatePres = statesPres[1, locStateEndSt] - statesPres[1,locStateStrtSt]

                        if forRepeats=='grabLongest':
                            if durationOfThisStatePres>durationOfStatePres:
                                durationOfStatePres = durationOfThisStatePres
                                # locStateStrtStChoose = locStateStrtSt
                                stTimeUse = statesPres[1, locStateStrtSt]
                        if forRepeats=='grabFirst':
                            stTimeUse = statesPres[1, locStateStrtSt]
                            break # can move on after the first has been found
                        elif forRepeats=='grabAll':
                            stTimeUse.append(statesPres[1,locStateStrtSt])

                    stateTmPres.append(np.array(stTimeUse))
                    # locState = np.where(locStateLog)[0][0]
                    # stateTmPres.append(statesPres[1, locState])
                else:
                    if forRepeats=='grabAll':
                        stateTmPres.append(np.array([np.nan]))
                    else:
                        stateTmPres.append(np.array(np.nan))
                
        return stateTmPres
            
    def computeStateStartAndEnd(self, stateName = 'Entire trial', ignoreStates = []):
        startTmsPres = []
        endTmsPres = []
        # the first two checks are to help out grabbing the start and end of
        # the entire trial; the last check is to grab the entire trial directly
        #
        # note that the first two are not super useful unless we have some time
        # chunk with unnamed neural data at the beginning, and we want to grab
        # that. Keeping here for the moment...
        stateNameStatesEnd = []
        stNm = self.stateNames
        if len(stNm)==1: # otherwise there's already a unique name per trial
            stNm = np.repeat(stNm, len(self.statesPresented), axis=0)

        if stateName == 'Start of trial':
            for stNames, statesPres in zip(stNm, self.statesPresented):
                startTmsPres.append(0)
                endTmsPres.append(statesPres[1,0])
                stateNamesStatesEnd.append(stNames[int(self.statesPres[1,1]-1)])
        elif stateName == 'End of trial':
            for trl, statesPres in enumerate(self.statesPresented):
                startTmsPres.append(statesPres[1,-1])
                endTmsPres.append(self.maxTimestamp[trl])
                stateNamesStatesEnd.append('End of trial')
        elif stateName == 'Entire trial':
            for trl, statePres in enumerate(self.statesPresented):
                startTmsPres.append(0)
                endTmsPres.append(self.maxTimestamp[trl]) # no true way to know how much longer it technically goes...
                stateNamesStatesEnd.append('End of trial')
        else:
            for stNames, statesPres in zip(stNm, self.statesPresented):
                # NOTE: a future warning comes up here if not all elements of
                # your state are strings! probably should fix this when you
                # first read in the dataset, or you could change this by first
                # running stNames = np.array([st[0].astype(str) for st in
                # stNames])... but I can't promise that'll behave correctly for
                # all formats of stNames since they're not... you know...
                # necessarily equivalent given where the dataset came from ;_;
                indStatePres = np.nonzero(stNames == stateName)[0]
                if indStatePres.size>0:
                    indStatePres = indStatePres[0]
                    locStateLog = statesPres[0,:]==(indStatePres+1) # remember Matlab is 1-indexed
                else:
                    locStateLog = []
                if np.any(locStateLog):
                    # initialize a 0 length duration, so that we can start
                    # comparing and pick the *longest* state presentation
                    # (right?)
                    durationOfStatePres = -1 
                    for locStateStrtSt in locStateLog.nonzero()[0]:
                        # locStateStrtSt = np.where(locStateLog)[0][0]
                        # delayStartSt = locStateStrtSt
                        locStateEndSt = locStateStrtSt + 1

                        stNumPres = int(statesPres[0, locStateEndSt] - 1) # go back to Python 0-indexing
                        # go to next state until we find one not to ignore
                        while stNames[stNumPres] in ignoreStates:
                            locStateEndSt = locStateEndSt + 1 
                            stNumPres = int(statesPres[0, locStateEndSt] - 1) # go back to Python 0-indexing

                        durationOfThisStatePres = statesPres[1, locStateEndSt] - statesPres[1,locStateStrtSt]

                        if durationOfThisStatePres>durationOfStatePres:
                            durationOfStatePres = durationOfThisStatePres
                            locStateStrtStChoose = locStateStrtSt
                            locStateEndStChoose = locStateEndSt
                            stNumPresChoose = stNumPres

                    stateNameStatesEnd.append(stNames[stNumPresChoose][0]) # result of how mats are loaded >.>

                    startTmsPres.append(statesPres[1, locStateStrtStChoose])
                    endTmsPres.append(statesPres[1, locStateEndStChoose])
                else:
                    startTmsPres.append(np.nan)
                    endTmsPres.append(np.nan)
#        startEndIndsPres = [ ]
        
        return startTmsPres, endTmsPres, stateNameStatesEnd 

    def computeDelayStartAndEnd(self, stateNameDelayStart = 'Delay Period', ignoreStates = []):
        startTmsPres, endTmsPres, stateNamesDelayEnd = self.computeStateStartAndEnd(stateName = stateNameDelayStart, ignoreStates = ignoreStates)
#        indDelayPres = np.where(self.stateNames == stateNameDelayStart)[1][0] # this is a horizontal vector...
#        startTmsPres = []
#        endTmsPres = []
#        stateNamesDelayEnd = []
#        for statesPres in self.statesPresented:
#            locDelayLog = statesPres[0,:]==(indDelayPres+1) # remember Matlab is 1-indexed
#            if np.any(locDelayLog):
#                locDelayStrtSt = np.where(locDelayLog)[0][0]
#                delayStartSt = locDelayStrtSt
#                locDelayEndSt = locDelayStrtSt + 1
#
#                stNumPres = int(statesPres[0, locDelayEndSt] - 1) # go back to Python 0-indexing
#                # go to next state until we find one not to ignore
#                while self.stateNames[0,stNumPres] in ignoreStates:
#                    locDelayEndSt = locDelayEndSt + 1 
#                    stNumPres = int(statesPres[0, locDelayEndSt] - 1) # go back to Python 0-indexing
#
#                stateNamesDelayEnd.append(self.stateNames[0,stNumPres][0]) # result of how mats are loaded >.>
#
#                startTmsPres.append(statesPres[1, delayStartSt])
#                endTmsPres.append(statesPres[1, locDelayEndSt])
#            else:
#                startTmsPres.append(np.nan)
#                endTmsPres.append(np.nan)
#        startEndIndsPres = [ ]
        
        return startTmsPres, endTmsPres, stateNamesDelayEnd 
    
#%% Plotting functions and such
    def plotTuningCurves(self, subpltRows = 5, subpltCols = 2):
        if self.cosTuningCurveParams['tuningCurves'].size == 0:
            self.computeCosTuningCurves()
        
        rowColCombos = range(1, subpltRows*subpltCols+1)
        #pw = plotWindow()
        uniqueTargAngle, trialsPresented = np.unique(self.markerTargAngles, axis=0, return_inverse=True)
        angs = np.linspace(np.min(uniqueTargAngle), np.min(uniqueTargAngle)+2*np.pi)
        for pltNum in range(len(self.cosTuningCurveParams['tuningCurves'])):
            if not pltNum % (subpltRows*subpltCols):
                plotInd = 0
                pltFig = plt.figure()
                #pw.addPlot('plot set ' + str(pltNum), pltFig)
            
            plt.subplot(subpltRows, subpltCols, rowColCombos[plotInd])
            plt.plot(angs, self.cosTuningCurveParams['tuningCurves'][pltNum, :])
            plt.plot(uniqueTargAngle, self.cosTuningCurveParams['targAvgNumSpks'][0:, pltNum], 'o')
            
            if plotInd>=((subpltRows*subpltCols) - 2):
                plt.xlabel('angle (rad)')
            elif not (plotInd % 2):
                plt.ylabel('average firing rate (Hz)')

            plotInd += 1
        # for that final plot
        plt.xlabel('angle (rad)')
    
    def plotKinematics(self, groupByTarget=True, statesStartEnd = ['Target Reach', 'Target Hold']):
        
        # startTmsPres, endTmsPres = self.computeDelayStartAndEnd()
        
        # tmMin = endTmsPres
        
        tmMin = self.timeOfState(statesStartEnd[0])
        tmMax = self.timeOfState(statesStartEnd[1])
        
        if self.kinematics is not None:
            if groupByTarget is True:
                angs, trlsPres = np.unique(self.markerTargAngles, return_inverse=True)
            else:
                angs = 0
                trlsPres = np.zeros_like(self.markerTargAngles)

            # mlab.figure()   
            plt.figure()
            # kinematPlot = plt.subplot(111, projection='3d')
            kinematPlot = plt.subplot(111)
            for idx, ang in enumerate(angs):
                for trlAng in np.where(trlsPres==idx)[0]:
                    if 'position' in self.kinematics[trlAng]:
                        kinsHere = self.kinematics[trlAng]['position'][0,0]
                        kinsHere = kinsHere-self.kinematicCenter
                        tmsHere = self.kinematics[trlAng]['time'][0,0].squeeze()
                        tmMinHere = tmMin[trlAng]
                        tmMaxHere = tmMax[trlAng]
                        ptsUse = np.logical_and(tmsHere>tmMinHere, tmsHere<tmMaxHere)
                        # mlab.plot3d(kinsHere[ptsUse,0], kinsHere[ptsUse,1], kinsHere[ptsUse,2], color = self.colorsetMayavi[idx])
                    else:
                        kinsHere = self.kinematics[trlAng][:, :2]
                        tmsHere = self.kinematics[trlAng][:, 2]
                        tmMinHere = tmMin[trlAng]
                        tmMaxHere = tmMax[trlAng]
                        ptsUse = np.logical_and(tmsHere>tmMinHere, tmsHere<tmMaxHere)
                        # mlab.plot3d(kinsHere[ptsUse,0], kinsHere[ptsUse,1], kinsHere[ptsUse,2], color = self.colorsetMayavi[idx])

                    kinematPlot.plot(kinsHere[ptsUse,0], kinsHere[ptsUse,1], color = self.colorset[idx])
                    
        else:
            print('no kinematics sorry!')

    def plotKeyStateInfo(self):
        from methods.plotUtils.ScatterBar import scatterBar
        statesToIgnore = self.metastates
        # statesToIgnore.append('Target Appear')
        # statesToIgnore.append('Target Reach')
        stateDurations = []
        binnedSpikesMn = []
        binnedSpikesSem = []
        for keyStateName, actualStateName in self.keyStates.items():
            if type(actualStateName) is list:
                startTmsPresTemp = []
                for stateNameHere in actualStateName:
                    startTmsPresTemp.append(self.timeOfState(state=stateNameHere, forRepeats='grabAll', ignoreStates = statesToIgnore))
                
                startTmsPres = [np.array(strtTms)[~np.isnan(strtTms)] for strtTms in zip(*startTmsPresTemp)]
            else:
                startTmsPres = self.timeOfState(state=actualStateName, forRepeats='grabAll', ignoreStates = statesToIgnore)

            if type(self.keyStateEnds[keyStateName]) is list:
                endTmsPresTemp = []
                for endStateNamesHere in self.keyStateEnds[keyStateName]:
                    endTmsPresTemp.append(self.timeOfState(state=endStateNamesHere, forRepeats='grabAll', ignoreStates = statesToIgnore))

                endTmsPres = [np.array(endTms)[~np.isnan(endTms)] for endTms in zip(*endTmsPresTemp)]
            else:
                endTmsPres = self.timeOfState(state=self.keyStateEnds[keyStateName], forRepeats='grabAll', ignoreStates = statesToIgnore)

            # this is making the assumption that the correct ordering of state
            # start and end will only happen once, and that the shortest of
            # those is the correct one
            allStateSeparations = [nd[None,:]-st[:,None] for st,nd in zip(startTmsPres, endTmsPres)]
            startIndsUse = np.asarray([np.meshgrid(np.arange(nd.shape[0]), np.arange(st.shape[0]))[1][stSep>0][np.nanargmin(stSep[stSep>0])] if np.any(stSep>0) else np.nan for st,nd,stSep in zip(startTmsPres, endTmsPres, allStateSeparations)])
            endIndsUse = np.asarray([np.meshgrid(np.arange(nd.shape[0]), np.arange(st.shape[0]))[0][stSep>0][np.nanargmin(stSep[stSep>0])] if np.any(stSep>0) else np.nan for st,nd,stSep in zip(startTmsPres, endTmsPres, allStateSeparations)])
            tmStState = np.asarray([st[int(stI)] if not np.isnan(stI) else np.nan for st,stI in zip(startTmsPres, startIndsUse)])
            tmEndState = np.asarray([nd[int(ndI)] if not np.isnan(ndI) else np.nan for nd,ndI in zip( endTmsPres, endIndsUse)])
            thisStateDurations = np.asarray([np.nanmin(stSep[stSep>0]) if np.any(stSep>0) else np.nan for stSep in allStateSeparations])

            stateDurations.append(thisStateDurations)
            
            binSizeMs = 50
            filterHasStateTrials = ~np.isnan(tmStState) & ~np.isnan(tmEndState)
            tmStState = tmStState[filterHasStateTrials]
            tmEndState = tmEndState[filterHasStateTrials]
            selfOnlyStateTrials = self.filterTrials(filterHasStateTrials)[0]
            timeBefore = -500
            timeAfter = 500
            breakpoint()
            binnedSpikesHere = selfOnlyStateTrials.binSpikeData(startMs = list(tmStState+timeBefore), endMs = list(tmStState+timeAfter+binSizeMs/20), binSizeMs=binSizeMs, alignmentPoints = list(zip(tmStState, tmStState)))
            binnedSpikesMn.append(binnedSpikesHere.trialAverage())
            binnedSpikesSem.append(binnedSpikesHere.trialSem())

        scatterXY, _ = scatterBar(stateDurations)
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, len(stateDurations))
        axDur = fig.add_subplot(gs[:3, :])
        durationList = np.arange(len(stateDurations))
        ptColors = plt.cm.Pastel2(durationList)
        scPtsInit = [axDur.scatter(mtrcWX[:,0], mtrcWX[:,1], color=col) for mtrcWX, col in zip(scatterXY.transpose(2,0,1), ptColors)]

        axDur.set_xticks(np.arange(len(stateDurations)))
        axDur.set_xticklabels(self.keyStates.keys())
        ylm = axDur.get_ylim()
        axDur.set_yticks(np.arange(ylm[1], step=100), minor=True)
        axDur.set_ylabel('duration')
        axDur.set_xlabel('key states')
        axDur.grid(axis='y', linestyle='--', linewidth=0.2, alpha=0.5, which='minor')
        axDur.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, which='major')

        for pltNum, mnTmcoursePerChan in enumerate(binnedSpikesMn):
            gspcTmcourse = gs[-1, pltNum]
            axTimecourse = fig.add_subplot(gspcTmcourse)
            timeBefore = -(mnTmcoursePerChan.alignmentBins[0]*binSizeMs)
            timeAfter = binSizeMs*(mnTmcoursePerChan.shape[1] - mnTmcoursePerChan.alignmentBins[0])
            mnTmcourse = np.nanmean(mnTmcoursePerChan, axis=0)
            tmsAll = np.arange(timeBefore, timeAfter, binSizeMs)
            axTimecourse.plot(tmsAll, mnTmcourse)
            semTmcourse = mnTmcoursePerChan.std(axis=0)/np.sqrt(mnTmcourse.shape[0])
            axTimecourse.fill_between(tmsAll, mnTmcourse - semTmcourse, mnTmcourse+semTmcourse, alpha=0.2)

            axTimecourse.set_xlim([timeBefore, timeAfter])
            axTimecourse.axvline(x=0, linestyle='--', color='k')
            


