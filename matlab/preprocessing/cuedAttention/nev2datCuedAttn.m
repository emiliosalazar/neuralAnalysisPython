function [dat, sortcodeLookupTable, rasterTimeline] = nev2datCuedAttn(nevFile)
% using the cued attention nev file, spit out the dat file that Adam Snyder
% had created (and the format that I've been using) so I can play with it
% more. This allows me to run the cued attention data through the NAS and
% keep playing with it as I was before

% man I used to have a recursive step in here instead of this simple thing.
% How embarrassing.
if iscell(nevFile) % allow to put in multiple NEV files to concatenate
    dat = nev2dat(nevFile{1});
    for fl = 2:length(nevFile)
        datFl = nev2dat(nevFile{fl});
        
       
        dat = [dat, datFl];
    end
else
    dat = nev2dat(nevFile);
end

codesPres = {dat.trialcodes};
% hard coded, but this exists in a codes matfile 
stimOnCodes = [10:20]; 

% this checks whether the monkey broke fixation *after* the stimulus got
% presented, as these are the only trials we'll use
codeChanCol = 1; % although this will always be 0 for digital codes...
codeNumCol = 2;
codeTimeCol = 3;
useTrial = cellfun(@(cP) any(ismember(cP(:, codeNumCol), stimOnCodes)), codesPres);

datUse = dat(useTrial);

% grab spike times in ms (ceilinged up so everything from 0-1ms is assigned
% to 1 ms); also, I believe these are spikes detected--i.e. threshold
% crossings
sampRate = 30000; % 30 kHz sampling is just the way it is
firstSpikes = cat(1,datUse.firstspike) - sampRate*cat(1, datUse.time);
firstSpikeOffsetFromStart = firstSpikes(:, 1); % column two is offset from end...
spikeTimesInTrial = cellfun(@(spDiff, fSp) fSp+[0; cumsum(double(spDiff))], {datUse.spiketimesdiff}, num2cell(firstSpikeOffsetFromStart)', 'uni', 0);
% add 1/(2*sampRate) to get rid of zeros while not inappropriately bumping
% things to the next level...
spikeTimesInTrialMs = cellfun(@(spT) ceil((spT/sampRate + 1/(2*sampRate))*1000), spikeTimesInTrial, 'uni', 0);

% grab channels + sort numbers for detected spikes
spikeInfos = {datUse.spikeinfo};

% remove all spikes assigned 0 or 255 sort numbers
spikeTimesInTrialGoodSpikes = cellfun(@(spT, spI) spT(spI(:, 2) ~= 0 & spI(:, 2) ~= 255, :), spikeTimesInTrialMs,spikeInfos, 'uni', 0);
spikeInfosGoodSpikes = cellfun(@(spI) spI(spI(:, 2) ~= 0 & spI(:, 2) ~= 255, :), spikeInfos, 'uni', 0);

% find unique channel + sort combos, and assign them an index
allSpInfo = cat(1, datUse.spikeinfo);
uniqueChSortCombos = unique(allSpInfo, 'rows');
uniqueChSortCombos = uniqueChSortCombos(uniqueChSortCombos(:, 2)~=0 & uniqueChSortCombos(:, 2) ~= 255, :);
indexForCombo = 1:size(uniqueChSortCombos, 1);

% create the sort lookup table based on the indexes
sortcodeLookupTable = zeros(max(uniqueChSortCombos));
sortcodeLookupTable(sub2ind(max(uniqueChSortCombos), uniqueChSortCombos(:, 1), uniqueChSortCombos(:, 2))) = indexForCombo;


% create a # neuron x timepoint uint16 array with >0 where there's one (or
% more) spike(s)
rowOfSpikeAtTime = cellfun(@(spI) sortcodeLookupTable(sub2ind(size(sortcodeLookupTable), spI(:, 1),  spI(:, 2))), spikeInfosGoodSpikes, 'uni', 0);
spikeArrays = cellfun(@(spR, spT) uint16(accumarray([double(spR), double(spT)], ones(size(spR)))), rowOfSpikeAtTime, spikeTimesInTrialGoodSpikes, 'uni', 0);

% find code timing within a trial in the same ms alignment for spikes
dataUseCodes = {datUse.trialcodes};
codesInTrialsMsFromSt = cellfun(@(tc, tms) [tc(:, 2) ceil(1000*(tc(:, 3)-tms(1)))], dataUseCodes, {datUse.time}, 'uni', 0);
% Alright this is kinda shit, but code 18 messes stuff up (hopefully
% always?) so we're removing it...
codesInTrialsMsFromStHas18 = any(cellfun(@(codes) any(codes(:, 1) == 18), codesInTrialsMsFromSt));
if codesInTrialsMsFromStHas18
    disp('Removing code 18 because... it does weird things?')
end
codesInTrialsMsFromSt = cellfun(@(code) code(code(:, 1) ~= 18, :), codesInTrialsMsFromSt, 'uni', 0);

% ah, the joys of hardcoding... I wouldn't be surprised if only the first
% few codes are actually 'stim codes', but I've found that the STIM8_ON
% code (18) for some reason shouldn'be be counted here...
codesTarg = [10:17 19:20 70:80];
codesUsedTarg = cellfun(@(codes) codes(ismember(codes(:, 1), codesTarg), :), codesInTrialsMsFromSt, 'uni', 0);
codesUsedAfterTarg = cellfun(@(codes) codes([false; ismember(codes(:, 1), codesTarg)], :), codesInTrialsMsFromSt, 'uni', 0);
codesUsedBeforeTarg = cellfun(@(codes) codes(find(ismember(codes(:, 1), codesTarg))-1, :), codesInTrialsMsFromSt, 'uni', 0);
rasterTimeline = -300:700;
spkArraysAroundStimPreses = cellfun(@(spArr, cdUsedTarg) cellfun(@(codeTime) computeSpikeArrayAroundCode(spArr, codeTime, [rasterTimeline(1), rasterTimeline(end)]), num2cell(cdUsedTarg(:, 2)), 'uni', 0), spikeArrays, codesUsedTarg, 'uni', 0);
sequenceLength = cellfun('length', spkArraysAroundStimPreses)';

ex_codes; % better be on your path >.>
allExCodes = codes;
codeNums = struct2array(allExCodes);
codeNames = fieldnames(allExCodes);
cellWithNames(codeNums) = codeNames;
cellWithNames = cellWithNames';

priorCode = cellfun(@(cUBT) cellWithNames(cUBT(:, 1)), codesUsedBeforeTarg, 'uni', 0);
priorCode = cat(1, priorCode{:});
thisCode = cellfun(@(cUT) cellWithNames(cUT(:, 1)), codesUsedTarg, 'uni', 0);
thisCode = cat(1, thisCode{:});
nextCode = cellfun(@(cUAT) cellWithNames(cUAT(:, 1)), codesUsedAfterTarg, 'uni', 0)';
nextCode = cat(1, nextCode{:});


timeInTarg = cellfun(@(cUT, cUAT) cUAT(:, 2) - cUT(:, 2), codesUsedTarg, codesUsedAfterTarg, 'uni', 0);
% the post-target interval, as defined by Adam Snyder, is the time from the
% end of one target to the beginning of the other + nan for the last
% stimulus
postTargInterval = cellfun(@(cUT, cUAT) [cUT(2:end, 2) - cUAT(1:end-1, 2); nan], codesUsedTarg, codesUsedAfterTarg, 'uni', 0);
postTargInterval = cat(1, postTargInterval{:});
% the pre-target interval is just the post-target interval rotated one
% down...
preTargInterval = [nan; postTargInterval(1:end-1)];
preTargIntervalForNans = cellfun(@(cUBT, cUT) cUT(:, 2) - cUBT(:, 2), codesUsedBeforeTarg, codesUsedTarg, 'uni', 0);
preTargIntervalForNans = cat(1, preTargIntervalForNans{:});
startOfSeq = isnan(preTargInterval);
preTargInterval(startOfSeq) = preTargIntervalForNans(startOfSeq);

% divide by 1000 to reflect the fact that the original data has these in ms
postTargInterval = num2cell(postTargInterval/1000);
preTargInterval = num2cell(preTargInterval/1000);

% codes for actual saccade, fixation breaks, false alarms... and END_TRIAL
% for some trials where that signals the end... also FIX_OFF for the
% corner case described a bit below for BROKE_FIX...
codesSaccade = [141, 152, 156, 3, 255];
codesUsedSaccade = cellfun(@(codes) codes(ismember(codes(:, 1), codesSaccade), :), codesInTrialsMsFromSt, 'uni', 0);
% we'll just choose the time of the first one that comes up as the actual
% 'reaction time' (or time after last stimulus end) I guess...
codesUsedSaccade = cellfun(@(codes) codes(1, :), codesUsedSaccade, 'uni', 0);
lastStimOnToMovement = cellfun(@(cUT, cUS) cUS(2) - cUT(end, 2), codesUsedTarg, codesUsedSaccade, 'uni', 0);
lastStimOnToMovement = repelem(lastStimOnToMovement', sequenceLength);

spkArraysAroundStimSep = cat(1, spkArraysAroundStimPreses{:});
sequenceLengthRep = num2cell(repelem(sequenceLength, sequenceLength));
trialId = num2cell(1:length(sequenceLengthRep))';
sequenceId = num2cell(repelem(find(useTrial)', sequenceLength));
sequencePositionCell = cellfun(@(x) (1:x)', num2cell(sequenceLength), 'uni', 0);
sequencePosition = num2cell(cat(1, sequencePositionCell{:}));

% no clue why, but the field that gives the cue location is not 'cue', but
% rather 'thisTrialCue'...
stimInfo = cellfun(@(prm) [prm.trial.thisTrialCue, prm.trial.oriPick, prm.trial.isValid, prm.trial.posPick, prm.trial.targAmp], {datUse.params}, 'uni', 0);
stimInfo = cat(1, stimInfo{:});
rfOri = zeros(size(stimInfo, 1), 1);
% I *believe* that what this means is that the RF is actually at position
% 1. This means that if the cue is at position *2*, the orientation on the
% RF is different from the orientation at the cue (because the RF isn't at
% the cue!)
rfOri(stimInfo(:, 1)==2 & stimInfo(:, 3)) = ~(stimInfo(stimInfo(:, 1)==2 & stimInfo(:, 3), 2)-1)+1;
% On the other hand, if the cue is at position *1*, which *is* the RF, then
% the orienation at the cue and RF are identical
rfOri(stimInfo(:, 1)==1 & stimInfo(:, 3)) = stimInfo(stimInfo(:, 1)==1 & stimInfo(:, 3), 2);
% Of course all of the above is for 'valid' trials, which is to
% differentiate from the catch trials. On *those*, I think the "cue" is
% still encoded at its original position, but because they're catch, it's
% actually at the *other* position. As a result, if its recorded in
% position 2, it's actually in position *1* which is the RF's location, and
% the cue and RF orienation are the same. Vice versa if the cue is recorded
% at position 2. Note that all of this I'm taking from how they're related
% from already processed data as I don't have the code that processed the
% data originally >.>
rfOri(stimInfo(:, 1)==2 & ~stimInfo(:, 3)) = stimInfo(stimInfo(:, 1)==2 & ~stimInfo(:, 3), 2);
rfOri(stimInfo(:, 1)==1 & ~stimInfo(:, 3)) = ~(stimInfo(stimInfo(:, 1)==1 & ~stimInfo(:, 3), 2)-1)+1;
stimInfo = [stimInfo rfOri];
stimInfo = num2cell(stimInfo);
stimInfoRep = repelem(stimInfo, sequenceLength, 1);
stimInfoCellSplit = mat2cell(stimInfoRep, size(stimInfoRep, 1), ones(size(stimInfoRep, 2), 1));
[cue, oriPick, isValid, posPick, targAmp, rfOri] = stimInfoCellSplit{:};

codesResult = 150:157;
codesUsedResult = cellfun(@(codes) codes(ismember(codes(:, 1), codesResult), :), codesInTrialsMsFromSt, 'uni', 0);
resCode = cellfun(@(cUR) cellWithNames(cUR(:, 1)), codesUsedResult, 'uni', 0);
% there's apparently a corner case where fixation turns off after the
% target stimulus is presented... it appears Adam decided this was a
% BROKE_FIX result as a opposed to a NO_CHOICE result, but I'm not sure why
% (actually, it appears that the *system* decided this was BROKE_FIX as
% opposed to NO_CHOICE... I think it's just a corner case that happens
% rarely but sometimes)
resCode(cellfun('isempty', resCode)) = {{'BROKE_FIX'}};
resCode = cat(1, resCode{:});
trialResult = repelem(resCode, sequenceLength);

% note that 71:80 are the numbered targets
codesChoice = [120:129];
targChoice = [71:80];
codesUsedChoice = cellfun(@(codes) codes(ismember(codes(:, 1), codesChoice), :), codesInTrialsMsFromSt, 'uni', 0);
choiceCode = cellfun(@(cUC) cellWithNames(cUC(:, 1)), codesUsedChoice, 'uni', 0);
codesUsedTargChoice = cellfun(@(codes) codes(ismember(codes(:, 1), targChoice), :), codesInTrialsMsFromSt, 'uni', 0);
targCode = cellfun(@(cUC) cellWithNames(cUC(:, 1)), codesUsedTargChoice, 'uni', 0);
% so no choices are recorded when things are correct >.> but they're just
% the cue position (which is also apparently the cue *choice*... unless
% it's not a valid trial >.>)
cuePos = stimInfo(:, 1);
valStat = stimInfo(:, 3);
cueCorrValPos = cuePos(cellfun('isempty', choiceCode') & strcmp(resCode, 'CORRECT') & [valStat{:}]');
cueCorrNotValPos = cellfun(@(pos) ~(pos-1)+1, cuePos(cellfun('isempty', choiceCode') & strcmp(resCode, 'CORRECT') & ~[valStat{:}]'), 'uni', 0);
[choiceCode{cellfun('isempty', choiceCode) & strcmp(resCode', 'CORRECT') & [valStat{:}]}] = deal(cueCorrValPos{:});
[choiceCode{cellfun('isempty', choiceCode) & strcmp(resCode', 'CORRECT') & ~[valStat{:}]}] = deal(cueCorrNotValPos{:});
% also no choices are recorded when the wrong target is chosen, so I need
% to compute the choice based on what the correct choice should have
% been... which includes whether this was a valid trial >.>
cueWrongValPos = cellfun(@(pos) ~(pos-1)+1, cuePos(cellfun('isempty', choiceCode') & strcmp(resCode, 'WRONG_TARG') & [valStat{:}]'), 'uni', 0);
cueWrongNotValPos = cuePos(cellfun('isempty', choiceCode') & strcmp(resCode, 'WRONG_TARG') & ~[valStat{:}]');
[choiceCode{cellfun('isempty', choiceCode) & strcmp(resCode', 'WRONG_TARG') & [valStat{:}]}] = deal(cueWrongValPos{:});
[choiceCode{cellfun('isempty', choiceCode) & strcmp(resCode', 'WRONG_TARG') & ~[valStat{:}]}] = deal(cueWrongNotValPos{:});

choiceCode(cellfun('isempty', choiceCode)) = {0};
choiceCode = cat(1, choiceCode{:});
targCode(cellfun('isempty', targCode)) = {0};
targCode = cat(1, targCode{:});
choiceCode(cellfun(@ischar, targCode) & strcmp(resCode, 'CORRECT')) = cellfun(@(cdNm) str2double(cdNm(regexp(cdNm, '\d'))), targCode(cellfun(@ischar, targCode) & strcmp(resCode, 'CORRECT')), 'uni', 0 );
choiceCode(cellfun(@ischar, targCode) & strcmp(resCode, 'WRONG_TARG')) = cellfun(@(cdNm) ~(str2double(cdNm(regexp(cdNm, '\d')))-1)+1, targCode(cellfun(@ischar, targCode) & strcmp(resCode, 'WRONG_TARG')), 'uni', 0 );
choiceCode(cellfun(@ischar, choiceCode)) = cellfun(@(cdNm) str2double(cdNm(regexp(cdNm, '\d'))), choiceCode(cellfun(@ischar, choiceCode)), 'uni', 0 );
trialChoice = repelem(choiceCode, sequenceLength);

dat = struct('trialId', trialId,...
    'spikes', spkArraysAroundStimSep,...
    'sequenceId', sequenceId,...
    'sequencePosition', sequencePosition,...
    'sequenceLength', sequenceLengthRep,...
    ...'stimFrames',
    ...'trialHasMissedFrame',
    'cue', cue,...
    'oriPick', oriPick,...
    'isValid', isValid,...
    'rfOri', rfOri,...
    'posPick', posPick,...
    'targAmp', targAmp,...
    'thisCode', thisCode,...
    'preStimInterval', preTargInterval,...
    'postStimInterval', postTargInterval,...
    'nextCode', nextCode,...
    'trialResult', trialResult,...
    'trialChoice', trialChoice,...
    ... this guy's gonna take the place of trialRt... partly because I still
    ... don't know how trialRt was computed
    'lastStimOnToMovement', lastStimOnToMovement... % 'trialRt'
);

end

function spikeArrayAroundCode = computeSpikeArrayAroundCode(spikeArray, codeTimeIndex, indexOffshiftBeforeAndAfterCode)
indexStart = codeTimeIndex + indexOffshiftBeforeAndAfterCode(1);
indexEnd = codeTimeIndex + indexOffshiftBeforeAndAfterCode(2) ;

spikeArrayAroundCode = spikeArray(:, indexStart:min([indexEnd end]));
spikeArrayAroundCode(:, end+1:(indexEnd-indexStart+1)) = 0; % pad end if necessary

end
