%%
% separate spike times of 1ms bins based on brain area
% take only the first flash in the task
% remove low firing rate units

areas = {'V4', 'PFC'};
conditions = {[1 1], [1 2], [2 1], [2 2]}; % [cue rfOri]
condNames = {'cue_1_rfOri_1', 'cue_1_rfOri_2', ...
              'cue_2_rfOri_1', 'cue_2_rfOri_2'};

filenames = {'Pe160202_s2ae_cuedAttention_0020_extractedStimuli.mat', ...
    'Pe160203_s3ae_cuedAttention_0020_extractedStimuli'};
for i = 2%1%:2
%     load(sprintf('./datasets/cuedA/%s', filenames{i}))
    load(sprintf('%s', filenames{i}))

    [code,chan] = find(sortcodeLookupTable');
    snrIdx = ismember(snr(:,1),chan) & ismember(snr(:,2),code);
    neuronSNR = snr(snrIdx,:);
    spikesV4 = setdiff(sortcodeLookupTable(1:96,:),0);
    numV4 = length(spikesV4);
    spikesPFC = setdiff(sortcodeLookupTable(97:end,:),0);
    numPFC = length(spikesPFC);
    areaIndx = {spikesV4, spikesPFC};
%     binWidth = 20;

   


    % take last flash of each sequence
    % set == 1 for first flash of each sequence
    % set == 2 or greater if you want the beginning of that delay...
    % set == [dat.sequenceLength] if you want the last trial before success
%     seqPosDesired = 2;%[dat.sequenceLength];
% 
%     origInd = num2cell(find([dat.sequencePosition] == seqPosDesired));
%     S = dat([dat.sequencePosition] == seqPosDesired);
%     [S.origInd] = deal(origInd{:});

    % Grabbing all the sequences here...
    seqPosDesired = 'all'; % this'll be the marker for all sequences
    origInd = num2cell(1:length(dat));
    S = dat;
    [S.origInd] = deal(origInd{:});

    save(sprintf('S_day%d.mat', i), 'S');


    % separate areas

    for iarea = 1:length(areas)

        S = [];
        data = load(sprintf('S_day%d.mat', i));

        timeStimStart = find(rasterTimeline==0);
        fullLengthStimPres = 400; % index positions; also ms
        
        
        for itrial = 1:length(data.S)
            S(itrial).stateNames = {'Blank Before','Blank Before Cut', 'Target', 'Blank After','Success', 'Failure BROKE_FIX',...
                'Failure BROKE_TARG', 'Failure NO_CHOICE', 'Failure FALSEALARM', 'Failure WRONG_TARG'};
            Stemp = data.S(itrial);
            xcSpks = [];
            S(itrial).trialResult = data.S(itrial).trialResult;
            if itrial==20
            a = 5;
            end
            if Stemp.sequencePosition ~= 1
                spikesTrialBefore = dat(Stemp.origInd-1).spikes;
                blankInterval = 1000*dat(Stemp.origInd-1).postStimInterval;
                if isnan(blankInterval)
                    warning('Previous position had no post-interval recorded, using current position''s preinterval instead...')
                    blankInterval = 1000*Stemp.preStimInterval;
                end
                overlapInterval = round(timeStimStart - (blankInterval-(1000-timeStimStart-fullLengthStimPres)));
                spikesEndOfLastTrial = spikesTrialBefore(:,(end - overlapInterval*2):end);
                spikesBeginThisTrial = data.S(itrial).spikes(:, 1:(overlapInterval*2));

                % this has to be done because the overlap is 15-30ms off what you'd expect
                % from just the postStimInterval + timeStimStart information; not sure why
                sumTest = [];
                for checkAlign = 1:overlapInterval*2
                    % looks like the last index might not be saved well in every trial...
                    sumTest(checkAlign) = all(all(spikesEndOfLastTrial(:, end-checkAlign+1:end-1) == spikesBeginThisTrial(:, 1:checkAlign-1)));
                end
                trueOverlap = find(sumTest, 1, 'last');
                
                spikesBlankBeforeLastTrial = spikesTrialBefore(:, timeStimStart+fullLengthStimPres:end);
                blankSpikesFromLastTrial = spikesBlankBeforeLastTrial(:, 1:end-trueOverlap);
                data.S(itrial).spikes =  [blankSpikesFromLastTrial data.S(itrial).spikes];
                
                S(itrial).statesPresented = [1 3 4; 1 timeStimStart+size(blankSpikesFromLastTrial,2) timeStimStart+size(blankSpikesFromLastTrial,2)+fullLengthStimPres];
            else
                S(itrial).statesPresented = [2 3 4; 1 timeStimStart timeStimStart+fullLengthStimPres];
            end
            
            if Stemp.sequencePosition ~= Stemp.sequenceLength % could also check that postStimInterval isn't nan
                spikesTrialAfter = dat(Stemp.origInd+1).spikes;
                blankInterval = 1000*Stemp.postStimInterval;
                if isnan(blankInterval)
                    warning('Current position had no post-interval recorded, using next position''s preinterval instead...')
                    blankInterval = 1000*dat(Stemp.origInd+1).preStimInterval;
                end
                overlapInterval = round(timeStimStart - (blankInterval-(1000-timeStimStart-fullLengthStimPres)));
                spikesBeginOfNextTrial = spikesTrialAfter(:,1:(overlapInterval*2));
                spikesEndThisTrial = data.S(itrial).spikes(:, end-(overlapInterval*2):end);

                % this has to be done because the overlap is 15-30ms off what you'd expect
                % from just the postStimInterval + timeStimStart information; not sure why
                sumTest = [];
                for checkAlign = 1:overlapInterval*2
                    sumTest(checkAlign) = all(all(spikesEndThisTrial(:, end-checkAlign+1:end) == spikesBeginOfNextTrial(:, 1:checkAlign)));
                end
                trueOverlap = find(sumTest, 1, 'last');
                
                spikesBlankBeforeNextTrial = spikesTrialAfter(:, 1:timeStimStart);%+fullLengthStimPres:end);
                blankSpikesFromNextTrial = spikesBlankBeforeNextTrial(:, trueOverlap+1:end);
                data.S(itrial).spikes =  [data.S(itrial).spikes blankSpikesFromNextTrial];
                S(itrial).statesPresented = [S(itrial).statesPresented [-1; size(data.S(itrial).spikes, 2)]];
                
                % we consider the trial successful if it moved to the next stage
                S(itrial).status = 1;
            else
                reactionTime = round(1000*data.S(itrial).trialRt);
                isCorrect = false;
                switch data.S(itrial).trialResult
                    case 'CORRECT'
                        S(itrial).statesPresented(:, end) = [find(strcmp(S(itrial).stateNames,  'Success')); S(itrial).statesPresented(2, end-1)+reactionTime];
                    case 'BROKE_FIX'
                        S(itrial).statesPresented(:, end) = [find(strcmp(S(itrial).stateNames,  'Failure BROKE_FIX')); S(itrial).statesPresented(2, end-1)+reactionTime];
                    case 'BROKE_TARG'
                        S(itrial).statesPresented(:, end) = [find(strcmp(S(itrial).stateNames,  'Failure BROKE_TARG')); S(itrial).statesPresented(2, end-1)+reactionTime];
                    case 'NO_CHOICE'
                        data.S(itrial).spikes(:, S(itrial).statesPresented(2, end)+1:end) = [];
                        S(itrial).statesPresented(1, end) = find(strcmp(S(itrial).stateNames,  'Failure NO_CHOICE'));
                    case 'FALSEALARM'
                        S(itrial).statesPresented(:, end) = [find(strcmp(S(itrial).stateNames,  'Failure FALSEALARM')); S(itrial).statesPresented(2, end-1)+reactionTime];
                    case 'WRONG_TARG'
                        S(itrial).statesPresented(:, end) = [find(strcmp(S(itrial).stateNames,  'Failure WRONG_TARG')); S(itrial).statesPresented(2, end-1)+reactionTime];
                end
                
               
                if strcmp(S(itrial).trialResult, 'CORRECT')
                    S(itrial).status = 1;
                elseif strcmp(S(itrial).trialResult, 'BROKE_FIX')
                    S(itrial).status = nan;
                else
                    S(itrial).status = 0;
                end
                
                % figure if there are more than 50ms with no spikes anywhere...
                % the trial end was reached... 
                zerosAtLeastLength = 50; 
                blockOfZeros = zeros(size(data.S(itrial).spikes, 1), zerosAtLeastLength);
                for endTrlInd = size(data.S(itrial).spikes, 2):-1:1
                    if ~all(all(data.S(itrial).spikes(:, endTrlInd-zerosAtLeastLength+1:endTrlInd) == blockOfZeros))
                        endOfZeroBlock = endTrlInd+1;
                        break
                    end
                end
                if endOfZeroBlock <= size(data.S(itrial).spikes, 2)
                    startOfZeroBlock = endOfZeroBlock-zerosAtLeastLength+1;
%                     a = figure;imagesc(data.S(itrial).spikes);hold on;
%                     plot([startOfZeroBlock startOfZeroBlock],[1 size(data.S(itrial).spikes, 1)], 'red')
%                     pause
%                     close(a)
                    data.S(itrial).spikes(:, startOfZeroBlock:end) = [];
                end
                S(itrial).statesPresented = [S(itrial).statesPresented [-1; size(data.S(itrial).spikes, 2)]];
            end
            
            S(itrial).spikes = data.S(itrial).spikes(areaIndx{iarea}, :);
            S(itrial).cue = data.S(itrial).cue;
            S(itrial).rfOri = data.S(itrial).rfOri;
            S(itrial).sequencePosition = Stemp.sequencePosition;
            S(itrial).sequenceLength = Stemp.sequenceLength;
        end

        if ischar(seqPosDesired) && strcmp(seqPosDesired, 'all')
            save(sprintf('Array%d_%s/processedData_allTrials_%s.mat', iarea, areas{iarea}, areas{iarea}), ...
                'S', '-v7.3');
            save(sprintf('Array%d_%s/processedData.mat', iarea, areas{iarea}), ...
                'S', '-v7.3');
        elseif length(seqPosDesired) == 1
            save(sprintf('Array%d_%s/processedData_flash%d_%s.mat', iarea, areas{iarea}, seqPosDesired, areas{iarea}), ...
                'S', '-v7.3');
%             save(sprintf('Array%d_%s/processedData.mat', iarea, areas{iarea}), ...
%                 'S', '-v7.3');
        else
            save(sprintf('Array%d_%s/processedData_flashLast_%s.mat', iarea, areas{iarea}, areas{iarea}), ...
                'S', '-v7.3');
        end
    end


    % ** I'm gonna do all this manipulation in Python... **
    % remove low FR units
%     minFR = 0; % spikes/sec
% 
%     for iarea = 1:length(areas)
% 
%         meanFRs = [];
%         load(sprintf('S_day%d_%s.mat', i, areas{iarea}));
% 
%         for itrial = 1:length(S)
%             T = size(S(itrial).spikes, 2)/1000; % second
%             meanFRs(:, itrial) = sum(S(itrial).spikes, 2)/T;
%         end
% 
%         keepUnits = mean(meanFRs, 2) >= minFR;
% 
%         for itrial = 1:length(S)
%             S(itrial).spikes = S(itrial).spikes(keepUnits, :);
%         end
% 
%         save(sprintf('S_day%d_%s.mat', i, areas{iarea}), ...
%             'S', '-v7.3');
%     end


    % bin spikes into 20ms bins

%     for iarea = 1:length(areas)
% 
%         load(sprintf('S_day%d_%s.mat', i, areas{iarea}));
% 
%         for itrial = 1:length(S)
%             S(itrial).counts = bin_spikes(S(itrial).spikes, binWidth);
%         end
% 
%         save(sprintf('S_day%d_%s.mat', i, areas{iarea}), ...
%             'S', '-v7.3');
%     end
end

