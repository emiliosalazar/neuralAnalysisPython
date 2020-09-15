% for newly received memory guided saccade data (Pe and Wa)
% align and balance data content
% remove low mean FR neurons
% separate PFC data for each hemisphere
% take spikes counts into 20ms bins

addpath('~/Documents/PostdocData/memoryGuidedSaccade')
filenames = {'Pe180714_sort0.2.mat', 'Pe180719_sort0.2.mat', 'Wa180211_mgs.mat'};
allCodeStruct = trialCodes();
allCodeNames = fieldnames(allCodeStruct);
allCodeNums = cellfun(@(nm) allCodeStruct.(nm), allCodeNames, 'uni', 0);
codeNameByNum([allCodeNums{:}]) = allCodeNames;

for i = 1%1:length(filenames)
    % load(sprintf('./datasets/dirMem/%s', filenames{i}));
    load(filenames{i})

    areas = {'PFC1', 'PFC2'};
    binSize = 0.001;
    minFR = 0.5;

    % NOT DOING THIS vvvv INSTEAD GRABBING START (1) TO END (255) OF TRIAL
    % fprintf('align data and take the FIXATE~FIX_OFF window\n');
    % ^^^ this would be inputs 140 (fixate) to 3 (fix off) ^^^

    if i == 3
        [dat, sortedN] = prepCalibCounts(dat, 1, 255, binSize);
    else
        [dat, sortedN] = prepCalibCounts(testDat, 1, 255, binSize, 'sortflag', 0);
    end

    codeMatAll = cat(1, dat.trialcodes);
    uniqueCodes = unique(codeMatAll(:, 2));
    codeNamesUsed = codeNameByNum(uniqueCodes);
    codeNumNewInd(uniqueCodes) = 1:length(uniqueCodes);
    % this is the value we use for the end of the trial...
    codeNumNewEndDiff = codeNumNewInd;
    codeNumNewEndDiff(255) = -1;
    
    % numBins = min([dat.nBins]);
    % trialT = binSize * numBins;
    % unitMeanFRs = cell(1, 2);

    % NOTE: I DO NOT WANT TO DO THIS! I'LL LET MY PYTHON CODE DEAL
    % %%%%%%%%%%%%%%%%%% truncate trials to same length %%%%%%%%%%%%%%%%%%%%

    % fprintf('unify trial length\n');

    % numTrials = length(dat);

    % for iTrial = 1:numTrials
    %     dat(iTrial).counts = dat(iTrial).counts(:, 1:numBins);
    % end



    % %%%%%%%%%%%%%%%%%% separate data for dual PFC %%%%%%%%%%%%%%%%%%%%%

    pfc1Units = sortedN(sortedN(:, 2) <= 96, 1);
    pfc2Units = sortedN(sortedN(:, 2) > 96, 1);
    areaIndx = {pfc1Units', pfc2Units'};
    msPerSec = 1000;

    for iArea = 1:length(areas)

        fprintf('extracting data for %s\n', areas{iArea});

        S = [];

        numTrials = length(dat);
        for iTrial = 1:numTrials
            S(iTrial).stateNames = codeNamesUsed;
            startTime = dat(iTrial).trialcodes(1, 3);
            S(iTrial).status = any(strcmp(codeNamesUsed(codeNumNewInd(dat(iTrial).trialcodes(:, 2))), 'CORRECT'));
            S(iTrial).statesPresented = [codeNumNewEndDiff(dat(iTrial).trialcodes(:, 2)); msPerSec*(dat(iTrial).trialcodes(:, 3)-startTime)'];
            S(iTrial).spikes = dat(iTrial).counts(areaIndx{iArea}, :);
            S(iTrial).angle = dat(iTrial).angle;
            S(iTrial).distance = dat(iTrial).distance;
%             S(iTrial).counts = [];
        end
        
        preprocessDate = date; % this will be useful for determining if GPFA should be rerun...
        save(sprintf('ArrayNoSort%d_%s/processedData_%s.mat', iArea, areas{iArea}(1:end-1), areas{iArea}),'S', 'preprocessDate', '-v7.3');
        save(sprintf('ArrayNoSort%d_%s/processedData.mat', iArea, areas{iArea}(1:end-1)),'S', 'preprocessDate', '-v7.3');
    end

    % NOTE ALL THIS WILL BE DONE IN PYTHON 
    % %%%%%%%%%%%%%%%%% remove units with mean FR < minFR %%%%%%%%%%%%%%%%%%%

    % for iArea = 1:length(areas)

    %     fprintf('keeping units with mean firing rates >= %d spikes/sec for %s\n', ...
    %         minFR, areas{iArea});

    %     mean_FRs = [];
    %     load(sprintf('./processed_data/processed_mem/S_day%d_%s.mat', i, areas{iArea}));

    %     numTrials = length(S);
    %     for iTrial = 1:numTrials
    %         mean_FRs(:, iTrial) = sum(S(iTrial).spikes, 2)/trialT; % divide by trial length
    %     end

    %     keepUnits = mean(mean_FRs, 2) >= minFR;
    %     unitMeanFRs{iArea} = mean(mean_FRs, 2);

    %     for iTrial = 1:numTrials
    %         S(iTrial).spikes = S(iTrial).spikes(keepUnits, :);
    %     end

    %     save(sprintf('./processed_data/processed_mem/S_day%d_%s.mat', i, areas{iArea}), 'S', '-v7.3');
    % end

    % NOTE ALL THIS GETS DONE IN PYTHON %
    %%%%%%%%%%%%%%%%%% bin spikes into 20ms bins %%%%%%%%%%%%%%%%%%%%%%%%%%

%     for iArea = 1:length(areas)
%         fprintf('bin spikes into 20ms bins\n');
% 
%         load(sprintf('./processed_data/processed_mem/S_day%d_%s.mat', i, areas{iArea}));
% 
%         binSize = 20;
% 
%         for iTrial = 1:length(S)
%             S(iTrial).counts = bin_spikes(S(iTrial).spikes, binSize);
%         end
% 
%         save(sprintf('./processed_data/processed_mem/S_day%d_%s.mat', i, areas{iArea}), 'S', '-v7.3');
%     end
end
