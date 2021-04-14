%% saving all the waveforms in one matrix
mtFls = dir('*.mat');
outputMat = [];
snInfoMat = [];
for flNum = 1:length(mtFls)
    dt = load(mtFls(flNum).name, 'Data');
    dt = dt.Data;
    trlDt = [dt.TrialData];
    tdt = {trlDt.TDT};
    snps = cellfun(@(x) x.snippetInfo, tdt, 'uni', 0);
    snw = cellfun(@(x) x.snippetWaveforms, tdt, 'uni', 0);
    snInf = cat(2, snps{:});
    snWv = cat(2, snw{:});
    outMt = cat(1, snInf(2, :) ~= 0 & snInf(2, :) ~= 31, snWv);
    outputMat = cat(2, outputMat, outMt);
    snInfoMat = cat(2, snInfoMat, snInf);
end

waveData = outputMat';

yr = '2020';
mnth = '12';
dy = '04';
animal = 'Quincy';
save(sprintf('%s%s%s%s_sessionWv_Sorts.mat',animal, yr,mnth,dy), 'waveData', '-v7.3')
save(sprintf('%s%s%s%s_sessionInf_Sorts.mat',animal,yr,mnth,dy), 'snInfoMat', '-v7.3')

%% plotting waveforms original sorts
close all
numPltPer = 9;
wvsPlt = 200;
snInf = snInfoMat;
snWv = waveData(:, 2:end)';
rng(0);
for ch = 1:numPltPer:96
    figure;
    for subpl = 1:numPltPer
        subplot(3,3,subpl);
        indsWvs = find(snInf(1, :)==ch+subpl-1);
        indsWvs = indsWvs(randperm(length(indsWvs),min(length(indsWvs), wvsPlt)));
        wvUse = zeros(size(snInf(1, :)));
        wvUse(indsWvs) = true;
        if ch+subpl-1==57
            bl = indsWvs;
        end
        worstSortUsed = 1;
        plot(snWv(:, (snInf(2, :)<worstSortUsed | snInf(2, :)==31) & wvUse), 'color','k') % original unsorted
        hold on;
        plot(snWv(:,snInf(2, :)>=worstSortUsed & snInf(2, :)<31 & wvUse), 'color','g') % original sorted 
        title(ch+subpl-1)
    end
    fprintf('ch %d to %d plotted\n', ch, ch+numPltPer-1)
end

hFigs = get(0, 'children');
for hh =  1:length(hFigs)
    path = sprintf('Lincoln_NickMbSrt20100316_origSort_fig%d', hh);
    set(hFigs(hh), 'renderer', 'painters');
    orient(hFigs(hh), 'landscape');
    print(hFigs(hh),'-dpdf','-bestfit',path);
end


%% train nas (Python)
%from NASNet import *
%netname = 'nas_PmdLincolnNick_M1QuincyEmilio'
%traindir = 'trainNas/pmdAndM1Train'
%ntimepts = 30
%trainnet(netname,ntimepts,traindir)

%% run nas
%filename = 'testNas/Earl_thresh20190318_sessionWv_Sorts.mat';
%filenameInf = 'testNas/Earl_thresh20190318_sessionInf_Sorts.mat';
% filename = 'testNas/Lincoln_NickMbSrt20100218_sessionWv_Sorts.mat';
% filenameInf = 'testNas/Lincoln_NickMbSrt20100218_sessionInf_Sorts.mat';
% filename = 'Quincy_thresh20201020_sessionWv_Sorts.mat';
% filenameInf = 'Quincy_thresh20201020_sessionInf_Sorts.mat';
% filename = 'Quincy_thresh20201021_sessionWv_Sorts.mat';
% filenameInf = 'Quincy_thresh20201021_sessionInf_Sorts.mat';
% filename = 'Quincy_thresh20201022_sessionWv_Sorts.mat';
% filenameInf = 'Quincy_thresh20201022_sessionInf_Sorts.mat';
% filename = 'Quincy_thresh20201107_sessionWv_Sorts.mat';
% filenameInf = 'Quincy_thresh20201107_sessionInf_Sorts.mat';
% filename = 'Quincy_thresh20201124_sessionWv_Sorts.mat';
% filenameInf = 'Quincy_thresh20201124_sessionInf_Sorts.mat';
% filename = 'Quincy_EmilioSort20201020_sessionWv_Sorts.mat';
% filenameInf = 'wvInfo/Quincy_EmilioSort20201020_sessionInf_Sorts.mat';
% filename = 'Quincy20201202_sessionWv_Sorts.mat';
% filenameInf = 'Quincy20201202_sessionInf_Sorts.mat';
filename = 'Quincy20201204_sessionWv_Sorts.mat';
filenameInf = 'Quincy20201204_sessionInf_Sorts.mat';

gamma = 0.5; 
net_name = '~/../../project/nspg/esalazar/nasTests/trainedNasPmdLinc/nas_PmdLincoln_NickSortsProbs';
%net_name = '~/../../project/nspg/esalazar/nasTests/trainedNasPmdLincM1Quincy/nas_PmdLincolnNick_M1QuincyEmilio';
[slabel,spikes, gammaWv] = runNASNet(filename,gamma,net_name);

%% plotting tested waveforms
fileInf = load(filenameInf);
fileWv = load(filename);
close all
numPltPer = 9;
wvsPlt = 200;
gammaSel = 0.05; gamma;
snInfAuto = gammaWv > gammaSel;
snInf = fileInf.snInfoMat;
snWv = fileWv.waveData';
if length(unique(snWv(1, :))) <= 2
    snWv = snWv(2:end, :);
end
rng(0);
wrstSort = 0;
chMax = max(snInf(1, :))
for ch = 1:numPltPer:chMax
    figure;
    for subpl = 1:numPltPer
        subplot(3,3,subpl);
        indsWvs = find(snInf(1, :)==ch+subpl-1);
        indsWvs = indsWvs(randperm(length(indsWvs),min(length(indsWvs), wvsPlt)));
        wvUse = zeros(size(snInf(1, :)));
        wvUse(indsWvs) = true;
        plot(snWv(:, ~snInfAuto & (snInf(2, :)<=wrstSort | snInf(2, :)==31) & wvUse), 'color','k') % true negative
%         plot(snWv(:, snInf(2, :)==0 & snInf(2, :)==31 & wvUse), 'color','k') % original unsorted
        hold on;
%         plot(snWv(:,snInf(2, :)>0 & snInf(2, :)<31 & wvUse), 'color','g') % original sorted 
        plot(snWv(:, snInfAuto & (snInf(2, :)<=wrstSort | snInf(2, :)==31) & wvUse), 'color','b') % false positive
        plot(snWv(:, ~snInfAuto & (snInf(2, :)>wrstSort & snInf(2, :)<31) & snInf(2, :)<31 & wvUse), 'color','r') % false negative
%         plot(snWv(:, snInfAuto & wvUse), 'color','g') % sort positive
%         plot(snWv(:, ~snInfAuto & wvUse), 'color','k') % sort negative
        plot(snWv(:, snInfAuto & snInf(2, :)>wrstSort & snInf(2, :)<31 & wvUse), 'color','g') % true positive
        title(ch+subpl-1)
    end
    fprintf('ch %d to %d plotted\n', ch, ch+numPltPer-1)
end

hFigs = get(0, 'children');
fnmUnderscore = find(filename=='_');
flnmSt = filename(1:fnmUnderscore(2)-1);
for hh =  1:length(hFigs)
    path = sprintf('%s_autoSortResAllLbG0-%02d_fig%d', flnmSt, round(gammaSel*100), hh);
%    path = sprintf('test3', round(gammaSel*10), hh);
    set(hFigs(hh), 'renderer', 'painters');
    orient(hFigs(hh), 'landscape');
    print(hFigs(hh),'-dpdf','-bestfit',path);
    fprintf('fig %d of %d saved\n', hh, length(hFigs))
end

%% plotting waveforms grouped by gamma out
fileInf = load(filenameInf);
fileWv = load(filename);
close all
numPltPer = 9;
wvsPlt = 200;
snInfAuto = slabel;
snInf = fileInf.snInfoMat;
snWv = fileWv.waveData';
if length(unique(snWv(1, :))) <= 2
    snWv = snWv(2:end, :);
end
gmWvBins = 0:0.1:1; % bins of 0.1 width
rng(0);
wrstSort = 0;
cols = winter(length(gmWvBins)-1);
for ch = 1:numPltPer:96
    figure;
    for subpl = 1:numPltPer
        subplot(3,3,subpl);
        hold on
        for bin = 1:length(gmWvBins)-1
            binMin = gmWvBins(bin);
            binMax = gmWvBins(bin+1);
            wvGrp = snWv(:, gammaWv > binMin & gammaWv <= binMax & snInf(1, :)==ch+subpl-1);
            numWvs = min(size(wvGrp, 2), 30);
            wvGrp = wvGrp(:, randperm(size(wvGrp, 2), numWvs));
            plot(wvGrp, 'color', [cols(bin, :) 0.3]) 
        end
%         for bin = 1:length(gmWvBins)-1
%             binMin = gmWvBins(bin);
%             binMax = gmWvBins(bin+1);
%             wvGrp = snWv(:, gammaWv > binMin & gammaWv <= binMax & snInf(1, :)==ch+subpl-1);
%             wvMn = mean(wvGrp, 2);
%             plot(wvMn, 'color', cols(bin, :), 'linewidth', 2) 
%         end
        title(ch+subpl-1)
    end
    fprintf('ch %d to %d plotted\n', ch, ch+numPltPer-1)
end

hFigs = get(0, 'children');
for hh =  7%1:length(hFigs)
    path = sprintf('lincoln100218_grpAutSrt0-1binsAll_fig%d', hh);
    set(hFigs(hh), 'renderer', 'painters');
    orient(hFigs(hh), 'landscape');
    print(hFigs(hh),'-dpdf','-bestfit',path);
%     print(hFigs(hh),'-dpng',path);
end

%% relabel dataset
% We have loaded Data and loaded new labels in slabel
slabel = gammaWv > gammaUse; % gammaUse is the chosen gamma for this dataset!
% when there were 96 channels
% quincyM1Chans = [1:48, 73:2:87, 89:96];
% quincyPmdChans =  [49:72, 74:2:88];
% and when there were 128 channels (starting 11/18)
quincyM1Chans = [33:96];
quincyPmdChans =  [1:32 97:128];
DataOrig = Data;
stInd = 1;
for trl = 1:length(Data)
    trlSnInf = Data(trl).TrialData.TDT.snippetInfo;
    endInd = stInd + size(trlSnInf, 2)-1;
    trlSnInf(2, :) = slabel(stInd:endInd);
    
    Data(trl).TrialData.TDT.snippetInfo = trlSnInf;
    
    stInd = endInd+1;
end

% run preprocessData *which exists in the Batista SVN repo*
Data = preprocessData(Data);
DataM1 = Data;
DataPMd = Data;
m1Channels = quincyM1Chans;
pmdChannels = quincyPmdChans;
for trl = 1:length(Data)
    if isfield(Data(trl).TrialData.spikes, 'channel')
        DataM1(trl).TrialData.spikes = Data(trl).TrialData.spikes(ismember([Data(trl).TrialData.spikes.channel], m1Channels));
        DataPMd(trl).TrialData.spikes = Data(trl).TrialData.spikes(ismember([Data(trl).TrialData.spikes.channel], pmdChannels));
    end

end

processDate = datestr(now)
DataPre = Data;
Data = DataM1;save('Array1_M1/processedData.mat', 'Data', 'processDate')
Data = DataPMd;save('Array2_PMd/processedData.mat', 'Data', 'processDate')  


% gamma 0.5 10/20, 0.2 for next two days, and 0.5 for 11/07 on Quincy
