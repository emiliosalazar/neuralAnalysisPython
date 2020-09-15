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

yr = '2019';
mnth = '03';
dy = '23';
animal = 'Earl_thresh';
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
% from NASNet import *
% netname = 'nas_PmdLincoln_NickSortsProbs'
% traindir = 'trainNas/pmdTrain'
% ntimepts = 30
% trainnet(netname,ntimepts,traindir)

%% run nas
gamma = 0.2; 
filename = 'testNas/Earl_thresh20190318_sessionWv_Sorts.mat';
filenameInf = 'testNas/Earl_thresh20190318_sessionInf_Sorts.mat';
% filename = 'testNas/Lincoln_NickMbSrt20100218_sessionWv_Sorts.mat';
% filenameInf = 'testNas/Lincoln_NickMbSrt20100218_sessionInf_Sorts.mat';
net_name = 'trainedNasPmdLinc/nas_PmdLincoln_NickSortsProbs';
[slabel,spikes, gammaWv] = runNASNet(filename,gamma,net_name);

%% plotting tested waveforms
close all
fileInf = load(filenameInf);
fileWv = load(filename);
numPltPer = 9;
wvsPlt = 200;
snInfAuto = slabel;
snInf = fileInf.snInfoMat;
snWv = fileWv.waveData';
if length(unique(snWv(1, :))) <= 2
    snWv = snWv(2:end, :);
end
rng(0);
wrstSort = 0;
for ch = 1:numPltPer:96
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
for hh =  1:length(hFigs)
    path = sprintf('earl190318_autoSortResAllLbG0-2_fig%d', hh);
    set(hFigs(hh), 'renderer', 'painters');
    orient(hFigs(hh), 'landscape');
    print(hFigs(hh),'-dpdf','-bestfit',path);
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
DataOrig = Data;
stInd = 1;
for trl = 1:length(Data)
    trlSnInf = Data(trl).TrialData.TDT.snippetInfo;
    endInd = stInd + size(trlSnInf, 2)-1;
    trlSnInf(2, :) = slabel(stInd:endInd);
    
    Data(trl).TrialData.TDT.snippetInfo = trlSnInf;
    
    stInd = endInd+1;
end